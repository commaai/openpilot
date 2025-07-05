from __future__ import annotations
import ctypes, time, functools, re, gzip, struct
from tinygrad.helpers import getenv, DEBUG, fetch, getbits, to_mv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP
from tinygrad.runtime.support.system import System, PCIDevImplBase

NV_DEBUG = getenv("NV_DEBUG", 0)

class NVReg:
  def __init__(self, nvdev, base, off, fields=None): self.nvdev, self.base, self.off, self.fields = nvdev, base, off, fields

  def __getitem__(self, idx:int): return NVReg(self.nvdev, self.base, self.off(idx), fields=self.fields)

  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)
  def with_base(self, base:int): return NVReg(self.nvdev, base + self.base, self.off, self.fields)

  def read(self): return self.nvdev.rreg(self.base + self.off)
  def read_bitfields(self) -> dict[str, int]: return self.decode(self.read())

  def write(self, _ini_val:int=0, **kwargs): self.nvdev.wreg(self.base + self.off, _ini_val | self.encode(**kwargs))

  def update(self, **kwargs): self.write(self.read() & ~self.mask(*kwargs.keys()), **kwargs)

  def mask(self, *names):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in names), 0)

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

class NVPageTableEntry:
  def __init__(self, nvdev, paddr, lv): self.nvdev, self.paddr, self.lv, self.entries = nvdev, paddr, lv, nvdev.vram.view(paddr, 0x1000, fmt='Q')

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, system=False, snooped=False, frag=0, valid=True):
    if not table:
      x = self.nvdev.NV_MMU_VER2_PTE.encode(valid=valid, address_sys=paddr >> 12, aperture=2 if system else 0, vol=uncached, kind=6)
    elif self.lv == 3:
      x = self.nvdev.NV_MMU_VER2_DUAL_PDE.encode(is_pte=False, address_small_sys=paddr >> 12, aperture_small=1 if valid else 0, vol_small=0, no_ats=1)
    else:
      x = self.nvdev.NV_MMU_VER2_PDE.encode(is_pte=False, address_sys=paddr >> 12, aperture=1 if valid else 0, vol=0, no_ats=1)

    if self.lv != 3: self.entries[entry_id] = x
    else:
      self.entries[2*entry_id] = x & 0xffffffffffffffff
      self.entries[2*entry_id+1] = x >> 64

  def entry(self, entry_id:int) -> int: return (self.entries[2*entry_id+1]<<64) | self.entries[2*entry_id] if self.lv == 3 else self.entries[entry_id]

  def read_fields(self, entry_id:int) -> dict:
    if self.is_pte(entry_id): return self.nvdev.NV_MMU_VER2_PTE.decode(self.entry(entry_id))
    return (self.nvdev.NV_MMU_VER2_DUAL_PDE if self.lv == 3 else self.nvdev.NV_MMU_VER2_PDE).decode(self.entry(entry_id))

  def is_pte(self, entry_id) -> bool: return (self.entry(entry_id) & 1 == 1) if self.lv <= 3 else True

  def valid(self, entry_id):
    if self.is_pte(entry_id): return self.read_fields(entry_id)['valid']
    return self.read_fields(entry_id)['aperture_small' if self.lv == 3 else 'aperture'] != 0

  def address(self, entry_id:int) -> int: return self.read_fields(entry_id)['address_small_sys' if self.lv == 3 else 'address_sys'] << 12

class NVMemoryManager(MemoryManager):
  va_allocator = TLSFAllocator((1 << 44), base=1 << 30) # global for all devices.

  def on_range_mapped(self): self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write((1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

class NVDev(PCIDevImplBase):
  def __init__(self, devfmt, mmio:MMIOInterface, vram:MMIOInterface, venid:int, subvenid:int, rev:int, bars:dict):
    self.devfmt, self.mmio, self.vram, self.venid, self.subvenid, self.rev, self.bars = devfmt, mmio, vram, venid, subvenid, rev, bars
    self.lock_fd = System.flock_acquire(f"nv_{self.devfmt}.lock")

    self.smi_dev, self.is_booting = False, True
    self._early_init()

    # UVM depth   HW level                            VA bits
    # 0           PDE3                                48:47
    # 1           PDE2                                46:38
    # 2           PDE1 (or 512M PTE)                  37:29
    # 3           PDE0 (dual 64k/4k PDE, or 2M PTE)   28:21
    # 4           PTE_64K / PTE_4K                    20:16 / 20:12
    self.mm = NVMemoryManager(self, self.vram_size, boot_size=(2 << 20), pt_t=NVPageTableEntry, pte_cnt=[4, 512, 512, 256, 512],
      pte_covers=[0x800000000000, 0x4000000000, 0x20000000, 0x200000, 0x1000], first_lv=0, first_page_lv=4, va_base=0)
    self.flcn:NV_FLCN = NV_FLCN(self)
    self.gsp:NV_GSP = NV_GSP(self)

    # Turn the booting early, gsp client is loaded from the clean.
    self.is_booting = False

    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()

  def reg(self, reg:str) -> NVReg: return self.__dict__[reg]
  def wreg(self, addr, value):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr): return self.mmio[addr // 4]

  def _early_init(self):
    self.reg_names:set[str] = set()
    self.reg_offsets:dict[str, tuple[int, int]] = {}

    self.include("src/common/inc/swref/published/nv_ref.h")
    self.chip_id = self.reg("NV_PMC_BOOT_0").read()
    self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
    self.chip_name = {0x17: "GA", 0x19: "AD"}[self.chip_details['architecture']] + str(100+self.chip_details['implementation'])

    self.include("src/common/inc/swref/published/turing/tu102/dev_fb.h")
    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0:
      if DEBUG >= 2: print(f"nv {self.devfmt}: WPR2 is up. Issuing a full reset.")
      System.pci_reset(self.devfmt)
      time.sleep(0.5)

    self.include("src/common/inc/swref/published/turing/tu102/dev_vm.h")
    self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island.h")
    self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island_addendum.h")

    # MMU Init
    self.reg_names.update(['NV_MMU_VER2_PTE', 'NV_MMU_VER2_PDE', 'NV_MMU_VER2_DUAL_PDE'])
    for name in ['NV_MMU_VER2_PTE', 'NV_MMU_VER2_PDE', 'NV_MMU_VER2_DUAL_PDE']: self.__dict__[name] = NVReg(self, None, None, fields={})
    self.include("kernel-open/nvidia-uvm/hwref/turing/tu102/dev_mmu.h")

    self.vram_size = self.reg("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42").read() << 20

  def _alloc_boot_struct(self, struct):
    va, paddrs = System.alloc_sysmem(sz:=ctypes.sizeof(type(struct)), contiguous=True)
    to_mv(va, sz)[:] = bytes(struct)
    return struct, paddrs[0]

  def _download(self, file) -> str:
    url = f"https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/e8113f665d936d9f30a6d508f3bacd1e148539be/{file}"
    return fetch(url, subdir="defines").read_text()

  def extract_fw(self, file:str, dname:str) -> bytes:
    # Extracts the firmware binary from the given header
    tname = file.replace("kgsp", "kgspGet")
    text = self._download(f"src/nvidia/generated/g_bindata_{tname}_{self.chip_name}.c")
    info, sl = text[text[:text.index(dnm:=f'{file}_{self.chip_name}_{dname}')].rindex("COMPRESSION:"):][:16], text[text.index(dnm) + len(dnm) + 7:]
    image = bytes.fromhex(sl[:sl.find("};")].strip().replace("0x", "").replace(",", "").replace(" ", "").replace("\n", ""))
    return gzip.decompress(struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3) + image) if "COMPRESSION: YES" in info else image

  def include(self, file:str):
    regs_off = {'NV_PFALCON_FALCON': 0x0, 'NV_PGSP_FALCON': 0x0, 'NV_PSEC_FALCON': 0x0, 'NV_PRISCV_RISCV': 0x1000, 'NV_PGC6_AON': 0x0,
      'NV_PGC6_BSI': 0x0, 'NV_PFALCON_FBIF': 0x600, 'NV_PFALCON2_FALCON': 0x1000, 'NV_PBUS': 0x0, 'NV_PFB': 0x0, 'NV_PMC': 0x0, 'NV_PGSP_QUEUE': 0x0,
      'NV_VIRTUAL_FUNCTION':0xb80000}

    for raw in self._download(file).splitlines():
      if not raw.startswith("#define "): continue

      if m:=re.match(r'#define\s+(\w+)\s+([0-9\+\-\*\(\)]+):([0-9\+\-\*\(\)]+)', raw): # bitfields
        name, hi, lo = m.groups()

        reg = next((r for r in self.reg_names if name.startswith(r+"_")), None)
        if reg is not None: self.__dict__[reg].add_field(name[len(reg)+1:].lower(), eval(lo), eval(hi))
        else: self.reg_offsets[name] = (eval(lo), eval(hi))
        continue

      if m:=re.match(r'#define\s+(\w+)\s*\(\s*(\w+)\s*\)\s*(.+)', raw): # reg set
        fn = m.groups()[2].strip().rstrip('\\').split('/*')[0].rstrip()
        name, value = m.groups()[0], eval(f"lambda {m.groups()[1]}: {fn}")
      elif m:=re.match(r'#define\s+(\w+)\s+([0-9A-Fa-fx]+)(?![^\n]*:)', raw): name, value = m.groups()[0], int(m.groups()[1], 0) # reg value
      else: continue

      reg_pref = next((prefix for prefix in regs_off.keys() if name.startswith(prefix)), None)
      not_already_reg = not any(name.startswith(r+"_") for r in self.reg_names)

      if reg_pref is not None and not_already_reg:
        fields = {k[len(name)+1:]: v for k, v in self.reg_offsets.items() if k.startswith(name+'_')}
        self.__dict__[name] = NVReg(self, regs_off[reg_pref], value, fields=fields)
        self.reg_names.add(name)
      else: self.__dict__[name] = value
