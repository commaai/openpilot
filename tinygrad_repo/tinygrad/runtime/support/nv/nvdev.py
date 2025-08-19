from __future__ import annotations
import ctypes, time, functools, re, gzip, struct
from tinygrad.helpers import getenv, DEBUG, fetch, getbits, to_mv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT, NV_GSP
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

  def _is_dual_pde(self) -> bool: return self.lv == self.nvdev.mm.level_cnt - 2

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, system=False, snooped=False, frag=0, valid=True):
    if not table:
      x = self.nvdev.pte_t.encode(valid=valid, address_sys=paddr >> 12, aperture=2 if system else 0, kind=6,
        **({'pcf': int(uncached)} if self.nvdev.mmu_ver == 3 else {'vol': uncached}))
    else:
      pde = self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t
      small, sys = ("_small" if self._is_dual_pde() else ""), "" if self.nvdev.mmu_ver == 3 else "_sys"
      x = pde.encode(is_pte=False, **{f'aperture{small}': 1 if valid else 0, f'address{small}{sys}': paddr >> 12},
        **({f'pcf{small}': 0b10} if self.nvdev.mmu_ver == 3 else {'no_ats': 1}))

    if self._is_dual_pde(): self.entries[2*entry_id], self.entries[2*entry_id+1] = x & 0xffffffffffffffff, x >> 64
    else: self.entries[entry_id] = x

  def entry(self, entry_id:int) -> int:
    return (self.entries[2*entry_id+1]<<64) | self.entries[2*entry_id] if self._is_dual_pde() else self.entries[entry_id]

  def read_fields(self, entry_id:int) -> dict:
    if self.is_huge_page(entry_id): return self.nvdev.pte_t.decode(self.entry(entry_id))
    return (self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t).decode(self.entry(entry_id))

  def is_huge_page(self, entry_id) -> bool: return (self.entry(entry_id) & 1 == 1) if self.lv < self.nvdev.mm.level_cnt - 1 else True
  def supports_huge_page(self, paddr:int): return self.lv >= self.nvdev.mm.level_cnt - 3 and paddr % self.nvdev.mm.pte_covers[self.lv] == 0

  def valid(self, entry_id):
    if self.is_huge_page(entry_id): return self.read_fields(entry_id)['valid']
    return self.read_fields(entry_id)['aperture_small' if self._is_dual_pde() else 'aperture'] != 0

  def address(self, entry_id:int) -> int:
    small, sys = ("_small" if self._is_dual_pde() else ""), "_sys" if self.nvdev.mmu_ver == 2 or self.lv == self.nvdev.mm.level_cnt - 1 else ""
    return self.read_fields(entry_id)[f'address{small}{sys}'] << 12

class NVMemoryManager(MemoryManager):
  va_allocator = TLSFAllocator((1 << 44), base=0x1000000000) # global for all devices.

  def on_range_mapped(self): self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write((1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

class NVDev(PCIDevImplBase):
  def __init__(self, devfmt:str, mmio:MMIOInterface, vram:MMIOInterface, venid:int, subvenid:int, rev:int, bars:dict):
    self.devfmt, self.mmio, self.vram, self.venid, self.subvenid, self.rev, self.bars = devfmt, mmio, vram, venid, subvenid, rev, bars
    self.lock_fd = System.flock_acquire(f"nv_{self.devfmt}.lock")

    self.smi_dev, self.is_booting = False, True
    self._early_init()

    # UVM depth   HW level                            VA bits
    # 0           PDE4                                56:56 (hopper+)
    # 1           PDE3                                55:47
    # 2           PDE2                                46:38
    # 3           PDE1 (or 512M PTE)                  37:29
    # 4           PDE0 (dual 64k/4k PDE, or 2M PTE)   28:21
    # 5           PTE_64K / PTE_4K                    20:16 / 20:12
    bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if self.mmu_ver == 3 else (48, [12, 21, 29, 38, 47])
    self.mm = NVMemoryManager(self, self.vram_size, boot_size=(2 << 20), pt_t=NVPageTableEntry, va_bits=bits, va_shifts=shifts, va_base=0,
      palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]])
    self.flcn:NV_FLCN|NV_FLCN_COT = NV_FLCN_COT(self) if self.fmc_boot else NV_FLCN(self)
    self.gsp:NV_GSP = NV_GSP(self)

    # Turn the booting early, gsp client is loaded from the clean.
    self.is_booting = False

    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()

  def fini(self):
    for ip in [self.gsp, self.flcn]: ip.fini_hw()

  def reg(self, reg:str) -> NVReg: return self.__dict__[reg]
  def wreg(self, addr:int, value:int):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr:int) -> int: return self.mmio[addr // 4]

  def _early_init(self):
    self.reg_names:set[str] = set()
    self.reg_offsets:dict[str, tuple[int, int]] = {}

    self.include("src/common/inc/swref/published/nv_ref.h")
    self.chip_id = self.reg("NV_PMC_BOOT_0").read()
    self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
    self.chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}[self.chip_details['architecture']] + f"{self.chip_details['implementation']:02d}"
    self.mmu_ver, self.fmc_boot = (3, True) if self.chip_details['architecture'] >= 0x1a else (2, False)

    self.include("src/common/inc/swref/published/turing/tu102/dev_fb.h")
    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0:
      if DEBUG >= 2: print(f"nv {self.devfmt}: WPR2 is up. Issuing a full reset.")
      System.pci_reset(self.devfmt)
      time.sleep(0.5)

    self.include("src/common/inc/swref/published/turing/tu102/dev_vm.h")
    self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island.h")
    self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island_addendum.h")

    # MMU Init
    self.reg_names.update(mmu_pd_names:=[f'NV_MMU_VER{self.mmu_ver}_PTE', f'NV_MMU_VER{self.mmu_ver}_PDE', f'NV_MMU_VER{self.mmu_ver}_DUAL_PDE'])
    for name in mmu_pd_names: self.__dict__[name] = NVReg(self, None, None, fields={})
    self.include(f"kernel-open/nvidia-uvm/hwref/{'hopper/gh100' if self.mmu_ver == 3 else 'turing/tu102'}/dev_mmu.h")
    self.pte_t, self.pde_t, self.dual_pde_t = tuple([self.__dict__[name] for name in mmu_pd_names])

    self.vram_size = self.reg("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42").read() << 20

  def _alloc_boot_struct(self, struct:ctypes.Structure) -> tuple[ctypes.Structure, int]:
    va, paddrs = System.alloc_sysmem(sz:=ctypes.sizeof(type(struct)), contiguous=True)
    to_mv(va, sz)[:] = bytes(struct)
    return type(struct).from_address(va), paddrs[0]

  def _download(self, file:str) -> str:
    url = f"https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/8ec351aeb96a93a4bb69ccc12a542bf8a8df2b6f/{file}"
    return fetch(url, subdir="defines").read_text()

  def extract_fw(self, file:str, dname:str) -> bytes:
    # Extracts the firmware binary from the given header
    tname = file.replace("kgsp", "kgspGet")
    text = self._download(f"src/nvidia/generated/g_bindata_{tname}_{self.chip_name}.c")
    info, sl = text[text[:text.index(dnm:=f'{file}_{self.chip_name}_{dname}')].rindex("COMPRESSION:"):][:16], text[text.index(dnm) + len(dnm) + 7:]
    image = bytes.fromhex(sl[:sl.find("};")].strip().replace("0x", "").replace(",", "").replace(" ", "").replace("\n", ""))
    return gzip.decompress(struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3) + image) if "COMPRESSION: YES" in info else image

  def include(self, file:str):
    regs_off = {'NV_PFALCON_FALCON': 0x0, 'NV_PGSP_FALCON': 0x0, 'NV_PSEC_FALCON': 0x0, 'NV_PRISCV_RISCV': 0x1000, 'NV_PGC6_AON': 0x0, 'NV_PFSP': 0x0,
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
