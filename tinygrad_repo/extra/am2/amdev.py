from __future__ import annotations
import ctypes, collections, dataclasses, functools, hashlib, array, time, struct
from tinygrad.helpers import mv_address, getenv, DEBUG, lo32, hi32, fetch_fw, to_mv, unwrap, to_function_name, wait_cond, to_name
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes
from tinygrad.device import Buffer, BufferSpec, Compiled
from tinygrad.engine.realize import run_linear, link_linear
from tinygrad.codegen.compile import lower_and_compile
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.autogen.am import am, fw
from tinygrad.runtime.support.amd import AMDReg, import_module, import_asic_regs
from tinygrad.runtime.support.memory import MemoryManager, AddrSpace, MMIOInterface
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.support.hcq2 import HCQInfo, lower_call
from tinygrad.runtime.support.usb import USB3, USBMMIOInterface, HOST_SIZE, usb_host
from extra.am2.ip import AM_SOC, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA, smu_is_smu_alive, am_reset, am_boot, am_fini, am_recover

AM_DEBUG = getenv("AM_DEBUG", 0)

@dataclasses.dataclass
class AMRegister(AMDReg):
  adev:AMDev

  def read(self, inst=0, direct=False): return self.adev.rreg(self.addr[inst], inst=inst, direct=direct)
  def read_bitfields(self, inst=0) -> dict[str, int]: return self.decode(self.read(inst=inst))

  def write(self, _am_val:int=0, inst=0, direct=False, **kwargs):
    self.adev.wreg(self.addr[inst], _am_val|self.encode(**kwargs), inst=inst, direct=direct)

  def update(self, inst=0, **kwargs): self.write(self.read(inst=inst) & ~self.fields_mask(*kwargs.keys()), inst=inst, **kwargs)

class AMFirmware:
  def __init__(self, adev):
    self.adev = adev
    def fmt_ver(hwip): return '_'.join(map(str, adev.ip_ver[hwip]))

    # Load SOS firmware
    self.sos_fw = {}

    blob, sos_hdr = self.load_fw(f"psp_{fmt_ver(am.MP0_HWIP)}_sos.bin", versioned_header='struct_psp_firmware_header')
    fw_bin = sos_hdr.psp_fw_bin

    for fw_i in range(sos_hdr.psp_aux_fw_bin_index if sos_hdr.header.header_version_minor == 1 else sos_hdr.psp_fw_bin_count):
      fw_bin_desc = am.struct_psp_fw_bin_desc.from_address(ctypes.addressof(fw_bin) + fw_i * ctypes.sizeof(am.struct_psp_fw_bin_desc))
      ucode_start_offset = fw_bin_desc.offset_bytes + sos_hdr.header.ucode_array_offset_bytes
      self.sos_fw[fw_bin_desc.fw_type] = blob[ucode_start_offset:ucode_start_offset+fw_bin_desc.size_bytes]

    # Load other fw
    self.ucode_start: dict[str, int] = {}
    self.descs: list[tuple[list[int], memoryview]] = []

    # SMU firmware
    if adev.ip_ver[am.MP1_HWIP] != (13,0,12):
      blob, hdr = self.load_fw(f"smu_{fmt_ver(am.MP1_HWIP)}.bin", versioned_header="struct_smc_firmware_header")
      if self.adev.ip_ver[am.GC_HWIP] >= (11,0,0):
        self.smu_psp_desc = self.desc(blob, hdr.v1_0.header.ucode_array_offset_bytes, hdr.v1_0.header.ucode_size_bytes, am.GFX_FW_TYPE_SMU)
      else:
        p2stables = (am.struct_smc_soft_pptable_entry * hdr.pptable_count).from_buffer(blob[hdr.pptable_entry_offset:])
        for p2stable in p2stables:
          if p2stable.id == (__P2S_TABLE_ID_X:=0x50325358):
            self.descs += [self.desc(blob, p2stable.ppt_offset_bytes, p2stable.ppt_size_bytes, am.GFX_FW_TYPE_P2S_TABLE)]

    # SDMA firmware
    blob, hdr = self.load_fw(f"sdma_{fmt_ver(am.SDMA0_HWIP)}.bin", versioned_header="struct_sdma_firmware_header")
    if hdr.header.header_version_major == 1:
      self.descs += [self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.header.ucode_size_bytes, am.GFX_FW_TYPE_SDMA0,
                               am.GFX_FW_TYPE_SDMA1, am.GFX_FW_TYPE_SDMA2, am.GFX_FW_TYPE_SDMA3)]
    elif hdr.header.header_version_major == 2:
      self.descs += [self.desc(blob, hdr.ctl_ucode_offset, hdr.ctl_ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH1)]
      self.descs += [self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.ctx_ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH0)]
    else: self.descs += [self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH0)]

    # PFP, ME, MEC firmware
    for (fw_name, fw_cnt) in ([('PFP', 1), ('ME', 1)] if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else []) + [('MEC', 1)]:
      blob, hdr = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_{fw_name.lower()}.bin", versioned_header="struct_gfx_firmware_header")

      ucode_off = hdr.header.ucode_array_offset_bytes
      if hdr.header.header_version_major == 1:
        # Code
        self.descs += [self.desc(blob, ucode_off, hdr.header.ucode_size_bytes - hdr.jt_size * 4, getattr(am, f'GFX_FW_TYPE_CP_{fw_name}'))]
        # JT
        self.descs += [self.desc(blob, ucode_off + hdr.jt_offset * 4, hdr.jt_size * 4, getattr(am, f'GFX_FW_TYPE_CP_{fw_name}_ME1'))]
      else:
        # Code
        self.descs += [self.desc(blob, ucode_off, hdr.ucode_size_bytes, getattr(am, f'GFX_FW_TYPE_RS64_{fw_name}'))]
        # Stack
        stack_fws = [getattr(am, f'GFX_FW_TYPE_RS64_{fw_name}_P{fwnum}_STACK') for fwnum in range(fw_cnt)]
        self.descs += [self.desc(blob, hdr.data_offset_bytes, hdr.data_size_bytes, *stack_fws)]
        self.ucode_start[fw_name] = hdr.ucode_start_addr_lo | (hdr.ucode_start_addr_hi << 32)

    # IMU firmware
    if self.adev.ip_ver[am.GC_HWIP] >= (11,0,0):
      blob, hdr = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_imu.bin", am.struct_imu_firmware_header_v1_0)
      imu_i_off, imu_i_sz, imu_d_sz = hdr.header.ucode_array_offset_bytes, hdr.imu_iram_ucode_size_bytes, hdr.imu_dram_ucode_size_bytes
      self.descs += [self.desc(blob, imu_i_off, imu_i_sz, am.GFX_FW_TYPE_IMU_I), self.desc(blob, imu_i_off+imu_i_sz, imu_d_sz, am.GFX_FW_TYPE_IMU_D)]

    # RLC firmware
    blob, hdr0, hdr1, hdr2, hdr3 = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_rlc.bin", am.struct_rlc_firmware_header_v2_0,
      am.struct_rlc_firmware_header_v2_1, am.struct_rlc_firmware_header_v2_2, am.struct_rlc_firmware_header_v2_3)

    if hdr0.header.header_version_minor == 1:
      for mem,fmem in [('LIST_SRM_CNTL', 'list_cntl'), ('LIST_GPM_MEM', 'list_gpm'), ('LIST_SRM_MEM', 'list_srm')]:
        off, sz = getattr(hdr1, f'save_restore_{fmem}_offset_bytes'), getattr(hdr1, f'save_restore_{fmem}_size_bytes')
        self.descs += [self.desc(blob, off, sz, getattr(am, f'GFX_FW_TYPE_RLC_RESTORE_{mem}'))]

    if hdr0.header.header_version_minor >= 2:
      for mem,fmem in [('IRAM', 'iram'), ('DRAM_BOOT', 'dram')]:
        off, sz = getattr(hdr2, f'rlc_{fmem}_ucode_offset_bytes'), getattr(hdr2, f'rlc_{fmem}_ucode_size_bytes')
        self.descs += [self.desc(blob, off, sz, getattr(am, f'GFX_FW_TYPE_RLC_{mem}'))]

    if hdr0.header.header_version_minor == 3:
      for mem in ['P', 'V']:
        off, sz = getattr(hdr3, f'rlc{mem.lower()}_ucode_offset_bytes'), getattr(hdr3, f'rlc{mem.lower()}_ucode_size_bytes')
        self.descs += [self.desc(blob, off, sz, getattr(am, f'GFX_FW_TYPE_RLC_{mem}'))]

    self.descs += [self.desc(blob, hdr0.header.ucode_array_offset_bytes, hdr0.header.ucode_size_bytes, am.GFX_FW_TYPE_RLC_G)]

  def load_fw(self, fname:str, *headers, versioned_header:str|None=None):
    blob = memoryview(bytearray(fetch_fw("amdgpu", fname, fw.hashes[fname])))
    if AM_DEBUG >= 1: print(f"am {self.adev.devfmt}: loading firmware {fname}: {hashlib.sha256(blob).hexdigest()}")
    if versioned_header:
      chdr = am.struct_common_firmware_header.from_address(mv_address(blob))
      headers += (getattr(am, versioned_header + f"_v{chdr.header_version_major}_{chdr.header_version_minor}"),)
    return tuple([blob] + [hdr.from_address(mv_address(blob)) for hdr in headers])

  def desc(self, blob:memoryview, offset:int, size:int, *types:int) -> tuple[list[int], memoryview]: return (list(types), blob[offset:offset+size])

class AMPageTableEntry:
  def __init__(self, adev, paddr, lv): self.adev, self.paddr, self.lv, self.entries = adev, paddr, lv, adev.vram.view(paddr, 0x1000, fmt='Q')

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True):
    is_sys = aspace is AddrSpace.SYS
    if aspace is AddrSpace.PHYS: paddr = self.adev.paddr2xgmi(paddr)
    assert paddr & self.adev.gmc.address_space_mask == paddr, f"Invalid physical address {paddr:#x}"
    self.entries[entry_id] = self.adev.gmc.get_pte_flags(self.lv, table, frag, uncached, is_sys, snooped, valid) | (paddr & 0x0000FFFFFFFFF000)

  def entry(self, entry_id:int) -> int: return self.entries[entry_id]
  def valid(self, entry_id:int) -> bool: return (self.entries[entry_id] & am.AMDGPU_PTE_VALID) != 0
  def address(self, entry_id:int) -> int:
    assert self.entries[entry_id] & am.AMDGPU_PTE_SYSTEM == 0, "should not be system address"
    return self.adev.xgmi2paddr(self.entries[entry_id] & 0x0000FFFFFFFFF000)
  def is_page(self, entry_id:int) -> bool: return self.lv == am.AMDGPU_VM_PTB or self.adev.gmc.is_pte_huge_page(self.lv, self.entries[entry_id])
  def supports_huge_page(self, paddr:int): return self.lv >= am.AMDGPU_VM_PDB2

class AMMemoryManager(MemoryManager):
  def on_range_mapped(self):
    # Invalidate TLB after mappings.
    self.dev.gmc.flush_tlb(ip='GC', vmid=0)
    self.dev.gmc.flush_tlb(ip='MM', vmid=0)

class AMDev:
  Version = 0xA000000D

  def _disable_aspm(self):
    # L1 across retimers makes reads oscillate to 0xffffffff; power on defaults it enabled. Clearing the GPU endpoint
    # alone suffices: L1 only engages when both ends of the link enable it.
    cap, seen = self.pci_dev.read_config(0x34, 1) & 0xfc, set() # bound the walk: a dead link can return 0xff pointers forever
    while cap and cap not in seen and self.pci_dev.read_config(cap, 1) != 0x10:
      seen.add(cap)
      cap = self.pci_dev.read_config(cap + 1, 1) & 0xfc
    if cap and cap not in seen: self.pci_dev.write_config_flush(cap + 0x10, self.pci_dev.read_config(cap + 0x10, 2) & ~3, 2) # PCIe cap lnkctl

  def __init__(self, pci_dev:PCIDevice, reset_mode=False):
    self.pci_dev, self.devfmt = pci_dev, pci_dev.pcibus
    pci_dev.reset_lock()
    self._disable_aspm()
    self.vram, self.doorbell64, self.mmio = self.pci_dev.map_bar(0), self.pci_dev.map_bar(2, fmt='Q'), self.pci_dev.map_bar(5, fmt='I')
    self.prog_init()

    self.is_vf = bool(self.mmio[am.mmRCC_IOV_FUNC_IDENTIFIER] & 1)
    assert not self.is_vf, "a VF is not supported"

    self._run_discovery()
    self._build_regs()

    # AM boot Process:
    # The GPU being passed can be in one of several states: 1. Not initialized. 2. Initialized by amdgpu. 3. Initialized by AM.
    # The 1st and 2nd states require a full GPU setup since their states are unknown. The 2nd state also requires a mode1 reset to
    # reinitialize all components.
    #
    # The 3rd state can be set up partially to optimize boot time. In this case, only the GFX and SDMA IPs need to be initialized.
    # To enable this, AM uses a separate boot memory that is guaranteed not to be overwritten. This physical memory is utilized for
    # all blocks that are initialized only during the initial AM boot.
    # To determine if the GPU is in the third state, AM uses regSCRATCH_REG7 as a flag.
    # To determine if the previous AM session finalized correctly, AM uses regSCRATCH_REG6 as a flag.
    self.is_booting = True # During boot only boot memory can be allocated. This flag is to validate this.
    self.init_sw(smi_dev=False)

    self.partial_boot = (self.reg("regSCRATCH_REG7").read() == AMDev.Version) and (getenv("AM_RESET", 0) != 1)
    if self.partial_boot and (self.reg("regSCRATCH_REG6").read() != 0 or self.reg(self.gmc.pf_status_reg("GC")).read() != 0):
      if DEBUG >= 2: print(f"am {self.devfmt}: Malformed state. Issuing a full reset.")
      self.partial_boot = False

    # aqua (gc 9.5.0): full boot over live state can kill the fabric (power cycle recovers); partial boot+reset_mec is the deepest safe reset
    if self.ip_ver[am.GC_HWIP] == (9,5,0) and self.reg("regSCRATCH_REG7").read() == AMDev.Version: self.partial_boot = True

    # Init hw for IP blocks where it is needed
    if not self.partial_boot:
      if self.psp.is_sos_alive() and self.run(smu_is_smu_alive)[1]:
        self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER, 2)
        if self.is_hive():
          if reset_mode: return # in reset mode, do not raise
          raise RuntimeError("Malformed state. Use extra/amdpci/hive_reset.py to reset the hive")
        self.run(am_reset)
        time.sleep(0.5) # 500ms
        # Config reads fail fast (0xffff) on a wedged gpu; mmio reads would hang until the root port
        # completion timeout, so check config space before touching mmio.
        wait_cond(self.pci_dev.read_config, 0, 2, value=0x1002, timeout_ms=2000,
          msg=f"am {self.devfmt}: gpu did not return from mode1 reset, reboot required")
      self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)

    self.psp.tmr_init(self.run(am_boot)[2])

    # Booting done
    self.is_booting = False

    pci_dev.reset_lock(False)
    if DEBUG >= 2: print(f"am {self.devfmt}: boot done")

  def init_sw(self, smi_dev=False):
    self.smi_dev, self.is_err_state = smi_dev, False

    # Memory manager & firmware
    self.mm = AMMemoryManager(self, self.vram_size - self.reserved_vram_size, boot_size=(3 << 20), pt_t=AMPageTableEntry, va_shifts=[12, 21, 30, 39],
      va_bits=48, first_lv=am.AMDGPU_VM_PDB2, va_base=MemoryManager.va_allocator.base, reserve_ptable=not self.large_bar,
      palloc_ranges=[(1 << (i + 12), (2 << 20) if i >= 9 else 0x1000) for i in range(9 * (3 - am.AMDGPU_VM_PDB2), -1, -1)])
    self.fw = AMFirmware(self)

    # Initialize IP blocks
    self.soc:AM_SOC = AM_SOC(self)
    self.gmc:AM_GMC = AM_GMC(self)
    self.ih:AM_IH = AM_IH(self)
    self.psp:AM_PSP = AM_PSP(self)
    self.smu:AM_SMU = AM_SMU(self)
    self.gfx:AM_GFX = AM_GFX(self)
    self.sdma:AM_SDMA = AM_SDMA(self)

    # Init sw for all IP blocks
    for ip in [self.soc, self.gmc, self.ih, self.psp, self.smu, self.gfx, self.sdma]: ip.init_sw()

  # *** the functions of ip.py that take the bus are uop functions: run as programs on the cpu the gpu is in, this one or the node's

  def prog_init(self):
    self.cpu = f"CPU:{peer[7:]}" if (peer:=getattr(self.pci_dev, 'peer_group', '')).startswith("remote:") else "CPU"
    self.tag, self.progs = to_function_name(f"am_{self.devfmt}").lower(), dict[tuple, tuple[UOp, list[Exception]]]()

    # the bus: what every access comes after, and the words of the programs ([5] the error, [8:] numbers for python)
    Compiled.link_bufs[name:=f"{self.tag}_bus"] = bus = Buffer(self.cpu, 8 * 16, options=BufferSpec(nolru=True), preallocate=True)
    self.bus, self.words = UOp.extern((16,), dtypes.uint64, self.cpu, name), bus.host.view(fmt='Q')

    # behind usb the bus is the link of the gpu's memory ([0] the handle, [1] the context): usb.py lowers the accesses of what is on its device
    if isinstance(self.mmio, USBMMIOInterface):
      words = [ctypes.addressof(x.contents) for x in (self.mmio.usb.usb.handle, USB3.ctx())]
      link = Buffer("CPU", HOST_SIZE, options=BufferSpec(nolru=True), initial_value=struct.pack('2Q', *words).ljust(HOST_SIZE, b'\0'))
      Compiled.link_bufs[to_name(dev:=f"USB:{self.tag}", "usb_host")] = link
      self.bus, self.words = usb_host(dev)[:8 * 16].bitcast(dtypes.uint64), link.host.view(0, 8 * 16, fmt='Q')
    self.rvram, self.rmmio = (self.mem(io) for io in (self.vram, self.mmio))

  def mem(self, io:MMIOInterface) -> UOp: # memory python mapped, as a program reaches it
    dev = f"USB:{self.tag}" if isinstance(io, USBMMIOInterface) else self.cpu
    Compiled.link_bufs[name:=f"{self.tag}_mem{io.addr:x}"] = Buffer(self.cpu, io.nbytes, options=BufferSpec(external_ptr=io.addr, nolru=True))
    return UOp.extern((io.nbytes,), dtypes.uint8, dev, name)

  def run(self, fn, *args, cache=False, **kwargs): # a uop function of ip.py as a program, now. gives the numbers it left for python
    if not cache or (prog:=self.progs.get(key:=(fn, args, *kwargs.items()))) is None:
      self.errors = list[Exception]() # what am_fail of this program can raise
      call = UOp.sink(fn(self.bus, self, *args, **kwargs), arg=KernelInfo(f"run_{fn.__name__}", optimized=True)).call(hcq=HCQInfo((self.cpu,)))
      prog = (link_linear(lower_and_compile(UOp(Ops.LINEAR, src=(unwrap(lower_call(call)),)))), self.errors)
      if cache: self.progs[key] = prog
    self.words[5] = 0
    run_linear(prog[0], jit=True)
    if (code:=self.words[5]) >> 63: raise RuntimeError(f"am {self.devfmt}: usb transfer failed with {code - (1 << 64)}") # a libusb error
    if code: raise prog[1][code - 1]
    return self.words.view(8 * 8)

  def fini(self):
    if DEBUG >= 2: print(f"am {self.devfmt}: Finalizing")
    self.run(am_fini)
    self.ih.interrupt_handler()
    self.reg("regSCRATCH_REG6").write(self.is_err_state) # set finalized state.

  def recover(self, force=False) -> bool:
    if not force and not self.is_err_state: return False
    if DEBUG >= 3: print(f"am {self.devfmt}: Start recovery")
    self.ih.interrupt_handler()
    self.run(am_recover)
    self.is_err_state = False
    if DEBUG >= 3: print(f"am {self.devfmt}: Recovery complete")
    return True

  # a hive has multiple XGMI regions; single-node parts (like MI350P) may still program LFB_SIZE with region 0 only
  def is_hive(self) -> bool: return self.gmc.xgmi_seg_sz > 0 and self.gmc.xgmi_max_region > 0

  def paddr2mc(self, paddr:int) -> int: return self.gmc.mc_base + paddr
  def paddr2xgmi(self, paddr:int) -> int: return self.gmc.paddr_base + paddr
  def xgmi2paddr(self, xgmi_paddr:int) -> int: return xgmi_paddr - self.gmc.paddr_base

  def reg(self, reg:str) -> AMRegister: return self.__dict__[reg]

  def rreg(self, reg:int, inst=0, direct=False) -> int:
    val = self.indirect_rreg(reg) if reg >= len(self.mmio) else self.mmio[reg]
    if AM_DEBUG >= 4 and getattr(self, '_prev_rreg', None) != (reg, val): print(f"am {self.devfmt}: Reading register {reg:#x} with value {val:#x}")
    self._prev_rreg = (reg, val)
    return val

  def wreg(self, reg:int, val:int, inst=0, direct=False):
    if AM_DEBUG >= 4: print(f"am {self.devfmt}: Writing register {reg:#x} with value {val:#x}")
    if reg >= len(self.mmio): self.indirect_wreg(reg, val)
    else: self.mmio[reg] = val

  def wreg_pair(self, reg_base:str, lo_suffix:str, hi_suffix:str, val:int, inst:int=0, direct=False):
    self.reg(f"{reg_base}{lo_suffix}").write(lo32(val), inst=inst, direct=direct)
    self.reg(f"{reg_base}{hi_suffix}").write(hi32(val), inst=inst, direct=direct)

  def indirect_rreg(self, reg:int) -> int:
    self.reg("regBIF_BX_PF0_RSMU_INDEX").write(reg * 4)
    return self.reg("regBIF_BX_PF0_RSMU_DATA").read()

  def indirect_wreg(self, reg:int, val:int):
    self.reg("regBIF_BX_PF0_RSMU_INDEX").write(reg * 4)
    self.reg("regBIF_BX_PF0_RSMU_DATA").write(val)

  def _read_vram(self, addr, size) -> bytes:
    assert addr % 4 == 0 and size % 4 == 0, f"Invalid address {addr:#x} or size {size:#x}"
    res = []
    for caddr in range(addr, addr + size, 4):
      self.wreg(0x06, caddr >> 31)
      self.wreg(0x00, (caddr & 0x7FFFFFFF) | 0x80000000)
      res.append(self.rreg(0x01))
    return bytes(array.array('I', res))

  def _run_discovery(self):
    # NOTE: Fixed register to query memory size without known ip bases to find the discovery table.
    #       The table is located at the end of VRAM - 64KB and is 10KB in size.
    mmRCC_CONFIG_MEMSIZE = 0xde3
    self.vram_size = self.rreg(mmRCC_CONFIG_MEMSIZE) << 20
    self.large_bar = self.vram.nbytes >= self.vram_size
    tmr_offset, tmr_size = self.vram_size - (64 << 10), (10 << 10)

    disc_tbl = self.vram.view(tmr_offset, tmr_size)[:] if self.large_bar else self._read_vram(tmr_offset, tmr_size)
    self.bhdr = am.struct_binary_header.from_buffer(bytearray(disc_tbl))
    ihdr = am.struct_ip_discovery_header.from_address(ctypes.addressof(self.bhdr) + self.bhdr.table_list[am.IP_DISCOVERY].offset)
    assert self.bhdr.binary_signature == am.BINARY_SIGNATURE and ihdr.signature == am.DISCOVERY_TABLE_SIGNATURE, "discovery signatures mismatch"

    self.regs_offset:dict[int, dict[int, tuple]] = collections.defaultdict(dict)
    self.ip_ver:dict[int, tuple[int, int, int]] = {}

    for num_die in range(ihdr.num_dies):
      dhdr = am.struct_die_header.from_address(ctypes.addressof(self.bhdr) + ihdr.die_info[num_die].die_offset)

      ip_offset = ctypes.addressof(self.bhdr) + ctypes.sizeof(dhdr) + ihdr.die_info[num_die].die_offset
      for _ in range(dhdr.num_ips):
        ip = am.struct_ip_v4.from_address(ip_offset)
        ba = ((ctypes.c_uint64 if ihdr.base_addr_64_bit else ctypes.c_uint32) * ip.num_base_address).from_address(ip_offset + 8)
        for hw_ip in range(1, am.MAX_HWIP):
          if hw_ip in am.hw_id_map and am.hw_id_map[hw_ip] == ip.hw_id:
            self.regs_offset[hw_ip][ip.instance_number] = tuple(list(ba))
            self.ip_ver[hw_ip] = (ip.major, ip.minor, ip.revision)

        ip_offset += 8 + (8 if ihdr.base_addr_64_bit else 4) * ip.num_base_address

    # HARV(EST) table: harvested instances must be excluded (like amdgpu_discovery_harvest_ip)
    # layout: u32 signature, u16 version, u16 size, then 32 entries of {hw_id:u16, inst:u8, rsv:u8}
    self.harvested:dict[int, set[int]] = collections.defaultdict(set)
    if (harv_off:=self.bhdr.table_list[am.HARVEST_INFO].offset) != 0 and \
       (blob:=to_mv(ctypes.addressof(self.bhdr) + harv_off, 8 + 32*4).cast('I'))[0] == am.HARVEST_TABLE_SIGNATURE:
      inv_hw_id = {hw_id: hw_ip for hw_ip, hw_id in am.hw_id_map.items()}
      for ent in blob[2:]:
        if (ip_:=inv_hw_id.get(ent & 0xffff)) is not None: self.harvested[ip_].add((ent >> 16) & 0xff)

    gc_info = am.struct_gc_info_v1_0.from_address(gc_addr:=ctypes.addressof(self.bhdr) + self.bhdr.table_list[am.GC].offset)
    self.gc_info = getattr(am, f"struct_gc_info_v{gc_info.header.version_major}_{gc_info.header.version_minor}").from_address(gc_addr)
    self.reserved_vram_size = (384 << 20) if self.ip_ver[am.GC_HWIP][:2] in {(9,4), (9,5)} else (64 << 20)

  @functools.cached_property
  def hwid_names(self) -> dict[int, str]: return {v:k.removesuffix('_HWID') for k,v in vars(am).items() if k.endswith('_HWID') and isinstance(v, int)}

  def _ip_module(self, prefix:str, hwip): return import_module(prefix, self.ip_ver[hwip])

  def _build_regs(self):
    mods = [("mp", am.MP0_HWIP), ("hdp", am.HDP_HWIP), ("gc", am.GC_HWIP), ("mmhub", am.MMHUB_HWIP), ("osssys", am.OSSSYS_HWIP),
      ("nbio" if self.ip_ver[am.GC_HWIP] < (12,0,0) else "nbif", am.NBIO_HWIP)]
    if self.ip_ver[am.SDMA0_HWIP] in {(4,4,2), (4,4,4)}: mods += [("sdma", am.SDMA0_HWIP)]

    for prefix, hwip in mods:
      regs = import_asic_regs(prefix, self.ip_ver[hwip], cls=functools.partial(AMRegister, adev=self, bases=self.regs_offset[hwip]))
      self.__dict__.update(regs)
    self.__dict__.update(import_asic_regs('mp', (11, 0, 0), cls=functools.partial(AMRegister, adev=self, bases=self.regs_offset[am.MP1_HWIP])))

    # Live AIDs like the kernel: 4 SDMAs per AID; the AID lives iff its group's alive-mask is 0xf/0x3/0xc.
    # Dead AIDs must never be touched via the indirect window: writes poison the whole fabric.
    live_sdma = {k for k in self.regs_offset[am.SDMA0_HWIP] if k not in self.harvested[am.SDMA0_HWIP]}
    max_aid = max((k >> 2 for k in self.regs_offset[am.SDMA0_HWIP]), default=0)
    self.aids = [0] + [aid for aid in range(1, max_aid + 1) if sum(1 << (i & 3) for i in live_sdma if i >> 2 == aid) in {0xf, 0x3, 0xc}]
