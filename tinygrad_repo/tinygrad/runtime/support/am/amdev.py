from __future__ import annotations
import ctypes, collections, time, dataclasses, functools, fcntl, os, hashlib
from tinygrad.helpers import mv_address, getenv, round_up, DEBUG, temp, fetch
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.amd import AMDReg, import_module, import_asic_regs
from tinygrad.runtime.support.allocator import TLSFAllocator
from tinygrad.runtime.support.am.ip import AM_SOC, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA

AM_DEBUG = getenv("AM_DEBUG", 0)

@dataclasses.dataclass(frozen=True)
class AMRegister(AMDReg):
  adev:AMDev

  def read(self): return self.adev.rreg(self.addr)
  def read_bitfields(self) -> dict[str, int]: return self.decode(self.read())

  def write(self, _am_val:int=0, **kwargs): self.adev.wreg(self.addr, _am_val | self.encode(**kwargs))

  def update(self, **kwargs): self.write(self.encode(**{**self.read_bitfields(), **kwargs}))

class AMFirmware:
  def __init__(self, adev):
    self.adev = adev
    def fmt_ver(hwip): return '_'.join(map(str, adev.ip_ver[hwip]))

    # Load SOS firmware
    self.sos_fw = {}

    blob, sos_hdr = self.load_fw(f"psp_{fmt_ver(am.MP0_HWIP)}_sos.bin", am.struct_psp_firmware_header_v2_0)
    fw_bin = sos_hdr.psp_fw_bin

    for fw_i in range(sos_hdr.psp_fw_bin_count):
      fw_bin_desc = am.struct_psp_fw_bin_desc.from_address(ctypes.addressof(fw_bin) + fw_i * ctypes.sizeof(am.struct_psp_fw_bin_desc))
      ucode_start_offset = fw_bin_desc.offset_bytes + sos_hdr.header.ucode_array_offset_bytes
      self.sos_fw[fw_bin_desc.fw_type] = blob[ucode_start_offset:ucode_start_offset+fw_bin_desc.size_bytes]

    # Load other fw
    self.ucode_start: dict[str, int] = {}
    self.descs: list[tuple[list[int], memoryview]] = []

    blob, hdr = self.load_fw(f"smu_{fmt_ver(am.MP1_HWIP)}.bin", am.struct_smc_firmware_header_v1_0)
    self.smu_psp_desc = self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.header.ucode_size_bytes, am.GFX_FW_TYPE_SMU)

    # SDMA firmware
    blob, hdr, hdr_v3 = self.load_fw(f"sdma_{fmt_ver(am.SDMA0_HWIP)}.bin", am.struct_sdma_firmware_header_v2_0, am.struct_sdma_firmware_header_v3_0)
    if hdr.header.header_version_major < 3:
      self.descs += [self.desc(blob, hdr.ctl_ucode_offset, hdr.ctl_ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH1)]
      self.descs += [self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.ctx_ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH0)]
    else: self.descs += [self.desc(blob, hdr_v3.header.ucode_array_offset_bytes, hdr_v3.ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH0)]

    # PFP, ME, MEC firmware
    for (fw_name, fw_cnt) in ([('PFP', 1), ('ME', 1)] if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else []) + [('MEC', 1)]:
      blob, hdr = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_{fw_name.lower()}.bin", am.struct_gfx_firmware_header_v2_0)

      # Code part
      self.descs += [self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.ucode_size_bytes, getattr(am, f'GFX_FW_TYPE_RS64_{fw_name}'))]

      # Stack
      stack_fws = [getattr(am, f'GFX_FW_TYPE_RS64_{fw_name}_P{fwnum}_STACK') for fwnum in range(fw_cnt)]
      self.descs += [self.desc(blob, hdr.data_offset_bytes, hdr.data_size_bytes, *stack_fws)]
      self.ucode_start[fw_name] = hdr.ucode_start_addr_lo | (hdr.ucode_start_addr_hi << 32)

    # IMU firmware
    blob, hdr = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_imu.bin", am.struct_imu_firmware_header_v1_0)
    imu_i_off, imu_i_sz, imu_d_sz = hdr.header.ucode_array_offset_bytes, hdr.imu_iram_ucode_size_bytes, hdr.imu_dram_ucode_size_bytes
    self.descs += [self.desc(blob, imu_i_off, imu_i_sz, am.GFX_FW_TYPE_IMU_I), self.desc(blob, imu_i_off + imu_i_sz, imu_d_sz, am.GFX_FW_TYPE_IMU_D)]

    # RLC firmware
    blob, hdr0, _hdr1, hdr2, hdr3 = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_rlc.bin", am.struct_rlc_firmware_header_v2_0,
      am.struct_rlc_firmware_header_v2_1, am.struct_rlc_firmware_header_v2_2, am.struct_rlc_firmware_header_v2_3)

    for mem,fmem in [('IRAM', 'iram'), ('DRAM_BOOT', 'dram')]:
      off, sz = getattr(hdr2, f'rlc_{fmem}_ucode_offset_bytes'), getattr(hdr2, f'rlc_{fmem}_ucode_size_bytes')
      self.descs += [self.desc(blob, off, sz, getattr(am, f'GFX_FW_TYPE_RLC_{mem}'))]

    if hdr0.header.header_version_minor == 3:
      for mem in ['P', 'V']:
        off, sz = getattr(hdr3, f'rlc{mem.lower()}_ucode_offset_bytes'), getattr(hdr3, f'rlc{mem.lower()}_ucode_size_bytes')
        self.descs += [self.desc(blob, off, sz, getattr(am, f'GFX_FW_TYPE_RLC_{mem}'))]

    self.descs += [self.desc(blob, hdr0.header.ucode_array_offset_bytes, hdr0.header.ucode_size_bytes, am.GFX_FW_TYPE_RLC_G)]

  def load_fw(self, fname:str, *headers):
    fpath = fetch(f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/45f59212aebd226c7630aff4b58598967c0c8c91/amdgpu/{fname}", subdir="fw")
    blob = memoryview(bytearray(fpath.read_bytes()))
    if AM_DEBUG >= 1: print(f"am {self.adev.devfmt}: loading firmware {fname}: {hashlib.sha256(blob).hexdigest()}")
    return tuple([blob] + [hdr.from_address(mv_address(blob)) for hdr in headers])

  def desc(self, blob:memoryview, offset:int, size:int, *types:int) -> tuple[list[int], memoryview]: return (list(types), blob[offset:offset+size])

@dataclasses.dataclass(frozen=True)
class AMMapping: va_addr:int; size:int; paddrs:list[tuple[int, int]]; uncached:bool=False; system:bool=False; snooped:bool=False # noqa: E702

class AMPageTableEntry:
  def __init__(self, adev, paddr, lv): self.adev, self.paddr, self.lv, self.entries = adev, paddr, lv, adev.vram.view(paddr, 0x1000, fmt='Q')

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, system=False, snooped=False, frag=0, valid=True):
    assert paddr & self.adev.gmc.address_space_mask == paddr, f"Invalid physical address {paddr:#x}"
    self.entries[entry_id] = self.adev.gmc.get_pte_flags(self.lv, table, frag, uncached, system, snooped, valid) | (paddr & 0x0000FFFFFFFFF000)

class AMPageTableTraverseContext:
  def __init__(self, adev, pt, vaddr, create_pts=False, free_pts=False, boot=False):
    self.adev, self.vaddr, self.create_pts, self.free_pts, self.boot = adev, vaddr - adev.gmc.vm_base, create_pts, free_pts, boot
    self.pt_stack:list[tuple[AMPageTableEntry, int, int]] = [(pt, self._pt_pte_idx(pt, vaddr), self._pt_pte_size(pt))]

  def _pt_pte_size(self, pt): return (1 << ((9 * (3-pt.lv)) + 12))
  def _pt_pte_idx(self, pt, va): return (va // self._pt_pte_size(pt)) % 512

  def level_down(self):
    pt, pte_idx, _ = self.pt_stack[-1]
    if (entry:=pt.entries[pte_idx]) & am.AMDGPU_PTE_VALID == 0:
      assert self.create_pts, "Not allowed to create new page table"
      pt.set_entry(pte_idx, self.adev.mm.palloc(0x1000, zero=True, boot=self.boot), table=True, valid=True)
      entry = pt.entries[pte_idx]

    assert not self.adev.gmc.is_pte_huge_page(entry), f"Must be table pt={pt.paddr:#x}, {pte_idx=} {entry=:#x}"
    child_page_table = AMPageTableEntry(self.adev, entry & 0x0000FFFFFFFFF000, lv=pt.lv+1)

    self.pt_stack.append((child_page_table, self._pt_pte_idx(child_page_table, self.vaddr), self._pt_pte_size(child_page_table)))
    return self.pt_stack[-1]

  def _try_free_pt(self) -> bool:
    pt, _, _ = self.pt_stack[-1]
    if self.free_pts and pt != self.adev.mm.root_page_table and all(pt.entries[i] & am.AMDGPU_PTE_VALID == 0 for i in range(512)):
      self.adev.mm.pfree(pt.paddr)
      parent_pt, parent_pte_idx, _ = self.pt_stack[-2]
      parent_pt.set_entry(parent_pte_idx, 0x0, valid=False)
      return True
    return False

  def level_up(self):
    while self._try_free_pt() or self.pt_stack[-1][1] == 512:
      _, pt_cnt, _ = self.pt_stack.pop()
      if pt_cnt == 512: self.pt_stack[-1] = (self.pt_stack[-1][0], self.pt_stack[-1][1] + 1, self.pt_stack[-1][2])

  def next(self, size:int, off=0):
    while size > 0:
      pt, pte_idx, pte_covers = self.pt_stack[-1]
      if self.create_pts:
        while pte_covers > size: pt, pte_idx, pte_covers = self.level_down()
      else:
        while pt.lv!=am.AMDGPU_VM_PTB and not self.adev.gmc.is_pte_huge_page(pt.entries[pte_idx]): pt, pte_idx, pte_covers = self.level_down()

      entries = min(size // pte_covers, 512 - pte_idx)
      assert entries > 0, "Invalid entries"
      yield off, pt, pte_idx, entries, pte_covers

      size, off, self.vaddr = size - entries * pte_covers, off + entries * pte_covers, self.vaddr + entries * pte_covers
      self.pt_stack[-1] = (pt, pte_idx + entries, pte_covers)
      self.level_up()

class AMMemoryManager:
  va_allocator = TLSFAllocator(512 * (1 << 30), base=0x7F0000000000) # global for all devices.

  def __init__(self, adev:AMDev, vram_size:int):
    self.adev, self.vram_size = adev, vram_size
    self.boot_allocator = TLSFAllocator(32 << 20, base=0) # per device
    self.pa_allocator = TLSFAllocator(vram_size - (64 << 20), base=self.boot_allocator.size) # per device
    self.root_page_table = AMPageTableEntry(self.adev, self.palloc(0x1000, zero=not self.adev.smi_dev, boot=True), lv=am.AMDGPU_VM_PDB1)

  def _frag_size(self, va, sz, must_cover=True):
    """
    Calculate the tlb fragment size for a given virtual address and size.
    If must_cover is True, the fragment size must cover the size, otherwise the biggest fragment size that fits the size is returned.
    Fragment 0 is 4KB, 1 is 8KB and so on.
    """
    va_pwr2_div, sz_pwr2_div, sz_pwr2_max = va & -(va) if va > 0 else (1 << 63), sz & -(sz), (1 << (sz.bit_length() - 1))
    return (min(va_pwr2_div, sz_pwr2_div) if must_cover else min(va_pwr2_div, sz_pwr2_max)).bit_length() - 1 - 12

  def map_range(self, vaddr:int, size:int, paddrs:list[tuple[int, int]], uncached=False, system=False, snooped=False, boot=False) -> AMMapping:
    if AM_DEBUG >= 2: print(f"am {self.adev.devfmt}: mapping {vaddr=:#x} ({size=:#x})")

    assert size == sum(p[1] for p in paddrs), f"Size mismatch {size=} {sum(p[1] for p in paddrs)=}"

    ctx = AMPageTableTraverseContext(self.adev, self.root_page_table, vaddr, create_pts=True, boot=boot)
    for paddr, psize in paddrs:
      for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(psize):
        for pte_off in range(pte_cnt):
          assert pt.entries[pte_idx + pte_off] & am.AMDGPU_PTE_VALID == 0, f"PTE already mapped: {pt.entries[pte_idx + pte_off]:#x}"
          pt.set_entry(pte_idx + pte_off, paddr + off + pte_off * pte_covers, uncached=uncached, system=system, snooped=snooped,
                       frag=self._frag_size(ctx.vaddr+off, pte_cnt * pte_covers), valid=True)

    # Invalidate TLB after mappings.
    self.adev.gmc.flush_tlb(ip='GC', vmid=0)
    self.adev.gmc.flush_tlb(ip='MM', vmid=0)
    return AMMapping(vaddr, size, paddrs, uncached=uncached, system=system, snooped=snooped)

  def unmap_range(self, vaddr:int, size:int):
    if AM_DEBUG >= 2: print(f"am {self.adev.devfmt}: unmapping {vaddr=:#x} ({size=:#x})")

    ctx = AMPageTableTraverseContext(self.adev, self.root_page_table, vaddr, free_pts=True)
    for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(size):
      for pte_id in range(pte_idx, pte_idx + pte_cnt):
        assert pt.entries[pte_id] & am.AMDGPU_PTE_VALID == am.AMDGPU_PTE_VALID, f"PTE not mapped: {pt.entries[pte_id]:#x}"
        pt.set_entry(pte_id, paddr=0x0, valid=False)

  @staticmethod
  def alloc_vaddr(size:int, align=0x1000) -> int: return AMMemoryManager.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))

  def valloc(self, size:int, align=0x1000, uncached=False, contigous=False) -> AMMapping:
    # Alloc physical memory and map it to the virtual address
    va = self.alloc_vaddr(size:=round_up(size, 0x1000), align)

    if contigous: paddrs = [(self.palloc(size, zero=True), size)]
    else:
      # Traverse the PT to find the largest contiguous sizes we need to allocate. Try to allocate the longest segment to reduce TLB pressure.
      paddrs = []
      ctx = AMPageTableTraverseContext(self.adev, self.root_page_table, va, create_pts=True)
      for off, _, _, seg_cnt, seg_size in ctx.next(size):
        rem_len = seg_cnt * seg_size
        while rem_len > 0:
          # Try to allocate as long segment (power of 2) as possible
          cont_seg_sz, paddr = 1 << (self._frag_size(ctx.vaddr+off, rem_len) + 12), None
          while cont_seg_sz >= 0x1000:
            try: paddr = self.palloc(cont_seg_sz, zero=False)
            except MemoryError: cont_seg_sz //= 2
            else: break

          if paddr is not None: paddrs += [(paddr, cont_seg_sz)]
          else:
            for paddr, _ in paddrs: self.pa_allocator.free(paddr)
            raise MemoryError(f"Failed to allocate a contiguous page. (allocation size={size:#x})")
          rem_len, off = rem_len - cont_seg_sz, off + cont_seg_sz

    return self.map_range(va, size, paddrs, uncached=uncached)

  def vfree(self, vm:AMMapping):
    self.unmap_range(vm.va_addr, vm.size)
    self.va_allocator.free(vm.va_addr)
    for paddr, _ in vm.paddrs: self.pa_allocator.free(paddr)

  def palloc(self, size:int, align:int=0x1000, zero=True, boot=False) -> int:
    assert self.adev.is_booting == boot, "During booting, only boot memory can be allocated"
    paddr = (self.boot_allocator if boot else self.pa_allocator).alloc(round_up(size, 0x1000), align)
    if zero: self.adev.vram[paddr:paddr+size] = bytes(size)
    return paddr

  def pfree(self, paddr:int): self.pa_allocator.free(paddr)

class AMDev:
  def __init__(self, devfmt, vram:MMIOInterface, doorbell:MMIOInterface, mmio:MMIOInterface, dma_regions:list[tuple[int, MMIOInterface]]|None=None):
    self.devfmt, self.vram, self.doorbell64, self.mmio, self.dma_regions = devfmt, vram, doorbell, mmio, dma_regions

    os.umask(0) # Set umask to 0 to allow creating files with 0666 permissions

    # Avoid O_CREAT because we don’t want to re-create/replace an existing file (triggers extra perms checks) when opening as non-owner.
    if os.path.exists(lock_name:=temp(f"am_{self.devfmt}.lock")): self.lock_fd = os.open(lock_name, os.O_RDWR)
    else: self.lock_fd = os.open(lock_name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)

    try: fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: raise RuntimeError(f"Failed to open AM device {self.devfmt}. It's already in use.")

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
    self.is_booting, self.smi_dev = True, False # During boot only boot memory can be allocated. This flag is to validate this.
    self.partial_boot = (self.reg("regSCRATCH_REG7").read() == (am_version:=0xA0000004)) and (getenv("AM_RESET", 0) != 1)

    # Memory manager & firmware
    self.mm = AMMemoryManager(self, self.vram_size)
    self.fw = AMFirmware(self)

    # Initialize IP blocks
    self.soc:AM_SOC = AM_SOC(self)
    self.gmc:AM_GMC = AM_GMC(self)
    self.ih:AM_IH = AM_IH(self)
    self.psp:AM_PSP = AM_PSP(self)
    self.smu:AM_SMU = AM_SMU(self)
    self.gfx:AM_GFX = AM_GFX(self)
    self.sdma:AM_SDMA = AM_SDMA(self)

    if self.partial_boot and (self.reg("regGCVM_CONTEXT0_CNTL").read() != 0):
      if DEBUG >= 2: print(f"am {self.devfmt}: MEC is active. Issue a full reset.")
      self.partial_boot = False

    # Init sw for all IP blocks
    for ip in [self.soc, self.gmc, self.ih, self.psp, self.smu, self.gfx, self.sdma]: ip.init_sw()

    # Init hw for IP blocks where it is needed
    if not self.partial_boot:
      if self.psp.is_sos_alive() and self.smu.is_smu_alive(): self.smu.mode1_reset()
      for ip in [self.soc, self.gmc, self.ih, self.psp, self.smu]:
        ip.init_hw()
        if DEBUG >= 2: print(f"am {self.devfmt}: {ip.__class__.__name__} initialized")

    # Booting done
    self.is_booting = False

    # Re-initialize main blocks
    for ip in [self.gfx, self.sdma]:
      ip.init_hw()
      if DEBUG >= 2: print(f"am {self.devfmt}: {ip.__class__.__name__} initialized")

    self.smu.set_clocks(level=-1) # last level, max perf.
    for ip in [self.soc, self.gfx]: ip.set_clockgating_state()
    self.reg("regSCRATCH_REG7").write(am_version)
    if DEBUG >= 2: print(f"am {self.devfmt}: boot done")

  def fini(self):
    if DEBUG >= 2: print(f"am {self.devfmt}: Finalizing")
    for ip in [self.sdma, self.gfx]: ip.fini_hw()
    self.smu.set_clocks(level=0)
    self.ih.interrupt_handler()
    os.close(self.lock_fd)

  def paddr2mc(self, paddr:int) -> int: return self.gmc.mc_base + paddr

  def reg(self, reg:str) -> AMRegister: return self.__dict__[reg]

  def rreg(self, reg:int) -> int:
    val = self.indirect_rreg(reg * 4) if reg > len(self.mmio) else self.mmio[reg]
    if AM_DEBUG >= 4 and getattr(self, '_prev_rreg', None) != (reg, val): print(f"am {self.devfmt}: Reading register {reg:#x} with value {val:#x}")
    self._prev_rreg = (reg, val)
    return val

  def wreg(self, reg:int, val:int):
    if AM_DEBUG >= 4: print(f"am {self.devfmt}: Writing register {reg:#x} with value {val:#x}")
    if reg > len(self.mmio): self.indirect_wreg(reg * 4, val)
    else: self.mmio[reg] = val

  def wreg_pair(self, reg_base:str, lo_suffix:str, hi_suffix:str, val:int):
    self.reg(f"{reg_base}{lo_suffix}").write(val & 0xffffffff)
    self.reg(f"{reg_base}{hi_suffix}").write(val >> 32)

  def indirect_rreg(self, reg:int) -> int:
    self.reg("regBIF_BX_PF0_RSMU_INDEX").write(reg)
    return self.reg("regBIF_BX_PF0_RSMU_DATA").read()

  def indirect_wreg(self, reg:int, val:int):
    self.reg("regBIF_BX_PF0_RSMU_INDEX").write(reg)
    self.reg("regBIF_BX_PF0_RSMU_DATA").write(val)

  def wait_reg(self, reg:AMRegister, value:int, mask=0xffffffff, timeout=10000) -> int:
    start_time = int(time.perf_counter() * 1000)
    while int(time.perf_counter() * 1000) - start_time < timeout:
      if ((rval:=reg.read()) & mask) == value: return rval
    raise RuntimeError(f'wait_reg timeout reg=0x{reg.addr:X} mask=0x{mask:X} value=0x{value:X} last_val=0x{rval}')

  def _run_discovery(self):
    # NOTE: Fixed register to query memory size without known ip bases to find the discovery table.
    #       The table is located at the end of VRAM - 64KB and is 10KB in size.
    mmRCC_CONFIG_MEMSIZE = 0xde3
    self.vram_size = self.rreg(mmRCC_CONFIG_MEMSIZE) << 20
    tmr_offset, tmr_size = self.vram_size - (64 << 10), (10 << 10)

    self.bhdr = am.struct_binary_header.from_buffer(bytearray(self.vram.view(tmr_offset, tmr_size)[:]))
    ihdr = am.struct_ip_discovery_header.from_address(ctypes.addressof(self.bhdr) + self.bhdr.table_list[am.IP_DISCOVERY].offset)
    assert ihdr.signature == am.DISCOVERY_TABLE_SIGNATURE and not ihdr.base_addr_64_bit, f"0x{ihdr.signature:X} != 0x{am.DISCOVERY_TABLE_SIGNATURE:X}"

    # Mapping of HW IP to Discovery HW IP
    hw_id_map = {am.__dict__[x]: int(y) for x,y in am.hw_id_map}
    self.regs_offset:dict[int, dict[int, tuple]] = collections.defaultdict(dict)
    self.ip_ver:dict[int, tuple[int, int, int]] = {}

    for num_die in range(ihdr.num_dies):
      dhdr = am.struct_die_header.from_address(ctypes.addressof(self.bhdr) + ihdr.die_info[num_die].die_offset)

      ip_offset = ctypes.addressof(self.bhdr) + ctypes.sizeof(dhdr) + ihdr.die_info[num_die].die_offset
      for _ in range(dhdr.num_ips):
        ip = am.struct_ip_v4.from_address(ip_offset)
        ba = (ctypes.c_uint32 * ip.num_base_address).from_address(ip_offset + 8)
        for hw_ip in range(1, am.MAX_HWIP):
          if hw_ip in hw_id_map and hw_id_map[hw_ip] == ip.hw_id:
            self.regs_offset[hw_ip][ip.instance_number] = tuple(list(ba))
            self.ip_ver[hw_ip] = (ip.major, ip.minor, ip.revision)

        ip_offset += 8 + (8 if ihdr.base_addr_64_bit else 4) * ip.num_base_address

    gc_info = am.struct_gc_info_v1_0.from_address(gc_addr:=ctypes.addressof(self.bhdr) + self.bhdr.table_list[am.GC].offset)
    self.gc_info = getattr(am, f"struct_gc_info_v{gc_info.header.version_major}_{gc_info.header.version_minor}").from_address(gc_addr)

  def _ip_module(self, prefix:str, hwip, prever_prefix:str=""): return import_module(prefix, self.ip_ver[hwip], prever_prefix)

  def _build_regs(self):
    mods = [("mp", am.MP0_HWIP), ("hdp", am.HDP_HWIP), ("gc", am.GC_HWIP), ("mmhub", am.MMHUB_HWIP), ("osssys", am.OSSSYS_HWIP),
      ("nbio" if self.ip_ver[am.GC_HWIP] < (12,0,0) else "nbif", am.NBIO_HWIP)]

    for prefix, hwip in mods:
      self.__dict__.update(import_asic_regs(prefix, self.ip_ver[hwip], cls=functools.partial(AMRegister, adev=self, bases=self.regs_offset[hwip][0])))
    self.__dict__.update(import_asic_regs('mp', (11, 0), cls=functools.partial(AMRegister, adev=self, bases=self.regs_offset[am.MP1_HWIP][0])))

