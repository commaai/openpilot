# mypy: ignore-errors
from __future__ import annotations
import ctypes, ctypes.util, struct, functools, os, mmap
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.amd import AMDReg, import_asic_regs
from test.mockgpu.amd.amdgpu import AMDGPU

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

VRAM_SIZE = 512 << 20

IP_VERSIONS = {
  am.GC_HWIP: (12, 0, 0), am.SDMA0_HWIP: (7, 0, 0), am.MMHUB_HWIP: (4, 1, 0), am.NBIO_HWIP: (6, 3, 1),
  am.MP0_HWIP: (14, 0, 2), am.MP1_HWIP: (14, 0, 2), am.HDP_HWIP: (7, 0, 0), am.OSSSYS_HWIP: (7, 0, 0),
}

def _pad(t, n=10): return t + (0,) * (n - len(t))
IP_BASES = {
  am.GC_HWIP:     _pad((0x00001260, 0x0000A000, 0x0001C000, 0x02402C00)),
  am.SDMA0_HWIP:  _pad((0x00001260, 0x0000A000, 0x0001C000, 0x02402C00)),
  am.MMHUB_HWIP:  _pad((0x0001A000, 0x02408800)),
  am.NBIO_HWIP:   _pad((0x00000000, 0x00000014, 0x00000D20, 0x00010400, 0x0241B000, 0x04040000)),
  am.MP0_HWIP:    _pad((0x00016000, 0x00DC0000, 0x00E00000, 0x00E40000, 0x0243FC00)),
  am.MP1_HWIP:    _pad((0x00016000, 0x00DC0000, 0x00E00000, 0x00E40000, 0x0243FC00)),
  am.HDP_HWIP:    _pad((0x00000F20, 0x0240A400)),
  am.OSSSYS_HWIP: _pad((0x000010A0, 0x0240A000)),
}

IP_HWIDS = {hwip: am.hw_id_map[hwip] for hwip in IP_VERSIONS}

GC_INFO = dict(gc_num_se=2, gc_num_cu_per_sh=8, gc_num_sh_per_se=2, gc_num_rb_per_se=4,
               gc_num_tccs=8, gc_wave_size=32, gc_max_waves_per_simd=16, gc_max_scratch_slots_per_cu=32, gc_lds_size=64)

def _build_ip_regs(prefix, hwip) -> dict[str, AMDReg]:
  try: return import_asic_regs(prefix, IP_VERSIONS[hwip], cls=functools.partial(AMDReg, bases={0: IP_BASES[hwip]}))
  except Exception: return {}

class MockMMU:
  def __init__(self, gpu:MockAMGPU):
    self.gpu = gpu
    self.tlb: dict[int, tuple[int, int, bool]] = {}

  def invalidate(self, pt_base:int, va_base:int):
    new_tlb: dict[int, tuple[int, int, bool]] = {}
    self._walk(pt_base, 0, 0, new_tlb, va_base)
    for va, (pa, sz, is_sys) in new_tlb.items():
      old = self.tlb.get(va)
      if not is_sys and (old is None or old[0] != pa): self.gpu.map_vram_at(va, pa, sz)
      if old is None: self.gpu.map_range(va, sz)
    self.tlb = new_tlb

  def _walk(self, pt_paddr:int, level:int, va_acc:int, out:dict, va_base:int):
    shift = [39, 30, 21, 12][level]
    for i in range(512):
      pte = struct.unpack_from('<Q', self.gpu.vram, pt_paddr + i * 8)[0]
      if not (pte & am.AMDGPU_PTE_VALID): continue
      va, pa = va_acc | (i << shift), pte & 0x0000FFFFFFFFF000
      if level == 3 or (pte & am.AMDGPU_PDE_PTE_GFX12):
        out[va_base + va] = (pa, 1 << shift, bool(pte & am.AMDGPU_PTE_SYSTEM))
      else:
        self._walk(pa, level + 1, va, out, va_base)

  def paddr_to_host(self, paddr:int) -> int:
    page, off = paddr & ~0xFFF, paddr & 0xFFF
    if page in self.gpu._sysmem_map: return self.gpu._sysmem_map[page] + off
    if paddr < VRAM_SIZE: return self.gpu.vram_addr + paddr
    raise ValueError(f"paddr {paddr:#x} not found in sysmem_map or VRAM")

  def addr_to_host(self, addr:int) -> int:
    gmc = self.gpu.mmio.gmc
    sys_lo = self.gpu.mmio.regs.get(gmc.reg('regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR') or 0, 0) << 18
    sys_hi = self.gpu.mmio.regs.get(gmc.reg('regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR') or 0, 0) << 18
    if sys_lo <= addr < sys_hi: return self.paddr_to_host(addr - self.gpu.mc_base)
    for tva, (pa, sz, is_sys) in self.tlb.items():
      if tva <= addr < tva + sz:
        paddr = pa + (addr - tva)
        if not is_sys: return self.gpu.vram_addr + paddr
        return self.paddr_to_host(paddr)
    raise ValueError(f"addr {addr:#x} not mapped (sys_aperture=[{sys_lo:#x}, {sys_hi:#x}])")

class MockIPBlock:
  def __init__(self, gpu:MockAMGPU, mmio:MockMMIOInterface, regs:dict[str, AMDReg]):
    self.gpu, self.mmio, self._regs = gpu, mmio, regs
    self._n2a = {n: r.addr[0] for n, r in regs.items()}
    self._a2n = {a: n for n, a in self._n2a.items()}
    self.addrs = set(self._n2a.values())
  def reg(self, name) -> int|None: return self._n2a.get(name)
  def decode(self, name) -> dict: return self._regs[name].decode(self.mmio.regs.get(self._n2a[name], 0))
  def read(self, reg:int) -> int: return self.mmio.regs.get(reg, 0)
  def write(self, reg:int, val:int): self.mmio.regs[reg] = val
  def _read_pair(self, pair) -> int:
    if pair[0] is None: return 0
    return self.mmio.regs.get(pair[0], 0) | (self.mmio.regs.get(pair[1], 0) << 32)

class MockPSP(MockIPBlock):
  def __init__(self, gpu, mmio):
    super().__init__(gpu, mmio, _build_ip_regs('mp', am.MP0_HWIP))
    self._sos_alive, self._ring_wptr = False, 0
    pref = "regMPASP_SMN_C2PMSG" if IP_VERSIONS[am.MP0_HWIP] >= (14,0,0) else "regMP0_SMN_C2PMSG"
    def r(n): return self.reg(f"{pref}_{n}")
    self._c2pmsg_35, self._c2pmsg_64, self._c2pmsg_67 = r(35), r(64), r(67)
    self._c2pmsg_69, self._c2pmsg_70, self._c2pmsg_81 = r(69), r(70), r(81)

  def read(self, reg:int) -> int:
    if reg == self._c2pmsg_35: return 0x80000000
    if reg == self._c2pmsg_81: return 0x1 if self._sos_alive else 0x0
    if reg == self._c2pmsg_64: return 0x80000000 if self._sos_alive else 0x0
    if reg == self._c2pmsg_67: return self._ring_wptr
    return super().read(reg)

  def write(self, reg:int, val:int):
    super().write(reg, val)
    if reg == self._c2pmsg_35 and val == am.PSP_BL__LOAD_SOSDRV: self._sos_alive = True
    if reg == self._c2pmsg_67: self._ring_submit(val)

  def _ring_submit(self, new_wptr:int):
    old_wptr = self._ring_wptr
    self._ring_wptr = new_wptr
    lo, hi = self._c2pmsg_69, self._c2pmsg_70
    if lo is None or hi is None: return
    ring_mc = self.mmio.regs.get(lo, 0) | (self.mmio.regs.get(hi, 0) << 32)
    ring_paddr = ring_mc - self.gpu.mc_base
    frame_off = ring_paddr + old_wptr * 4
    frame = am.struct_psp_gfx_rb_frame.from_buffer_copy(bytes(self.gpu.vram[frame_off:frame_off + ctypes.sizeof(am.struct_psp_gfx_rb_frame)]))
    fence_paddr = ((frame.fence_addr_hi << 32) | frame.fence_addr_lo) - self.gpu.mc_base
    if 0 <= fence_paddr < len(self.gpu.vram):
      struct.pack_into('<I', self.gpu.vram, fence_paddr, frame.fence_value)
    cmd_paddr = ((frame.cmd_buf_addr_hi << 32) | frame.cmd_buf_addr_lo) - self.gpu.mc_base
    if 0 <= cmd_paddr < len(self.gpu.vram):
      struct.pack_into('<I', self.gpu.vram, cmd_paddr + 864, 0)

class MockSMU(MockIPBlock):
  def __init__(self, gpu, mmio):
    regs = import_asic_regs('mp', (11, 0, 0), cls=functools.partial(AMDReg, bases={0: IP_BASES[am.MP1_HWIP]}))
    super().__init__(gpu, mmio, regs)
    self._msg_pending = False
    def r(n): return self.reg(f"mmMP1_SMN_C2PMSG_{n}")
    self._c2pmsg_53, self._c2pmsg_54, self._c2pmsg_66 = r(53), r(54), r(66)
    self._c2pmsg_75, self._c2pmsg_82, self._c2pmsg_90 = r(75), r(82), r(90)

  def read(self, reg:int) -> int:
    if reg == self._c2pmsg_90 or reg == self._c2pmsg_54: return 0x1 if self._msg_pending else super().read(reg)
    if reg == self._c2pmsg_82: return self.mmio.regs.get(reg, 3)
    return super().read(reg)

  def write(self, reg:int, val:int):
    super().write(reg, val)
    if reg == self._c2pmsg_66 or reg == self._c2pmsg_75: self._msg_pending = True
    if (reg == self._c2pmsg_90 or reg == self._c2pmsg_54) and val == 0: self._msg_pending = False

class MockSDMA(MockIPBlock):
  def __init__(self, gpu, mmio):
    all_gc = _build_ip_regs('gc', am.GC_HWIP)
    super().__init__(gpu, mmio, {n: r for n, r in all_gc.items() if 'SDMA' in n})

  def write(self, reg:int, val:int):
    super().write(reg, val)
    name = self._a2n.get(reg, '')
    if name.endswith('_RB_CNTL') and self._regs[name].decode(val).get('rb_enable', 0):
      self._activate_queue(name.rsplit('_RB_CNTL', 1)[0])

  def _activate_queue(self, prefix:str):
    ring_addr = self._read_pair((self.reg(f'{prefix}_RB_BASE'), self.reg(f'{prefix}_RB_BASE_HI'))) << 8
    rptr_addr = self._read_pair((self.reg(f'{prefix}_RB_RPTR_ADDR_LO'), self.reg(f'{prefix}_RB_RPTR_ADDR_HI')))
    wptr_addr = self._read_pair((self.reg(f'{prefix}_RB_WPTR_POLL_ADDR_LO'), self.reg(f'{prefix}_RB_WPTR_POLL_ADDR_HI')))
    rb_size = self.decode(f'{prefix}_RB_CNTL')['rb_size']
    self.gpu.add_sdma_queue(self.gpu.mmu.addr_to_host(ring_addr), 4 << rb_size,
                            self.gpu.mmu.addr_to_host(rptr_addr), self.gpu.mmu.addr_to_host(wptr_addr))

class MockGFX(MockIPBlock):
  def __init__(self, gpu, mmio):
    super().__init__(gpu, mmio, _build_ip_regs('gc', am.GC_HWIP))
    self._pt_base = (self.reg('regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32'), self.reg('regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32'))
    self._pt_start = (self.reg('regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32'), self.reg('regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32'))
    self._gc_inv_ack = self.reg('regGCVM_INVALIDATE_ENG17_ACK')
    self._gc_inv_req = self.reg('regGCVM_INVALIDATE_ENG17_REQ')
    self._hqd_active = self.reg('regCP_HQD_ACTIVE')

  def read(self, reg:int) -> int:
    if reg == self.reg('regCP_STAT') or reg == self.reg('regRLC_SAFE_MODE'): return 0
    if reg == self.reg('regRLC_RLCS_BOOTLOAD_STATUS'): return 0x2
    if reg == self._gc_inv_ack: return 0x1
    return super().read(reg)

  def write(self, reg:int, val:int):
    super().write(reg, val)
    if reg == self.reg('regCP_HQD_DEQUEUE_REQUEST'):
      if self._hqd_active is not None: self.mmio.regs[self._hqd_active] = 0
    if reg == self._hqd_active and val == 1: self._activate_pm4_queue()
    if reg == self._gc_inv_req: self.gpu.mmu.invalidate(self.get_pt_base(), self.get_va_base())

  def _activate_pm4_queue(self):
    ring_addr = self._read_pair((self.reg('regCP_HQD_PQ_BASE'), self.reg('regCP_HQD_PQ_BASE_HI'))) << 8
    rptr_addr = self._read_pair((self.reg('regCP_HQD_PQ_RPTR_REPORT_ADDR'), self.reg('regCP_HQD_PQ_RPTR_REPORT_ADDR_HI')))
    wptr_addr = self._read_pair((self.reg('regCP_HQD_PQ_WPTR_POLL_ADDR'), self.reg('regCP_HQD_PQ_WPTR_POLL_ADDR_HI')))
    queue_size = self.decode('regCP_HQD_PQ_CONTROL')['queue_size']
    self.gpu.add_pm4_queue(self.gpu.mmu.addr_to_host(ring_addr), 4 << (queue_size + 1),
                           self.gpu.mmu.addr_to_host(rptr_addr), self.gpu.mmu.addr_to_host(wptr_addr))

  def get_pt_base(self) -> int: return self._read_pair(self._pt_base) & 0x0000FFFFFFFFF000
  def get_va_base(self) -> int: return self._read_pair(self._pt_start) << 12

class MockGMC(MockIPBlock):
  def __init__(self, gpu, mmio, gfx:MockGFX):
    super().__init__(gpu, mmio, _build_ip_regs('mmhub', am.MMHUB_HWIP))
    self._gfx = gfx
    self._inv_ack = self.reg('regMMVM_INVALIDATE_ENG17_ACK')
    self._inv_sem = self.reg('regMMVM_INVALIDATE_ENG17_SEM')
    self._inv_req = self.reg('regMMVM_INVALIDATE_ENG17_REQ')
    self._fb_loc_top = self.reg('regMMMC_VM_FB_LOCATION_TOP')

  def read(self, reg:int) -> int:
    if reg == self._inv_ack or reg == self._inv_sem: return 0x1
    if reg == self._fb_loc_top: return VRAM_SIZE >> 24
    return super().read(reg)

  def write(self, reg:int, val:int):
    super().write(reg, val)
    if reg == self._inv_req: self.gpu.mmu.invalidate(self._gfx.get_pt_base(), self._gfx.get_va_base())

class MockNBIO(MockIPBlock):
  def __init__(self, gpu, mmio):
    regs = _build_ip_regs('nbif', am.NBIO_HWIP)
    regs.update(_build_ip_regs('hdp', am.HDP_HWIP))
    super().__init__(gpu, mmio, regs)
    self._remap_hdp = self.reg('regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL')
    self._hdp_flush = self.reg('regHDP_MEM_FLUSH_CNTL')

  def read(self, reg:int) -> int:
    if reg == self._remap_hdp and self._hdp_flush is not None: return self._hdp_flush * 4
    return super().read(reg)

class MockMMIOInterface:
  def __init__(self, gpu:MockAMGPU):
    self.gpu = gpu
    self.regs: dict[int, int] = {}
    gfx = MockGFX(gpu, self)
    self.gmc = MockGMC(gpu, self, gfx)
    self.blocks = [MockPSP(gpu, self), MockSMU(gpu, self), MockSDMA(gpu, self), gfx, self.gmc, MockNBIO(gpu, self)]
    self._addr_block: dict[int, MockIPBlock] = {}
    for block in self.blocks:
      for addr in block.addrs: self._addr_block.setdefault(addr, block)

  def __getitem__(self, index:int|slice) -> int|list[int]:
    if isinstance(index, slice): return [self[i] for i in range(index.start or 0, index.stop or 0, index.step or 1)]  # type: ignore[misc]
    if index == 0xde3: return VRAM_SIZE >> 20
    if block := self._addr_block.get(index): return block.read(index)
    return self.regs.get(index, 0)

  def __setitem__(self, index:int|slice, val:int|list[int]|tuple[int, ...]):
    if isinstance(index, slice):
      vals = val if isinstance(val, (list, tuple)) else [val] * ((index.stop - index.start) // (index.step or 1))  # type: ignore[operator]
      for i, v in zip(range(index.start or 0, index.stop or 0, index.step or 1), vals): self[i] = v
      return
    assert isinstance(val, int)
    self.regs[index] = val
    if block := self._addr_block.get(index): block.write(index, val)

  def __len__(self): return 0x10000000

class MockAMGPU(AMDGPU):
  def __init__(self, gpuid:int=0):
    super().__init__(gpuid)
    self.vram_fd = os.memfd_create("vram")
    os.ftruncate(self.vram_fd, VRAM_SIZE)
    self.vram_addr = libc.mmap(0, VRAM_SIZE, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, self.vram_fd, 0)
    self.vram = (ctypes.c_ubyte * VRAM_SIZE).from_address(self.vram_addr)
    self.doorbell_fd = os.memfd_create("doorbell")
    os.ftruncate(self.doorbell_fd, 0x2000)
    self.arch = "rdna4"
    self._sysmem_map:dict[int,int] = {}
    self._next_sysmem_paddr = 0x100000000
    self.mmu = MockMMU(self)
    self.mmio = MockMMIOInterface(self)
    self._preboot()

  def translate_addr(self, addr:int) -> int: return self.mmu.addr_to_host(addr)

  def map_vram_at(self, va:int, paddr:int, size:int):
    libc.mmap(va, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | 0x10, self.vram_fd, paddr)

  def _preboot(self):
    ip_data = bytearray()
    for hwip, (major, minor, rev) in IP_VERSIONS.items():
      ip = am.struct_ip_v4(hw_id=IP_HWIDS[hwip], num_base_address=len(IP_BASES[hwip]), major=major, minor=minor, revision=rev)
      ip_data += bytes(ip) + b'\x00'
      for b in IP_BASES[hwip]: ip_data += struct.pack('<I', b)

    dhdr = am.struct_die_header(num_ips=len(IP_VERSIONS))
    ihdr = am.struct_ip_discovery_header(signature=am.DISCOVERY_TABLE_SIGNATURE, version=4, num_dies=1)
    ip_disc_off = ctypes.sizeof(am.struct_binary_header)
    ihdr.die_info[0].die_offset = ip_disc_off + ctypes.sizeof(am.struct_ip_discovery_header)

    gc = am.struct_gc_info_v2_1()
    gc.header.table_id, gc.header.version_major, gc.header.version_minor = am.GC, 2, 1
    gc.header.size = ctypes.sizeof(am.struct_gc_info_v2_1)
    for field, val in GC_INFO.items(): setattr(gc, field, val)

    gc_off = ip_disc_off + ctypes.sizeof(am.struct_ip_discovery_header) + ctypes.sizeof(am.struct_die_header) + len(ip_data)
    bhdr = am.struct_binary_header(binary_signature=am.BINARY_SIGNATURE)
    bhdr.table_list[am.IP_DISCOVERY].offset = ip_disc_off
    bhdr.table_list[am.GC].offset = gc_off

    tbl = bytes(bhdr) + bytes(ihdr) + bytes(dhdr) + ip_data + bytes(gc)
    tbl_offset = VRAM_SIZE - (64 << 10)
    self.vram[tbl_offset:tbl_offset + len(tbl)] = list(tbl)

  @property
  def mc_base(self) -> int:
    fb_loc_base = self.mmio.gmc.reg('regMMMC_VM_FB_LOCATION_BASE') or 0
    return (self.mmio.regs.get(fb_loc_base, 0) & 0xFFFFFF) << 24
