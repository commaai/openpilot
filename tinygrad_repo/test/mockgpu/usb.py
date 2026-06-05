from __future__ import annotations
import ctypes, mmap, struct, sys
if sys.platform != "win32": from tinygrad.runtime.autogen import libc

class MockUSB:
  def __init__(self, mem):
    self.mem = mem
  def read(self, address, size): return bytes(self.mem[address:address+size])
  def write(self, address, data, ignore_cache=False): self.mem[address:address+len(data)] = data
  def pcie_mem_req(self, address, value=None, size=1):
    if value is None: return int.from_bytes(self.mem[address:address+size], "little")
    else: self.mem[address:address+size] = value.to_bytes(size, "little")
  def pcie_mem_write(self, address, values, size):
    for i, value in enumerate(values): self.pcie_mem_req(address + i * size, value, size)

# *** ASM24 Controller Mock ***

_mock_usb_state: MockASM24State|None = None

class MockASM24State:
  """Mock ASM24 controller: XRAM memory map, DMA windows, TLP engine, PCI config space.

  Memory map (64KB XRAM):
    0xA000-0xAFFF: DMA window -> sys 0x820000
    0xB000-0xB1FF: DMA window -> sys 0x800000
    0xB200-0xB7FF: PCI MMIO (TLP engine)
    0xF000-0xFFFF: DMA window -> sys 0x200000 (512KB)
  """
  XRAM_SIZE = 0x10000

  TLP_FMT_TYPE = 0xB210
  TLP_BYTE_EN = 0xB217
  TLP_ADDR_LO = 0xB218
  TLP_ADDR_HI = 0xB21C
  TLP_DATA = 0xB220
  TLP_COMPL = 0xB22A
  TLP_TRIGGER = 0xB254
  TLP_LINK_STATUS = 0xB284
  TLP_STATUS = 0xB296

  def __init__(self, gpu, driver, vram_size:int, doorbell_size:int, mmio_size:int):
    self.gpu, self.driver = gpu, driver
    self._xram = bytearray(self.XRAM_SIZE)

    self._doorbell_addr = libc.mmap(0, doorbell_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, gpu.doorbell_fd, 0)
    self._doorbell = (ctypes.c_ubyte * doorbell_size).from_address(self._doorbell_addr)

    # DMA windows: ctrl_addr -> (host_addr, size)
    self._dma_regions: dict[int, tuple[int, int]] = {}
    self._add_dma_window(0xF000, 0x200000, 0x80000)
    self._add_dma_window(0xA000, 0x820000, 0x1000)
    self._add_dma_window(0xB000, 0x800000, 0x200)

    # PCI config space: (bus,dev,fn) -> bytearray(4096)
    self._pci_cfg: dict[tuple[int,int,int], bytearray] = {}

    # GPU BAR definitions: reg_offset -> (size, type_bits, is_64bit)
    self._gpu_bars: dict[int, tuple[int, int, bool]] = {
      0x10: (vram_size, 0x0C, True),       # BAR0: VRAM, 64-bit prefetchable
      0x18: (doorbell_size, 0x00, False),   # BAR2: doorbell, 32-bit
      0x1C: (0, 0x00, False),              # BAR3: unused
      0x20: (0, 0x00, False),              # BAR4: unused
      0x24: (mmio_size, 0x00, False),      # BAR5: MMIO, 32-bit
    }
    self._bar_addrs: dict[int, tuple[int, int]] = {}  # reg_offset -> (addr, size)

    # Initialize GPU config space (bus=4, dev=0, fn=0) with BAR type bits and REBAR capability
    gpu_cfg = self._get_cfg(4, 0, 0)
    for reg_off, (sz, type_bits, _) in self._gpu_bars.items():
      if sz > 0: struct.pack_into('<I', gpu_cfg, reg_off, type_bits)
    struct.pack_into('<I', gpu_cfg, 0x100, 0x15 | (1 << 16))  # REBAR cap header: id=0x15, version=1, next=0
    struct.pack_into('<I', gpu_cfg, 0x104, sum(1 << (i + 4) for i in range(10)))  # supported sizes up to 512MB

  def _get_cfg(self, bus:int, dev:int, fn:int) -> bytearray:
    if (key:=(bus, dev, fn)) not in self._pci_cfg: self._pci_cfg[key] = bytearray(4096)
    return self._pci_cfg[key]

  def _add_dma_window(self, ctrl_addr:int, sys_addr:int, size:int):
    host_addr = libc.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, -1, 0)
    self._dma_regions[ctrl_addr] = (host_addr, size)
    for off in range(0, size, 0x1000): self.gpu._sysmem_map[sys_addr + off] = host_addr + off

  # --- XRAM access ---

  def _xram_read(self, addr:int, length:int) -> bytes:
    for ctrl_addr, (host_addr, dma_size) in self._dma_regions.items():
      if ctrl_addr <= addr < ctrl_addr + dma_size:
        return bytes((ctypes.c_ubyte * length).from_address(host_addr + (addr - ctrl_addr)))
    return bytes(self._xram[addr:addr+length])

  def _xram_write_byte(self, addr:int, value:int):
    for ctrl_addr, (host_addr, dma_size) in self._dma_regions.items():
      if ctrl_addr <= addr < ctrl_addr + dma_size:
        (ctypes.c_ubyte * 1).from_address(host_addr + (addr - ctrl_addr))[0] = value
        return
    if addr == self.TLP_STATUS:
      self._xram[addr] &= ~value & 0xFF
      return
    self._xram[addr] = value
    if addr == self.TLP_TRIGGER and value == 0x0F: self._process_tlp()

  # --- TLP engine ---

  def _process_tlp(self):
    fmt_type, byte_en = self._xram[self.TLP_FMT_TYPE], self._xram[self.TLP_BYTE_EN]
    addr_lo = int.from_bytes(self._xram[self.TLP_ADDR_LO:self.TLP_ADDR_LO+4], 'big')
    addr_hi = int.from_bytes(self._xram[self.TLP_ADDR_HI:self.TLP_ADDR_HI+4], 'big')
    address = addr_lo | (addr_hi << 32)

    size, offset, tmp = 0, 0, byte_en
    while tmp and not (tmp & 1):
      offset += 1
      tmp >>= 1
    while tmp:
      size += tmp & 1
      tmp >>= 1

    is_write, is_cfg = bool(fmt_type & 0x40), (fmt_type & 0xbe) == 0x04

    if is_cfg:
      bus, dev, fn, byte_addr = (address >> 24) & 0xFF, (address >> 19) & 0x1F, (address >> 16) & 0x7, address & 0xFFC
      if is_write:
        data = int.from_bytes(self._xram[self.TLP_DATA:self.TLP_DATA+4], 'big')
        self._cfg_write(bus, dev, fn, byte_addr + offset, (data >> (8 * offset)) & ((1 << (8 * size)) - 1), size)
      else:
        self._xram[self.TLP_DATA:self.TLP_DATA+4] = int.from_bytes(self._get_cfg(bus, dev, fn)[byte_addr:byte_addr+4], 'little').to_bytes(4, 'big')
      self._xram[self.TLP_COMPL:self.TLP_COMPL+2] = (4).to_bytes(2, 'big')
      self._xram[self.TLP_LINK_STATUS] = 0x01 if not is_write else 0x00
      self._xram[self.TLP_STATUS] = 0x02
      return

    if is_write:
      data = int.from_bytes(self._xram[self.TLP_DATA:self.TLP_DATA+4], 'big')
      self._pcie_dispatch(address + offset, (data >> (8 * offset)) & ((1 << (8 * size)) - 1), size)
    else:
      result = self._pcie_dispatch(address + offset, None, size)
      if result is not None:
        self._xram[self.TLP_DATA:self.TLP_DATA+4] = ((result << (8 * offset)) & 0xFFFFFFFF).to_bytes(4, 'big')

    self._xram[self.TLP_COMPL:self.TLP_COMPL+2] = (size & 0xFFF).to_bytes(2, 'big')
    self._xram[self.TLP_LINK_STATUS] = 0x01 if not is_write else 0x00
    self._xram[self.TLP_STATUS] = 0x02

  def _cfg_write(self, bus:int, dev:int, fn:int, byte_addr:int, val:int, size:int):
    cfg = self._get_cfg(bus, dev, fn)

    # Handle BAR register writes for GPU device (bus=4, dev=0, fn=0)
    if (bus, dev, fn) == (4, 0, 0) and 0x10 <= byte_addr < 0x28 and size == 4:
      reg_off = byte_addr & ~0x3
      if (bar_def:=self._gpu_bars.get(reg_off)) is not None:
        bar_size, type_bits, is_64 = bar_def
        if bar_size == 0: return  # unused BAR
        if val == 0xFFFFFFFF:  # size probe
          struct.pack_into('<I', cfg, reg_off, (~(bar_size - 1)) & 0xFFFFFFF0 | type_bits)
        else:
          struct.pack_into('<I', cfg, reg_off, val)
          hi = struct.unpack_from('<I', cfg, reg_off + 4)[0] if is_64 else 0
          self._bar_addrs[reg_off] = ((hi << 32) | (val & ~0xF), bar_size)
        return
      # Check if upper 32 bits of a 64-bit BAR
      for breg, (bsz, _, b64) in self._gpu_bars.items():
        if b64 and reg_off == breg + 4:
          struct.pack_into('<I', cfg, reg_off, 0xFFFFFFFF if val == 0xFFFFFFFF else val)
          if val != 0xFFFFFFFF:
            self._bar_addrs[breg] = ((val << 32) | (struct.unpack_from('<I', cfg, breg)[0] & ~0xF), bsz)
          return

    # Generic config write
    for i in range(size): cfg[byte_addr + i] = (val >> (8 * i)) & 0xFF

  def _pcie_dispatch(self, address:int, value:int|None, size:int) -> int|None:
    for reg_off, (bar_addr, bar_size) in self._bar_addrs.items():
      if bar_addr <= address < bar_addr + bar_size:
        offset = address - bar_addr
        if reg_off == 0x10:  # BAR0 - VRAM
          if value is None: return int.from_bytes(bytes(self.gpu.vram[offset:offset+size]), "little")
          self.gpu.vram[offset:offset+size] = list(value.to_bytes(size, "little"))
          return None
        if reg_off == 0x18:  # BAR2 - Doorbell
          if value is None: return int.from_bytes(bytes(self._doorbell[offset:offset+size]), "little")
          for i, b in enumerate(value.to_bytes(size, "little")): self._doorbell[offset + i] = b
          self.driver._emulate_execute()
          return None
        if reg_off == 0x24:  # BAR5 - MMIO
          if value is None: return self.gpu.mmio[offset // 4]
          self.gpu.mmio[offset // 4] = value
          return None
    raise ValueError(f"PCIe address {address:#x} not mapped to any BAR")

  # --- CDB processing (called by MockUSB3.send_batch) ---

  def process_cdb(self, cdb:bytes, rlen:int, send_data:bytes|None) -> bytes|None:
    op = cdb[0]
    if op == 0xE5:  # write byte
      self._xram_write_byte(((cdb[2] << 16) | (cdb[3] << 8) | cdb[4]) & 0xFFFF, cdb[1])
      return None
    if op == 0xE4:  # read
      return self._xram_read(((cdb[2] << 16) | (cdb[3] << 8) | cdb[4]) & 0xFFFF, cdb[1])
    if op == 0x8A and send_data is not None and 0xF000 in self._dma_regions:  # SCSI write
      host_addr, dma_size = self._dma_regions[0xF000]
      ctypes.memmove(host_addr, send_data, min(len(send_data), dma_size))
    return None

class MockUSB3:
  @classmethod
  def list_devices(cls, vendor, dev): return [(0, "usb:mock")]
  def __init__(self, *args, **kwargs):
    self.product, self.is_custom = "", False
  def send_batch(self, cdbs:list[bytes], idata:list[int]|None=None, odata:list[bytes|None]|None=None) -> list[bytes|None]:
    assert _mock_usb_state is not None
    idata, odata = idata or [0] * len(cdbs), odata or [None] * len(cdbs)
    results: list[bytes|None] = []
    for cdb, rlen, sdata in zip(cdbs, idata, odata):
      result = _mock_usb_state.process_cdb(cdb, rlen, sdata)
      results.append(result if rlen > 0 else None)
    return results
