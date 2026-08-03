from __future__ import annotations
import ctypes, mmap, struct, sys
if sys.platform != "win32": from tinygrad.runtime.autogen import libc

class MockUSB:
  def __init__(self, mem):
    self.mem = mem
  def read(self, address, size): return bytes(self.mem[address:address+size])
  def write(self, address, data): self.mem[address:address+len(data)] = data
  def pcie_mem_read(self, address, nbytes): return bytes(self.mem[address:address+nbytes])
  def pcie_mem_write(self, address, data): self.mem[address:address+len(data)] = data

# *** ASM24 Controller Mock ***

_mock_usb_state: MockASM24State|None = None

class MockASM24State:
  """Mock custom ASM24 controller: XRAM, DMA windows, PCI config space, and GPU BARs.

  Memory map (64KB XRAM):
    0xA000-0xAFFF: DMA window -> sys 0x820000
    0xB000-0xB1FF: DMA window -> sys 0x800000
    0xB200-0xB7FF: controller PCI MMIO
    0xF000-0xFFFF: DMA window -> sys 0x200000 (512KB)
  """
  XRAM_SIZE = 0x10000

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
    self._xram[addr] = value

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

  def _find_bar(self, address:int, size:int) -> tuple[int, int]:
    for reg_off, (bar_addr, bar_size) in self._bar_addrs.items():
      if bar_addr <= address and address + size <= bar_addr + bar_size: return reg_off, address - bar_addr
    raise ValueError(f"PCIe range {address:#x}+{size:#x} not mapped to any BAR")

  def _pcie_read(self, address:int, size:int) -> bytes:
    reg_off, offset = self._find_bar(address, size)
    if reg_off == 0x10: return bytes(self.gpu.vram[offset:offset+size])
    if reg_off == 0x18: return bytes(self._doorbell[offset:offset+size])
    if reg_off == 0x24: return bytes((self.gpu.mmio[(offset+i)//4] >> (8*((offset+i)&3))) & 0xFF for i in range(size))
    raise RuntimeError(f"unsupported BAR register {reg_off:#x}")

  def _pcie_write(self, address:int, data:bytes):
    reg_off, offset = self._find_bar(address, len(data))
    if reg_off == 0x10: self.gpu.vram[offset:offset+len(data)] = list(data)
    elif reg_off == 0x18:
      self._doorbell[offset:offset+len(data)] = list(data)
      self.driver._emulate_execute()
    elif reg_off == 0x24:
      updates: dict[int, int] = {}
      for i, byte in enumerate(data):
        idx, shift = (offset+i)//4, 8*((offset+i)&3)
        updates[idx] = (updates.get(idx, self.gpu.mmio[idx]) & ~(0xFF << shift)) | (byte << shift)
      for idx, val in updates.items(): self.gpu.mmio[idx] = val
    else: raise RuntimeError(f"unsupported BAR register {reg_off:#x}")

  def _pcie_dispatch(self, address:int, value:int|None, size:int) -> int|None:
    if value is None: return int.from_bytes(self._pcie_read(address, size), 'little')
    self._pcie_write(address, value.to_bytes(size, 'little'))
    return None

class MockUSB3:
  @classmethod
  def list_devices(cls, vendor, dev): return [(0, "usb:mock")]
  def __init__(self, *args, **kwargs):
    self.product = "custom mock"
    self._bulk_read_op: tuple[str, int, int]|None = None
    self._bulk_write_op: tuple[str, int, int]|None = None
    self._f0_reply = bytes(8)

  @property
  def state(self) -> MockASM24State:
    assert _mock_usb_state is not None
    return _mock_usb_state

  def control_write(self, request:int, value:int=0, index:int=0, data:bytes=b'', timeout:int=1000):
    if request == 0xF3:
      self.state._xram[0xB450] = 0x78 if value else 0
    elif request == 0xE5:
      self.state._xram_write_byte(value, index)
    elif request == 0xF2:
      op = ("sram_read" if value & 0x8000 else "sram_write", 0xF000, (value & 0x7FFF) * 512)
      if value & 0x8000: self._bulk_read_op = op
      else: self._bulk_write_op = op
    elif request == 0xF0:
      address_lo, address_hi, payload = struct.unpack('<III', data)
      address, fmt_type, byte_en = address_lo | (address_hi << 32), value & 0xFF, value >> 8
      if index == 1: self._bulk_write_op = ("pcie_write", address, payload * 4)
      elif index == 2: self._bulk_read_op = ("pcie_read", address, payload * 4)
      else:
        assert index == 0 and byte_en
        offset = (byte_en & -byte_en).bit_length() - 1
        size, is_write, is_cfg = byte_en.bit_count(), bool(fmt_type & 0x40), (fmt_type & 0xBE) == 0x04
        if is_cfg:
          bus, dev, fn, byte_addr = (address >> 24) & 0xFF, (address >> 19) & 0x1F, (address >> 16) & 0x7, address & 0xFFC
          if is_write: self.state._cfg_write(bus, dev, fn, byte_addr + offset, (payload >> (8 * offset)) & ((1 << (8 * size))-1), size)
          else: payload = int.from_bytes(self.state._get_cfg(bus, dev, fn)[byte_addr:byte_addr+4], 'little')
        elif is_write:
          self.state._pcie_dispatch(address + offset, (payload >> (8 * offset)) & ((1 << (8 * size))-1), size)
        else: payload = (self.state._pcie_dispatch(address + offset, None, size) or 0) << (8 * offset)
        self._f0_reply = struct.pack('<I', payload & 0xFFFFFFFF) + bytes(4)
    else: raise ValueError(f"unsupported control OUT request 0x{request:02X}")

  def control_read(self, request:int, length:int, value:int=0, index:int=0, timeout:int=1000) -> memoryview:
    if request == 0xE4: data = self.state._xram_read(value, length)
    elif request == 0xF0: data = self._f0_reply
    else: raise ValueError(f"unsupported control IN request 0x{request:02X}")
    return memoryview(data[:length])

  def bulk_write(self, data:bytes, timeout:int=1000):
    assert self._bulk_write_op is not None
    op, address, size = self._bulk_write_op
    assert len(data) == size
    if op == "sram_write":
      host_addr, region_size = self.state._dma_regions[address]
      ctypes.memmove(host_addr, data, min(len(data), region_size))
    elif op == "pcie_write": self.state._pcie_write(address, data)
    else: raise RuntimeError(f"cannot bulk write for {op}")
    self._bulk_write_op = None

  def bulk_read(self, length:int, timeout:int=1000) -> memoryview:
    assert self._bulk_read_op is not None
    op, address, size = self._bulk_read_op
    assert length == size
    if op == "sram_read":
      host_addr, region_size = self.state._dma_regions[address]
      data = bytes((ctypes.c_ubyte * min(length, region_size)).from_address(host_addr))
    elif op == "pcie_read": data = self.state._pcie_read(address, length)
    else: raise RuntimeError(f"cannot bulk read for {op}")
    self._bulk_read_op = None
    return memoryview(data)
