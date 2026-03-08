from __future__ import annotations
import os, mmap, array, functools, ctypes, select, contextlib, dataclasses, sys, itertools, struct, socket, subprocess, time, enum
from typing import ClassVar
from tinygrad.helpers import round_up, getenv, OSX, temp, ceildiv, unwrap, fetch, system
from tinygrad.runtime.autogen import libc, pci, vfio, iokit, corefoundation
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface, HCQBuffer, hcq_filter_visible_devices
from tinygrad.runtime.support.memory import MemoryManager, VirtMapping, AddrSpace
from tinygrad.runtime.support.usb import ASM24Controller, USBMMIOInterface

MAP_FIXED, MAP_LOCKED, MAP_POPULATE, MAP_NORESERVE = 0x10, 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000), 0x400

@dataclasses.dataclass(frozen=True)
class PCIBarInfo: addr:int; size:int # noqa: E702

class _System:
  def write_sysfs(self, path:str, value:str, msg:str, expected:str|None=None):
    if FileIOInterface(path, os.O_RDONLY).read().splitlines()[0] != (expected or value):
      os.system(cmd:=f"sudo sh -c 'echo {value} > {path}'")
      if FileIOInterface(path, os.O_RDONLY).read().splitlines()[0] != (expected or value): raise RuntimeError(f"{msg}. Please run {cmd} manually.")

  @functools.cached_property
  def atomic_lib(self): return ctypes.CDLL(ctypes.util.find_library('atomic')) if sys.platform == "linux" else None

  @functools.cached_property
  def libsys(self): return ctypes.CDLL(ctypes.util.find_library("System"))

  @functools.cached_property
  def pagemap(self) -> FileIOInterface:
    self.write_sysfs("/proc/sys/vm/compact_unevictable_allowed", "0", "Failed to disable migration of locked pages")
    return FileIOInterface("/proc/self/pagemap", os.O_RDONLY)

  @functools.cached_property
  def vfio(self) -> FileIOInterface|None:
    try:
      if not FileIOInterface.exists("/sys/module/vfio"): os.system("sudo modprobe vfio-pci disable_idle_d3=1")

      FileIOInterface("/sys/module/vfio/parameters/enable_unsafe_noiommu_mode", os.O_RDWR).write("1")
      vfio_fd = FileIOInterface("/dev/vfio/vfio", os.O_RDWR)
      vfio.VFIO_CHECK_EXTENSION(vfio_fd, vfio.VFIO_NOIOMMU_IOMMU)

      return vfio_fd
    except OSError: return None

  def reserve_hugepages(self, cnt): os.system(f"sudo sh -c 'echo {cnt} > /proc/sys/vm/nr_hugepages'")

  def memory_barrier(self): lib.atomic_thread_fence(__ATOMIC_SEQ_CST:=5) if (lib:=self.libsys if OSX else self.atomic_lib) is not None else None

  def lock_memory(self, addr:int, size:int):
    if libc.mlock(ctypes.c_void_p(addr), size): raise RuntimeError(f"Failed to lock memory at {addr:#x} with size {size:#x}")

  def system_paddrs(self, vaddr:int, size:int) -> list[int]:
    self.pagemap.seek(vaddr // mmap.PAGESIZE * 8)
    return [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', self.pagemap.read(size//mmap.PAGESIZE*8, binary=True))]

  def pci_scan_bus(self, vendor:int, devices:list[tuple[int, list[int]]], base_class:int|None=None) -> list[str]:
    all_devs = []
    if OSX:
      def read_prop(svc, key) -> int:
        cfkey = corefoundation.CFStringCreateWithCString(None, key.encode(), corefoundation.kCFStringEncodingUTF8)
        cfdata = ctypes.cast(iokit.IORegistryEntryCreateCFProperty(svc, ctypes.cast(cfkey, iokit.CFStringRef), None, 0), corefoundation.CFDataRef)
        corefoundation.CFDataGetBytes(cfdata, corefoundation.CFRange(0, corefoundation.CFDataGetLength(cfdata)), buf:=(ctypes.c_uint8*8)())
        return int.from_bytes(bytes(buf), "little")

      iokit.IOServiceGetMatchingServices(0, iokit.IOServiceMatching(b"IOPCIDevice"), ctypes.byref(iterator:=ctypes.c_uint()))
      while svc:=iokit.IOIteratorNext(iterator): all_devs.append((v:=read_prop(svc, "vendor-id"), d:=read_prop(svc, "device-id"), f"{v:x}:{d:x}"))
    else:
      for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
        if base_class is not None and int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/class").read(), 16) >> 16 != base_class: continue
        all_devs.append((int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16),
                         int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16), pcibus))

    return sorted([val for vendor, device, val in all_devs if vendor == vendor and any((device & mask) in devlist for mask, devlist in devices)])

  def pci_setup_usb_bars(self, usb:ASM24Controller, gpu_bus:int, mem_base:int, pref_mem_base:int) -> dict[int, PCIBarInfo]:
    for bus in range(gpu_bus):
      # All 3 values must be written at the same time.
      buses = (0 << 0) | ((bus+1) << 8) | ((gpu_bus) << 16)
      usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=buses, size=4)

      usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=(mem_base>>16) & 0xffff, size=2)
      usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=(pref_mem_base>>16) & 0xffff, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)
      usb.pcie_cfg_req(pci.PCI_PREF_BASE_UPPER32,  bus=bus, dev=0, fn=0, value=pref_mem_base >> 32, size=4)
      usb.pcie_cfg_req(pci.PCI_PREF_LIMIT_UPPER32, bus=bus, dev=0, fn=0, value=0xffffffff, size=4)

      usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

    # resize bar 0
    cap_ptr = 0x100
    while cap_ptr:
      if pci.PCI_EXT_CAP_ID(hdr:=usb.pcie_cfg_req(cap_ptr, bus=gpu_bus, dev=0, fn=0, size=4)) == pci.PCI_EXT_CAP_ID_REBAR:
        cap = usb.pcie_cfg_req(cap_ptr + 0x04, bus=gpu_bus, dev=0, fn=0, size=4)
        new_ctrl = (usb.pcie_cfg_req(cap_ptr + 0x08, bus=gpu_bus, dev=0, fn=0, size=4) & ~0x1F00) | ((int(cap >> 4).bit_length() - 1) << 8)
        usb.pcie_cfg_req(cap_ptr + 0x08, bus=gpu_bus, dev=0, fn=0, value=new_ctrl, size=4)

      cap_ptr = pci.PCI_EXT_CAP_NEXT(hdr)

    mem_space_addr, bar_off, bars = [mem_base, pref_mem_base], 0, {}
    while bar_off < 24:
      cfg = usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, size=4)
      bar_mem, bar_64 = bool(cfg & pci.PCI_BASE_ADDRESS_MEM_PREFETCH), cfg & pci.PCI_BASE_ADDRESS_MEM_TYPE_64

      if (cfg & pci.PCI_BASE_ADDRESS_SPACE) == pci.PCI_BASE_ADDRESS_SPACE_MEMORY:
        usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, value=0xffffffff, size=4)
        lo = (usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, size=4) & 0xfffffff0)

        if bar_64: usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off + 4, bus=gpu_bus, dev=0, fn=0, value=0xffffffff, size=4)
        hi = (usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off + 4, bus=gpu_bus, dev=0, fn=0, size=4) if bar_64 else 0)

        bar_size = ((~(((hi << 32) | lo) & ~0xf)) + 1) & (0xffffffffffffffff if bar_64 else 0xffffffff)

        usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=gpu_bus, dev=0, fn=0, value=mem_space_addr[bar_mem] & 0xffffffff, size=4)
        if bar_64: usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off + 4, bus=gpu_bus, dev=0, fn=0, value=mem_space_addr[bar_mem] >> 32, size=4)

        bars[bar_off // 4] = PCIBarInfo(mem_space_addr[bar_mem], bar_size)
        mem_space_addr[bar_mem] += round_up(bar_size, 2 << 20)

      bar_off += 8 if bar_64 else 4

    usb.pcie_cfg_req(pci.PCI_COMMAND, bus=gpu_bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)
    return bars

  def flock_acquire(self, name:str) -> int:
    import fcntl # to support windows

    os.umask(0) # Set umask to 0 to allow creating files with 0666 permissions

    # Avoid O_CREAT because we don’t want to re-create/replace an existing file (triggers extra perms checks) when opening as non-owner.
    if os.path.exists(lock_name:=temp(name)): self.lock_fd = os.open(lock_name, os.O_RDWR)
    else: self.lock_fd = os.open(lock_name, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)

    try: fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: raise RuntimeError(f"Failed to acquire lock file {name}. `sudo lsof {lock_name}` may help identify the process holding the lock.")

    return self.lock_fd

System = _System()

# *** PCI Devices

class PCIDevice:
  def __init__(self, devpref:str, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    self.pcibus, self.irq_poller = pcibus, None

    try: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/enable", os.O_RDWR)
    except PermissionError: raise PermissionError(f"Cannot access PCI device {pcibus}: run `extra/amdpci/setup_python_cap.sh` or use sudo")

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", os.O_WRONLY).write(self.pcibus)

    for i in resize_bars or []:
      if FileIOInterface.exists(rpath:=f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize"):
        try: FileIOInterface(rpath, os.O_RDWR).write(str(int(FileIOInterface(rpath, os.O_RDONLY).read(), 16).bit_length() - 1))
        except OSError as e: raise RuntimeError(f"Cannot resize BAR {i}: {e}. Ensure the resizable BAR option is enabled.") from e

    if getenv("VFIO", 0) and (vfio_fd:=System.vfio) is not None:
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver_override", os.O_WRONLY).write("vfio-pci")
      FileIOInterface("/sys/bus/pci/drivers_probe", os.O_WRONLY).write(self.pcibus)
      iommu_group = FileIOInterface.readlink(f"/sys/bus/pci/devices/{self.pcibus}/iommu_group").split('/')[-1]

      self.vfio_group = FileIOInterface(f"/dev/vfio/noiommu-{iommu_group}", os.O_RDWR)
      vfio.VFIO_GROUP_SET_CONTAINER(self.vfio_group, ctypes.c_int(vfio_fd.fd))

      with contextlib.suppress(OSError): vfio.VFIO_SET_IOMMU(vfio_fd, vfio.VFIO_NOIOMMU_IOMMU) # set iommu works only once for the fd.
      self.vfio_dev = FileIOInterface(fd=vfio.VFIO_GROUP_GET_DEVICE_FD(self.vfio_group, ctypes.create_string_buffer(self.pcibus.encode())))

      self.irq_fd = FileIOInterface.eventfd(0, 0)
      self.irq_poller = select.poll()
      self.irq_poller.register(self.irq_fd.fd, select.POLLIN)

      irqs = vfio.struct_vfio_irq_set(index=vfio.VFIO_PCI_MSI_IRQ_INDEX, flags=vfio.VFIO_IRQ_SET_DATA_EVENTFD|vfio.VFIO_IRQ_SET_ACTION_TRIGGER,
        argsz=ctypes.sizeof(vfio.struct_vfio_irq_set), count=1, data=(ctypes.c_int * 1)(self.irq_fd.fd))
      vfio.VFIO_DEVICE_SET_IRQS(self.vfio_dev, irqs)
    else: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/enable", os.O_RDWR).write("1")

    self.cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
    self.bar_fds = {b: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{b}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC) for b in bars}

    res = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource", os.O_RDONLY).read().splitlines()
    self.bar_info = {j:PCIBarInfo(int(s,16), int(e,16)-int(s,16)+1) for j,(s,e,_) in enumerate(l.split() for l in res)}

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False) -> tuple[MMIOInterface, list[int]]:
    assert not contiguous or size <= (2 << 20), "Contiguous allocation is only supported for sizes up to 2MB"
    flags = (libc.MAP_HUGETLB if contiguous and (size:=round_up(size, mmap.PAGESIZE)) > mmap.PAGESIZE else 0) | (MAP_FIXED if vaddr else 0)
    va = FileIOInterface.anon_mmap(vaddr, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS|MAP_POPULATE|MAP_LOCKED|flags, 0)
    sysmem_view, paddrs = MMIOInterface(va, size), [(x, mmap.PAGESIZE) for x in System.system_paddrs(va, size)]
    return sysmem_view, [p + i for p, sz in paddrs for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]
  def read_config(self, offset:int, size:int): return int.from_bytes(self.cfg_fd.read(size, binary=True, offset=offset), byteorder='little')
  def write_config(self, offset:int, value:int, size:int): self.cfg_fd.write(value.to_bytes(size, byteorder='little'), binary=True, offset=offset)
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    fd, sz = self.bar_fds[bar], size or (self.bar_info[bar].size - off)
    libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
    return MMIOInterface(loc, sz, fmt=fmt)
  def reset(self): os.system(f"sudo sh -c 'echo 1 > /sys/bus/pci/devices/{self.pcibus}/reset'")

class USBPCIDevice(PCIDevice):
  def __init__(self, devpref:str, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    self.usb = ASM24Controller()
    self.pcibus, self.bar_info = pcibus, System.pci_setup_usb_bars(self.usb, gpu_bus=4, mem_base=0x10000000, pref_mem_base=(32 << 30))
  def map_bar(self, bar, off=0, addr=0, size=None, fmt='B'):
    return USBMMIOInterface(self.usb, self.bar_info[bar].addr + off, size or self.bar_info[bar].size, fmt)
  def dma_view(self, ctrl_addr, size): return USBMMIOInterface(self.usb, ctrl_addr, size, fmt='B', pcimem=False)

class PCIDevImplBase:
  mm: MemoryManager

@dataclasses.dataclass
class PCIAllocationMeta: mapping:VirtMapping; has_cpu_mapping:bool; hMemory:int=0 # noqa: E702

class LNXPCIIfaceBase:
  dev_impl:PCIDevImplBase
  gpus:ClassVar[list[str]] = []

  def __init__(self, dev, dev_id, vendor, devices:list[tuple[int, list[int]]], bars, vram_bar, va_start, va_size, base_class:int|None=None):
    if len((cls:=type(self)).gpus) == 0:
      cls.gpus = hcq_filter_visible_devices(System.pci_scan_bus(vendor, devices, base_class))

      # Acquire va range to avoid collisions.
      FileIOInterface.anon_mmap(va_start, va_size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, 0)
    self.pci_dev, self.dev, self.vram_bar = PCIDevice(dev.__class__.__name__[:2], cls.gpus[dev_id], bars=bars, resize_bars=[vram_bar]), dev, vram_bar
    self.p2p_base_addr = self.pci_dev.bar_info[vram_bar].addr

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False, **kwargs) -> HCQBuffer:
    # NOTE: logic on macos is different, since bar is small
    should_use_sysmem = host or ((cpu_access if OSX else (uncached and cpu_access)) and not force_devmem)
    if should_use_sysmem:
      vaddr = self.dev_impl.mm.alloc_vaddr(size:=round_up(size, mmap.PAGESIZE), align=mmap.PAGESIZE)
      memview, paddrs = self.pci_dev.alloc_sysmem(size, vaddr=vaddr, contiguous=contiguous)
      mapping = self.dev_impl.mm.map_range(vaddr, size, [(paddr, 0x1000) for paddr in paddrs], aspace=AddrSpace.SYS, snooped=True, uncached=True)
      return HCQBuffer(vaddr, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=True, hMemory=paddrs[0]), view=memview, owner=self.dev)

    mapping = self.dev_impl.mm.valloc(size:=round_up(size, 0x1000), uncached=uncached, contiguous=cpu_access)
    barview = self.pci_dev.map_bar(bar=self.vram_bar, off=mapping.paddrs[0][0], size=mapping.size) if cpu_access else None
    return HCQBuffer(mapping.va_addr, size, view=barview, meta=PCIAllocationMeta(mapping, cpu_access, hMemory=mapping.paddrs[0][0]), owner=self.dev)

  def free(self, b:HCQBuffer):
    for dev in b.mapped_devs[1:]: dev.iface.dev_impl.mm.unmap_range(b.va_addr, b.size)
    if b.meta.mapping.aspace is AddrSpace.PHYS: self.dev_impl.mm.vfree(b.meta.mapping)
    if b.owner == self.dev and b.meta.has_cpu_mapping and not OSX: FileIOInterface.munmap(b.va_addr, b.size)

  def map(self, b:HCQBuffer):
    if b.owner is not None and b.owner._is_cpu():
      System.lock_memory(int(b.va_addr), b.size)
      paddrs, aspace = [(x, 0x1000) for x in System.system_paddrs(int(b.va_addr), round_up(b.size, 0x1000))], AddrSpace.SYS
      snooped, uncached = True, True
    elif (ifa:=getattr(b.owner, "iface", None)) is not None and isinstance(ifa, LNXPCIIfaceBase):
      snooped, uncached = True, b.meta.mapping.uncached
      if b.meta.mapping.aspace is AddrSpace.SYS: paddrs, aspace = b.meta.mapping.paddrs, AddrSpace.SYS
      elif hasattr(ifa.dev_impl, 'paddr2xgmi') and ifa.dev_impl.gmc.xgmi_seg_sz > 0:
        paddrs, aspace = [(ifa.dev_impl.paddr2xgmi(p), sz) for p, sz in b.meta.mapping.paddrs], AddrSpace.PEER
      else: paddrs, aspace = [(p + ifa.p2p_base_addr, sz) for p, sz in b.meta.mapping.paddrs], AddrSpace.SYS
    else: raise RuntimeError(f"map failed: {b.owner} -> {self.dev}")

    self.dev_impl.mm.map_range(int(b.va_addr), round_up(b.size, 0x1000), paddrs, aspace=aspace, snooped=snooped, uncached=uncached)

# *** Remote PCI Devices

class RemoteCmd(enum.IntEnum): MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, RESET, MMIO_READ, MMIO_WRITE = 1, 2, 3, 4, 5, 6, 7

class RemoteMMIOInterface(MMIOInterface):
  def __init__(self, dev:RemotePCIDevice, residx:int, nbytes:int, fmt='B', off=0):
    self.dev, self.residx, self.nbytes, self.fmt, self.off, self.el_sz = dev, residx, nbytes, fmt, off, struct.calcsize(fmt)

  def __getitem__(self, index):
    sl = index if isinstance(index, slice) else slice(index, index + 1)
    start, stop = (sl.start or 0) * self.el_sz, (sl.stop or len(self)) * self.el_sz
    data = self.dev._bulk_read(RemoteCmd.MMIO_READ, self.residx, self.off + start, stop - start)
    result = data if self.fmt == 'B' else list(struct.unpack(f'<{(stop - start) // self.el_sz}{self.fmt}', data))
    return result if isinstance(index, slice) else result[0]

  def __setitem__(self, index, val):
    start = (index.start or 0) * self.el_sz if isinstance(index, slice) else index * self.el_sz
    data = (val if self.fmt == 'B' else struct.pack(f'<{len(val)}{self.fmt}', *val)) if isinstance(index, slice) else struct.pack(f'<{self.fmt}', val)
    self.dev._bulk_write(RemoteCmd.MMIO_WRITE, self.residx, self.off + start, data)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return RemoteMMIOInterface(self.dev, self.residx, size or (self.nbytes - offset), fmt or self.fmt, self.off + offset)

class RemotePCIDevice(PCIDevice):
  def __init__(self, devpref:str, pcibus:str, bars:list[int], sock:socket.socket):
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    self.pcibus, self.sock = pcibus, sock
    for buft in [socket.SO_SNDBUF, socket.SO_RCVBUF]: self.sock.setsockopt(socket.SOL_SOCKET, buft, 64 << 20)
    self.bar_info = {b: PCIBarInfo(0, self._rpc(RemoteCmd.MAP_BAR, b)[0]) for b in bars}

  def _recvall(self, n:int) -> bytes:
    data = b''
    while len(data) < n and (chunk:=self.sock.recv(n - len(data))): data += chunk
    if len(data) < n: raise RuntimeError("Connection closed")
    return data

  def _recv_with_fd(self) -> tuple[bytes, int|None]:
    msg, anc, _, _ = self.sock.recvmsg(17, socket.CMSG_LEN(4))
    return msg, struct.unpack('<i', anc[0][2][:4])[0]

  def _rpc(self, cmd:int, *args:int, readout_size:int=0, has_fd=False) -> tuple[int, int, bytes|None, int|None]:
    self.sock.sendall(struct.pack('<BBQQQ', cmd, *(*args, 0, 0, 0, 0)[:4]))
    msg, fd = self._recv_with_fd() if has_fd else (self._recvall(17), None)
    if (resp:=struct.unpack('<BQQ', msg))[0] != 0:
      raise RuntimeError(f"RPC failed: {self._recvall(resp[1]).decode('utf-8') if resp[1] > 0 else 'unknown error'}")
    return (resp[1], resp[2]) + ((self._recvall(readout_size) if readout_size > 0 else None),) + (fd,)

  def _bulk_read(self, cmd:int, idx:int, offset:int, size:int) -> bytes: return unwrap(self._rpc(cmd, idx, offset, size, readout_size=size)[2])
  def _bulk_write(self, cmd:int, idx:int, offset:int, data:bytes): self.sock.sendall(struct.pack('<BBQQQ', cmd, idx, offset, len(data), 0) + data)

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False) -> tuple[MMIOInterface, list[int]]:
    mapped_size, _, _, fd = self._rpc(RemoteCmd.MAP_SYSMEM_FD, 0, 0, size, has_fd=True)
    memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), mapped_size, fmt='B')

    # paddrs are returned as (paddr, size) pairs until a (paddr=0, size=0) terminator in the beginning of the mapping.
    paddrs_raw = list(itertools.takewhile(lambda p: p[1] != 0, zip(memview.view(fmt='Q')[0::2], memview.view(fmt='Q')[1::2])))
    return memview, [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]
  def read_config(self, offset:int, size:int): return self._rpc(RemoteCmd.CFG_READ, 0, offset, size)[0]
  def write_config(self, offset:int, value:int, size:int): self._rpc(RemoteCmd.CFG_WRITE, 0, offset, size, value)
  def reset(self): self._rpc(RemoteCmd.RESET, 0, 0, 0)
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    return RemoteMMIOInterface(self, bar, size or self.bar_info[bar].size, fmt).view(off, size, fmt)

class APLRemotePCIDevice(RemotePCIDevice):
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"

  @staticmethod
  def install_tinygpu():
    print("Downloading TinyGPU.app...")
    system(f"ditto -xk {fetch('https://github.com/nimlgen/tinygpu_releases/raw/8120b5508b43149d27bf22f9a4e6d7c5a4b401e9/TinyGPU.zip')} /Applications")
    print(system(f"{APLRemotePCIDevice.APP_PATH} install"))

  def __init__(self, devpref:str, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    sock_path, sock = getenv("APL_REMOTE_SOCK", temp("tinygpu.sock")), socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for i in range(100):
      with contextlib.suppress(ConnectionRefusedError, FileNotFoundError):
        sock.connect(sock_path)
        break
      if i == 0: subprocess.Popen([self.APP_PATH, "server", sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      time.sleep(0.05)
    else: raise RuntimeError(f"Failed to connect to TinyGPU server at {sock_path}.")
    super().__init__(devpref, pcibus, bars, sock)

class APLRemoteIfaceBase(LNXPCIIfaceBase):
  def __init__(self, dev, dev_id, vendor, devices:list[tuple[int, list[int]]], bars, vram_bar, va_start, va_size, base_class:int|None=None):
    if not (cls:=type(self)).gpus:
      cls.gpus = System.pci_scan_bus(vendor, devices, base_class)
      if not cls.gpus: raise RuntimeError("No supported GPUs found")
      if not os.path.exists(APLRemotePCIDevice.APP_PATH): APLRemotePCIDevice.install_tinygpu()
    self.pci_dev = APLRemotePCIDevice(dev.__class__.__name__[:2], f'remote:{dev_id}', bars)
    self.dev, self.vram_bar = dev, vram_bar

  def free(self, b:HCQBuffer):
    for dev in b.mapped_devs[1:]: dev.iface.dev_impl.mm.unmap_range(b.va_addr, b.size)

  def map(self, b:HCQBuffer): raise RuntimeError(f"P2P mapping not supported for remote devices: {b.owner} -> {self.dev}")

PCIIfaceBase:type = APLRemoteIfaceBase if OSX else LNXPCIIfaceBase
