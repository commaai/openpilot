from __future__ import annotations
import os, mmap, array, functools, ctypes, ctypes.util, select, contextlib, dataclasses, sys, struct, socket, enum, itertools
from tinygrad.device import BufferStorage, Buffer, Device
from tinygrad.helpers import round_up, getenv, OSX, temp, DEBUG, pluralize
from tinygrad.runtime.autogen import libc, pci, vfio
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface, hcq_filter_visible_devices
from tinygrad.runtime.support.memory import VirtMapping, AddrSpace, BumpAllocator
from tinygrad.runtime.support.usb import USB3, CustomASM24Controller, USBMMIOInterface

MAP_FIXED, MAP_FIXED_NOREPLACE = 0x10, 0x100000
MAP_LOCKED, MAP_POPULATE, MAP_NORESERVE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000), 0x400

def ipv4_to_gid(ip:str) -> bytes: return bytes(10) + b'\xff\xff' + socket.inet_aton(ip)

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

  @functools.cache
  def reserve_va(self, va_start, va_size):
    # cached, runs only once per range. used to not collide with other mappings.
    FileIOInterface.anon_mmap(va_start, va_size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, 0)

  def memory_barrier(self): lib.atomic_thread_fence(__ATOMIC_SEQ_CST:=5) if (lib:=self.libsys if OSX else self.atomic_lib) is not None else None

  def lock_memory(self, addr:int, size:int):
    if libc.mlock(ctypes.c_void_p(addr), size): raise RuntimeError(f"Failed to lock memory at {addr:#x} with size {size:#x}")

  def system_paddrs(self, vaddr:int, size:int) -> list[int]:
    self.pagemap.seek(vaddr // mmap.PAGESIZE * 8)
    return [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', self.pagemap.read(size//mmap.PAGESIZE*8, binary=True))]

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False) -> tuple[MMIOInterface, list[int]]:
    assert not contiguous or size <= (2 << 20), "Contiguous allocation is only supported for sizes up to 2MB"
    flags = (libc.MAP_HUGETLB if contiguous and (size:=round_up(size, mmap.PAGESIZE)) > mmap.PAGESIZE else 0) | (MAP_FIXED if vaddr else 0)
    va = FileIOInterface.anon_mmap(vaddr, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS|MAP_POPULATE|MAP_LOCKED|flags, 0)
    return MMIOInterface(va, size), self.system_paddrs(va, round_up(size, mmap.PAGESIZE))

  def pci_scan_bus(self, vendor:int, devices:tuple[tuple[int, tuple[int, ...]], ...], base_class:int|None=None) -> list[str]:
    all_devs = []
    if OSX:
      from tinygrad.runtime.autogen import iokit, corefoundation
      def read_prop(svc, key) -> int:
        cfkey = corefoundation.CFStringCreateWithCString(None, key.encode(), corefoundation.kCFStringEncodingUTF8)
        cfdata = ctypes.cast(iokit.IORegistryEntryCreateCFProperty(svc, ctypes.cast(cfkey, iokit.CFStringRef), None, 0), corefoundation.CFDataRef)
        corefoundation.CFDataGetBytes(cfdata, corefoundation.CFRange(0, corefoundation.CFDataGetLength(cfdata)), buf:=(ctypes.c_uint8*8)())
        return int.from_bytes(bytes(buf), "little")

      iokit.IOServiceGetMatchingServices(0, iokit.IOServiceMatching(b"IOPCIDevice"), ctypes.byref(iterator:=ctypes.c_uint()))
      while svc:=iokit.IOIteratorNext(iterator):
        if base_class is not None and read_prop(svc, "class-code") >> 16 != base_class: continue
        all_devs.append((v:=read_prop(svc, "vendor-id"), d:=read_prop(svc, "device-id"), f"{v:x}:{d:x}"))
    else:
      try: devs = FileIOInterface("/sys/bus/pci/devices")
      except FileNotFoundError: raise RuntimeError("no pcie")
      for pcibus in devs.listdir():
        if base_class is not None and int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/class").read(), 16) >> 16 != base_class: continue
        all_devs.append((int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16),
                         int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16), pcibus))

    return sorted([val for vndr, device, val in all_devs if vndr == vendor and any((device & mask) in devlist for mask, devlist in devices)])

  @functools.cache
  def list_devices(self, vendor:int, devices:tuple[tuple[int, tuple[int, ...]], ...], base_class:int|None=None):
    if getenv("REMOTE", ""): return [(functools.partial(RemotePCIDevice, sock=s), x) for s, x in RemotePCIDevice.scan(vendor, devices, base_class)]
    return [(PCIDevice, x) for x in System.pci_scan_bus(vendor, devices, base_class)]

  def pci_probe_device(self, device:str, dev_id:int, vendor:int, devices:tuple[tuple[int, tuple[int, ...]], ...], base_class:int|None=None):
    try: cl, pcibus = (ds:=hcq_filter_visible_devices(self.list_devices(vendor, devices, base_class), device))[dev_id]
    except IndexError: raise RuntimeError(f"{device}:{dev_id} does not exist ({pluralize('device', len(ds))} available)")
    return cl(device[:2], pcibus)

  def pci_setup_usb_bars(self, usb:CustomASM24Controller, gpu_bus:int, mem_base:int, pref_mem_base:int) -> dict[int, tuple[int, int]]:
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

        bars[bar_off // 4] = (mem_space_addr[bar_mem], bar_size)
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
  def __init__(self, devpref:str, pcibus:str):
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    self.pcibus, self.irq_poller = pcibus, None

    try: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/enable", os.O_RDWR)
    except PermissionError: raise PermissionError(f"Cannot access PCI device {pcibus}: run `extra/amdpci/setup_python_cap.sh` or use sudo")

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", os.O_WRONLY).write(self.pcibus)
    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"): raise RuntimeError(f"Driver is bound to {pcibus}")

    # remove sibling functions of the gpu, if any
    for fn in range(1, 8):
      if FileIOInterface.exists(sib:=f"/sys/bus/pci/devices/{self.pcibus[:-1]}{fn}"): FileIOInterface(f"{sib}/remove", os.O_WRONLY).write("1")

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
        argsz=ctypes.sizeof(vfio.struct_vfio_irq_set) + ctypes.sizeof(ctypes.c_int), count=1)
      vfio.VFIO_DEVICE_SET_IRQS(self.vfio_dev, (ctypes.c_byte * irqs.argsz).from_buffer(bytearray(bytes(irqs)) + struct.pack('i', self.irq_fd.fd)))
    else: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/enable", os.O_RDWR).write("1")

    self.cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False) -> tuple[MMIOInterface, list[int]]:
    return System.alloc_sysmem(size, vaddr, contiguous)

  def reset(self): os.system(f"sudo sh -c 'echo 1 > /sys/bus/pci/devices/{self.pcibus}/reset'")
  def read_config(self, offset:int, size:int): return int.from_bytes(self.cfg_fd.read(size, binary=True, offset=offset), byteorder='little')
  def write_config(self, offset:int, value:int, size:int): self.cfg_fd.write(value.to_bytes(size, byteorder='little'), binary=True, offset=offset)
  def write_config_flush(self, offset:int, value:int, size:int):
    self.write_config(offset, value, size)
    self.read_config(offset, size)

  @functools.cache
  def bar_fd(self, bar_idx:int) -> FileIOInterface:
    return FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{bar_idx}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
  @functools.cache
  def bar_info(self, bar_idx:int) -> tuple[int, int]:
    s, e, _ = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource", os.O_RDONLY).read().splitlines()[bar_idx].split()
    return (int(s, 16), int(e, 16) - int(s, 16) + 1)
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    fd, sz = self.bar_fd(bar), size or (self.bar_info(bar)[1] - off)
    libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
    return MMIOInterface(loc, sz, fmt=fmt)
  def resize_bar(self, bar_idx:int):
    rpath = f"/sys/bus/pci/devices/{self.pcibus}/resource{bar_idx}_resize"
    try: FileIOInterface(rpath, os.O_RDWR).write(str(int(FileIOInterface(rpath, os.O_RDONLY).read(), 16).bit_length() - 1))
    except OSError as e: raise RuntimeError(f"Cannot resize BAR {bar_idx}: {e}. Ensure the resizable BAR option is enabled.") from e

class USBPCIDevice(PCIDevice):
  def __init__(self, devpref:str, dev, pcibus):
    self.pcibus, self.peer_group = pcibus, f"USBPCIDevice_{pcibus}"
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    usb = USB3(dev)
    if DEBUG >= 1: print(f"am {self.pcibus}: product string: {usb.product!r}")
    self.usb: CustomASM24Controller = CustomASM24Controller(usb)
    self._bar_info = System.pci_setup_usb_bars(self.usb, gpu_bus=4, mem_base=0x10000000, pref_mem_base=(32 << 30))
    self.sram = BumpAllocator(size=0x80000, wrap=False) # asm24 controller sram

  def dma_view(self, ctrl_addr, size): return USBMMIOInterface(self.usb, ctrl_addr, size, fmt='B', pcimem=False)
  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False) -> tuple[MMIOInterface, list[int]]:
    return self.dma_view(0xf000 + (off:=self.sram.alloc(size)), size), [0x200000 + off]

  def read_config(self, offset:int, size:int): return self.usb.pcie_cfg_req(offset, bus=4, dev=0, fn=0, size=size)
  def write_config(self, offset:int, value:int, size:int): self.usb.pcie_cfg_req(offset, bus=4, dev=0, fn=0, value=value, size=size)

  def bar_info(self, bar_idx:int) -> tuple[int, int]: return self._bar_info[bar_idx]  # type: ignore[override]
  def map_bar(self, bar, off=0, addr=0, size=None, fmt='B'):
    return USBMMIOInterface(self.usb, self.bar_info(bar)[0] + off, size or self.bar_info(bar)[1], fmt)
  def resize_bar(self, bar_idx:int): pass # already resized

@dataclasses.dataclass
class PCIAllocationMeta: mapping:VirtMapping; has_cpu_mapping:bool; hMemory:int=0 # noqa: E702

class PCIIfaceBase:
  @property
  def peer_group(self) -> str: return getattr(self.pci_dev, 'peer_group', type(self.pci_dev).__name__)
  @property
  def remote(self) -> RemotePCIDevice|None: return self.pci_dev if isinstance(self.pci_dev, RemotePCIDevice) else None
  def is_bar_small(self) -> bool: return self.pci_dev.bar_info(self.vram_bar)[1] == (256 << 20)

  def __init__(self, dev, dev_id, vendor, devices:tuple[tuple[int, tuple[int, ...]], ...], vram_bar, va_start, va_size,
               dev_impl_t, base_class:int|None=None):
    self.pci_dev = System.pci_probe_device(dn:=dev.__class__.__name__[:-6], dev_id, vendor, devices, base_class=base_class)
    if self.remote is None: System.reserve_va(va_start, va_size)
    with contextlib.suppress(Exception): self.pci_dev.resize_bar(vram_bar)
    self.dev_impl = dev_impl_t(self.pci_dev)
    self.dev, self.vram_bar, self.count = dev, vram_bar, len(hcq_filter_visible_devices(System.list_devices(vendor, devices, base_class), dn))

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False, zero=False,
            **kwargs) -> BufferStorage:
    should_use_sysmem = host or (cpu_access and self.is_bar_small() and not force_devmem)

    # Align size to huge pages for large allocations, otherwise the unaligned tail falls back to 4KB pages, increasing TLB pressure.
    size = round_up(size, mmap.PAGESIZE if should_use_sysmem else ((2 << 20) if size >= (8 << 20) else (4 << 10)))

    if should_use_sysmem:
      vaddr = self.dev_impl.mm.alloc_vaddr(size:=round_up(size, mmap.PAGESIZE), align=mmap.PAGESIZE)
      memview, paddrs = self.pci_dev.alloc_sysmem(size, vaddr=vaddr, contiguous=contiguous)
      mapping = self.dev_impl.mm.map_range(vaddr, size, [(paddr, 0x1000) for paddr in paddrs], aspace=AddrSpace.SYS, snooped=True, uncached=True)
      return BufferStorage(vaddr, PCIAllocationMeta(mapping, has_cpu_mapping=True, hMemory=paddrs[0]), memview)

    mapping = self.dev_impl.mm.valloc(size:=round_up(size, 0x1000), uncached=uncached, contiguous=cpu_access, zero=zero)
    barview = self.pci_dev.map_bar(bar=self.vram_bar, off=mapping.paddrs[0][0], size=mapping.size) if cpu_access else None
    return BufferStorage(mapping.va_addr, PCIAllocationMeta(mapping, cpu_access, hMemory=mapping.paddrs[0][0]), barview)

  def free(self, storage:BufferStorage):
    if storage.meta.mapping.aspace is AddrSpace.PHYS: self.dev_impl.mm.vfree(storage.meta.mapping)
    if storage.meta.has_cpu_mapping and self.remote is None: FileIOInterface.munmap(storage.buf, storage.meta.mapping.size)

  def unmap(self, mapping:BufferStorage): self.dev_impl.mm.unmap_range(*mapping.meta)

  def p2p_paddrs(self, paddrs:list[tuple[int,int]]) -> tuple[list[tuple[int,int]], AddrSpace]:
    return [(p + self.pci_dev.bar_info(self.vram_bar)[0], sz) for p, sz in paddrs], AddrSpace.SYS

  def map(self, b:Buffer) -> BufferStorage:
    if b.device.split(":")[0] in {"CPU", "PYTHON", "NPY"}:
      if self.dev.host != Device[b.device].host: raise RuntimeError(f"host memory is not on the node of {self.dev.device}")
      if b._buf % 0x1000: raise RuntimeError("Host mapping requires a page-aligned address")
      lo, size = b._buf, round_up(b.nbytes, 0x1000)
      if not self.dev_impl.mm.va_base <= lo < lo + size <= self.dev_impl.mm.va_base + (1 << self.dev_impl.mm.va_bits):
        raise RuntimeError(f"Host address {lo:#x} is outside the GPU virtual address range")
      if self.remote is None: System.lock_memory(lo, size)
      paddrs = [(x, 0x1000) for x in (b.meta if self.remote is not None else System.system_paddrs(lo, size))]
      aspace, snooped, uncached = AddrSpace.SYS, True, True
    elif isinstance(ifa:=getattr(Device[b.device], "iface", None), PCIIfaceBase):
      if ifa.is_bar_small(): raise RuntimeError(f"P2P mapping not supported for small bar devices: {b.device} -> {self.dev.device}")
      if ifa.peer_group != self.peer_group: raise RuntimeError(f"P2P mapping across peer groups: {b.device} -> {self.dev.device}")
      lo, size, snooped, uncached = b._buf, b.meta.mapping.size, True, b.meta.mapping.uncached
      if b.meta.mapping.aspace is AddrSpace.SYS: paddrs, aspace = b.meta.mapping.paddrs, AddrSpace.SYS
      else: paddrs, aspace = ifa.p2p_paddrs(b.meta.mapping.paddrs)
    else: raise RuntimeError(f"map failed: {b.device} -> {self.dev.device}")

    self.dev_impl.mm.map_range(lo, size, paddrs, aspace=aspace, snooped=snooped, uncached=uncached)
    return BufferStorage(b._buf, (lo, size))

# *** Remote PCI: memory and program execution on extra/remote/serve.py

class RemoteCmd(enum.IntEnum):
  PROBE,MAP_BAR,MAP_SYSMEM_FD,CFG_READ,CFG_WRITE,RESET,MMIO_READ,MMIO_WRITE,MAP_SYSMEM,SYSMEM_READ,SYSMEM_WRITE,RESIZE_BAR,PING = range(13)
  LOAD_PROG,EXEC_PROG,UNMAP_SYSMEM = range(13, 16)

REMOTE_REQ, REMOTE_RESP = '<BIIQQQ', '<BQQ' # (cmd, dev, bar, arg0, arg1, arg2) and (status, resp0, resp1)

class RemoteMMIOInterface(MMIOInterface):
  # memory of the node at its address: reads are round trips, writes are posted in order
  def __init__(self, dev:RemotePCIDevice, residx:int, addr:int, nbytes:int, fmt='B', off=0, sysmem=False):
    self.dev, self.residx, self.addr, self.nbytes, self.fmt, self.off, self.sysmem = dev, residx, addr, nbytes, fmt, off, sysmem
  def __getitem__(self, k):
    sl, el = k if isinstance(k, slice) else slice(k, k + 1), struct.calcsize(self.fmt)
    st, en = (sl.start or 0) * el, (len(self) if sl.stop is None else sl.stop) * el
    cmd, addr = (RemoteCmd.SYSMEM_READ, self.addr) if self.sysmem else (RemoteCmd.MMIO_READ, self.off)
    data = self.dev.rpc(cmd, addr + st, en - st, bar=self.residx, readout_size=en - st)[2]
    res = data if self.fmt == 'B' else list(struct.unpack(f'<{(en - st) // el}{self.fmt}', data))
    return res if isinstance(k, slice) else res[0]
  def __setitem__(self, k, v):
    st = ((k.start or 0) if isinstance(k, slice) else k) * struct.calcsize(self.fmt)
    data = (bytes(v) if self.fmt == 'B' else struct.pack(f'<{len(v)}{self.fmt}', *v)) if isinstance(k, slice) else struct.pack(f'<{self.fmt}', v)
    cmd = RemoteCmd.SYSMEM_WRITE if self.sysmem else RemoteCmd.MMIO_WRITE
    self.dev._post(self.dev.sock, cmd, (self.addr if self.sysmem else self.off) + st, len(data), dev=self.dev.dev_id, bar=self.residx, payload=data)
  def view(self, offset:int=0, size:int|None=None, fmt=None) -> MMIOInterface:
    return RemoteMMIOInterface(self.dev, self.residx, self.addr + offset, (self.nbytes - offset) if size is None else size,
                               fmt or self.fmt, self.off + offset, self.sysmem)

class RemotePCIDevice(PCIDevice):
  @staticmethod
  def connect(host:str, port:int) -> socket.socket: return RemotePCIDevice._connect(socket.gethostbyname(host), port)
  @staticmethod
  @functools.cache
  def _connect(host:str, port:int) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=getenv("REMOTE_TIMEOUT", 60))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF): sock.setsockopt(socket.SOL_SOCKET, opt, 64 << 20)
    return sock

  @staticmethod
  def scan(vendor:int, devices:tuple[tuple[int, tuple[int, ...]], ...], base_class:int|None) -> list[tuple[socket.socket, str]]:
    payload, ret = array.array('I', itertools.chain.from_iterable((m, d) for m, ds in devices for d in ds)).tobytes(), []
    for r in [r.strip() for r in getenv("REMOTE", "").split(",") if r.strip()]:
      host, port = r.split(":")[0], int(r.split(":")[1]) if ":" in r else 6667
      sock = RemotePCIDevice.connect(host, port)
      n, _, _ = RemotePCIDevice._rpc(sock, RemoteCmd.PROBE, base_class or 0, len(payload), vendor, payload=payload)
      data = RemotePCIDevice._recvall(sock, n)
      ret += [(sock, f"remote:{host}:{port}:{d}") for d in data.decode().split()]
    return ret

  @staticmethod
  def _recvall(sock:socket.socket, n:int) -> bytes:
    data = bytearray()
    while len(data) < n:
      if not (chunk:=sock.recv(n - len(data))): raise ConnectionError("remote connection closed")
      data += chunk
    return bytes(data)
  @staticmethod
  def _post(sock:socket.socket, cmd:RemoteCmd, *args:int, dev:int=0, bar:int=0, payload:bytes|memoryview=b''):
    sock.sendall(struct.pack(REMOTE_REQ, cmd, dev, bar, *(*args, 0, 0, 0)[:3]))
    if len(payload): sock.sendall(payload)
  @staticmethod
  def _rpc(sock:socket.socket, cmd:RemoteCmd, *args:int, dev:int=0, bar:int=0, payload:bytes|memoryview=b'',
           readout_size:int=0) -> tuple[int, int, bytes]:
    RemotePCIDevice._post(sock, cmd, *args, dev=dev, bar=bar, payload=payload)
    status, r0, r1 = struct.unpack(REMOTE_RESP, RemotePCIDevice._recvall(sock, struct.calcsize(REMOTE_RESP)))
    if status: raise RuntimeError(f"remote {cmd.name} failed: {RemotePCIDevice._recvall(sock, r0).decode()}")
    return r0, r1, RemotePCIDevice._recvall(sock, readout_size)

  def __init__(self, devpref:str, pcibus:str, sock:socket.socket):
    self.sock, self.pcibus, self.dev_id, self.irq_poller = sock, pcibus, int(pcibus.split(':')[-1]), None
    self.peer_group = "remote:%s:%d" % sock.getpeername()[:2]

  def rpc(self, cmd:RemoteCmd, *args:int, bar:int=0, payload:bytes|memoryview=b'', readout_size:int=0) -> tuple[int, int, bytes]:
    return RemotePCIDevice._rpc(self.sock, cmd, *args, dev=self.dev_id, bar=bar, payload=payload, readout_size=readout_size)

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False) -> tuple[MMIOInterface, list[int]]:
    n, _, _ = self.rpc(RemoteCmd.MAP_SYSMEM, size, int(contiguous), vaddr)
    host_va, *paddrs = struct.unpack(f'<{n // 8}Q', self._recvall(self.sock, n))
    return self.cpu_view(host_va, size), paddrs
  def free_sysmem(self, view:MMIOInterface): self.rpc(RemoteCmd.UNMAP_SYSMEM, view.addr, view.nbytes)
  def cpu_view(self, addr:int, size:int) -> MMIOInterface: return RemoteMMIOInterface(self, 0, addr, size, sysmem=True)
  def reset(self): self.rpc(RemoteCmd.RESET)
  def read_config(self, offset:int, size:int): return self.rpc(RemoteCmd.CFG_READ, offset, size)[0]
  def write_config(self, offset:int, value:int, size:int): self.rpc(RemoteCmd.CFG_WRITE, offset, size, value)
  def resize_bar(self, bar_idx:int): self.rpc(RemoteCmd.RESIZE_BAR, bar=bar_idx)
  @functools.cache
  def _bar(self, bar:int) -> tuple[int, int, int]: # paddr, size, the node's address of the whole bar
    paddr, size, data = self.rpc(RemoteCmd.MAP_BAR, bar=bar, readout_size=8)
    return paddr, size, struct.unpack('<Q', data)[0]
  def bar_info(self, bar_idx:int) -> tuple[int, int]: return self._bar(bar_idx)[:2]  # type: ignore[override]
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    _, sz, host_va = self._bar(bar)
    return RemoteMMIOInterface(self, bar, host_va + off, size or (sz - off), fmt, off=off)
