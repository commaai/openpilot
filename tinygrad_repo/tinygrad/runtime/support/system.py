import os, mmap, array, functools, ctypes, select, contextlib, dataclasses, sys, errno, itertools
from typing import cast, ClassVar
from tinygrad.helpers import round_up, getenv, OSX, temp, ceildiv
from tinygrad.runtime.autogen import libc, vfio, pci
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface, HCQBuffer, hcq_filter_visible_devices
from tinygrad.runtime.support.memory import MemoryManager, VirtMapping
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
  def iokit(self): return ctypes.CDLL(ctypes.util.find_library("IOKit"))

  @functools.cached_property
  def libsys(self): return ctypes.CDLL(ctypes.util.find_library("System"))

  @functools.cached_property
  def mach_task_self(self): return ctypes.cast(self.libsys.mach_task_self_, ctypes.POINTER(ctypes.c_uint)).contents.value

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

  @functools.cached_property
  def macos_tinygpu_conn(self):
    self.iokit.IOServiceNameMatching.restype = ctypes.c_void_p # CFMutableDictionaryRef
    if not (mdict:=self.iokit.IOServiceNameMatching("tinygpu".encode("utf-8"))): raise RuntimeError("IOServiceNameMatching returned NULL")
    if not (service:=self.iokit.IOServiceGetMatchingService(ctypes.c_uint(0), ctypes.c_void_p(mdict))):
      raise RuntimeError('Service "tinygpu" is not running')
    if self.iokit.IOServiceOpen(service, self.mach_task_self, ctypes.c_uint32(0), ctypes.byref(conn:=ctypes.c_uint(0))):
      raise RuntimeError("IOServiceOpen failed")
    return conn

  def iokit_pci_memmap(self, typ:int):
    if self.iokit.IOConnectMapMemory64(self.macos_tinygpu_conn, ctypes.c_uint32(typ), System.mach_task_self,
      ctypes.byref(addr:=ctypes.c_uint64(0)), ctypes.byref(size:=ctypes.c_uint64(0)), 0x1): raise RuntimeError(f"IOConnectMapMemory64({typ=}) failed")
    return MMIOInterface(addr.value, size.value)

  def iokit_pci_rpc(self, sel:int, *args:int):
    in_scalars = (ctypes.c_uint64 * len(args))(*args) if args else ctypes.POINTER(ctypes.c_uint64)()
    if (self.iokit.IOConnectCallMethod(self.macos_tinygpu_conn, sel, in_scalars, len(args), None, ctypes.c_size_t(0),
        out_scalars:=(ctypes.c_uint64*16)(), ctypes.byref(outcnt:=ctypes.c_uint32(16)), None, ctypes.byref(ctypes.c_size_t(0)))):
      raise RuntimeError(f"IOConnectCallMethod({sel=}, {args=}) failed")
    return out_scalars[:outcnt.value]

  def reserve_hugepages(self, cnt): os.system(f"sudo sh -c 'echo {cnt} > /proc/sys/vm/nr_hugepages'")

  def memory_barrier(self): lib.atomic_thread_fence(__ATOMIC_SEQ_CST:=5) if (lib:=self.libsys if OSX else self.atomic_lib) is not None else None

  def lock_memory(self, addr:int, size:int):
    if libc.mlock(ctypes.c_void_p(addr), size): raise RuntimeError(f"Failed to lock memory at {addr:#x} with size {size:#x}")

  def system_paddrs(self, vaddr:int, size:int) -> list[int]:
    self.pagemap.seek(vaddr // mmap.PAGESIZE * 8)
    return [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', self.pagemap.read(size//mmap.PAGESIZE*8, binary=True))]

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False, data:bytes|None=None) -> tuple[MMIOInterface, list[int]]:
    if OSX:
      sysmem_view = System.iokit_pci_memmap(round_up(size, mmap.PAGESIZE))
      paddrs = list(itertools.takewhile(lambda p: p[1] != 0, zip(sysmem_view.view(fmt='Q')[0::2], sysmem_view.view(fmt='Q')[1::2])))
      assert not contiguous or len(paddrs) == 1, "not contiguous, but required"
    else:
      assert not contiguous or size <= (2 << 20), "Contiguous allocation is only supported for sizes up to 2MB"
      flags = (libc.MAP_HUGETLB if contiguous and (size:=round_up(size, mmap.PAGESIZE)) > mmap.PAGESIZE else 0) | (MAP_FIXED if vaddr else 0)
      va = FileIOInterface.anon_mmap(vaddr, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS|MAP_POPULATE|MAP_LOCKED|flags, 0)
      sysmem_view, paddrs = MMIOInterface(va, size), [(x, mmap.PAGESIZE) for x in self.system_paddrs(va, size)]

    if data is not None: sysmem_view[:len(data)] = data
    return sysmem_view, [p + i for p, sz in paddrs for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]

  def pci_scan_bus(self, target_vendor:int, target_devices:list[tuple[int, list[int]]]) -> list[str]:
    result = []
    for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
      vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
      device = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
      if vendor == target_vendor and any((device & mask) in devlist for mask, devlist in target_devices): result.append(pcibus)
    return sorted(result)

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
    except OSError: raise RuntimeError(f"Failed to take lock file {name}. It's already in use.")

    return self.lock_fd

System = _System()

class PCIDevice:
  def __init__(self, devpref:str, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    self.pcibus, self.irq_poller = pcibus, None

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", os.O_WRONLY).write(self.pcibus)

    for i in resize_bars or []:
      if FileIOInterface.exists(rpath:=f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize"):
        try: FileIOInterface(rpath, os.O_RDWR).write(str(int(FileIOInterface(rpath, os.O_RDONLY).read(), 16).bit_length() - 1))
        except OSError as e:
          if e.errno in {errno.EPERM, errno.EACCES}:
            raise RuntimeError(f"Cannot resize BAR {i}: {e}. Permission error: run `extra/amdpci/setup_python_cap.sh`"
                                " to allow python accessing device or run with sudo") from e
          raise RuntimeError(f"Cannot resize BAR {i}: {e}. Ensure the resizable BAR option is enabled on your system.") from e

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

  def read_config(self, offset:int, size:int): return int.from_bytes(self.cfg_fd.read(size, binary=True, offset=offset), byteorder='little')
  def write_config(self, offset:int, value:int, size:int): self.cfg_fd.write(value.to_bytes(size, byteorder='little'), binary=True, offset=offset)
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    fd, sz = self.bar_fds[bar], size or (self.bar_info[bar].size - off)
    libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
    return MMIOInterface(loc, sz, fmt=fmt)
  def reset(self): os.system(f"sudo sh -c 'echo 1 > /sys/bus/pci/devices/{self.pcibus}/reset'")

class APLPCIDevice(PCIDevice):
  def __init__(self, devpref:str, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    self.pcibus, self.bars = pcibus, {b: System.iokit_pci_memmap(b) for b in bars}
    self.bar_info = {b:PCIBarInfo(0, self.bars[b].nbytes-1 if b in self.bars else 0) for b in range(6)} # NOTE: fake bar info for nv.
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface: return self.bars[bar].view(off, size, fmt)
  def read_config(self, offset:int, size:int): return System.iokit_pci_rpc(__TinyGPURPCReadCfg:=0, offset, size)[0]
  def write_config(self, offset:int, value:int, size:int): System.iokit_pci_rpc(__TinyGPURPCWriteCfg:=1, offset, size, value)
  def reset(self): System.iokit_pci_rpc(__TinyGPURPCReset:=2)

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

  def __init__(self, dev, dev_id, vendor, devices:list[tuple[int, list[int]]], bars, vram_bar, va_start, va_size):
    if len((cls:=type(self)).gpus) == 0:
      cls.gpus = hcq_filter_visible_devices(System.pci_scan_bus(vendor, devices))

      # Acquire va range to avoid collisions.
      FileIOInterface.anon_mmap(va_start, va_size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, 0)
    self.pci_dev, self.dev, self.vram_bar = PCIDevice(dev.__class__.__name__[:2], cls.gpus[dev_id], bars=bars, resize_bars=[vram_bar]), dev, vram_bar
    self.p2p_base_addr = self.pci_dev.bar_info[vram_bar].addr

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False, **kwargs) -> HCQBuffer:
    # NOTE: logic on macos is different, since bar is small
    should_use_sysmem = host or ((cpu_access if OSX else (uncached and cpu_access)) and not force_devmem)
    if should_use_sysmem:
      vaddr = self.dev_impl.mm.alloc_vaddr(size:=round_up(size, mmap.PAGESIZE), align=mmap.PAGESIZE)
      memview, paddrs = System.alloc_sysmem(size, vaddr=vaddr, contiguous=contiguous)
      mapping = self.dev_impl.mm.map_range(vaddr, size, [(paddr, 0x1000) for paddr in paddrs], system=True, snooped=True, uncached=True)
      return HCQBuffer(vaddr, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=True, hMemory=paddrs[0]), view=memview, owner=self.dev)

    mapping = self.dev_impl.mm.valloc(size:=round_up(size, 0x1000), uncached=uncached, contiguous=cpu_access)
    barview = self.pci_dev.map_bar(bar=self.vram_bar, off=mapping.paddrs[0][0], size=mapping.size) if cpu_access else None
    return HCQBuffer(mapping.va_addr, size, view=barview, meta=PCIAllocationMeta(mapping, cpu_access, hMemory=mapping.paddrs[0][0]), owner=self.dev)

  def free(self, b:HCQBuffer):
    for dev in b.mapped_devs[1:]: dev.iface.dev_impl.mm.unmap_range(b.va_addr, b.size)
    if not b.meta.mapping.system: self.dev_impl.mm.vfree(b.meta.mapping)
    if b.owner == self.dev and b.meta.has_cpu_mapping and not OSX: FileIOInterface.munmap(b.va_addr, b.size)

  def map(self, b:HCQBuffer):
    if b.owner is not None and b.owner._is_cpu():
      System.lock_memory(cast(int, b.va_addr), b.size)
      paddrs, snooped, uncached = [(x, 0x1000) for x in System.system_paddrs(cast(int, b.va_addr), round_up(b.size, 0x1000))], True, True
    elif (ifa:=getattr(b.owner, "iface", None)) is not None and isinstance(ifa, LNXPCIIfaceBase):
      paddrs = [(paddr if b.meta.mapping.system else (paddr + ifa.p2p_base_addr), size) for paddr,size in b.meta.mapping.paddrs]
      snooped, uncached = b.meta.mapping.snooped, b.meta.mapping.uncached
    else: raise RuntimeError(f"map failed: {b.owner} -> {self.dev}")

    self.dev_impl.mm.map_range(cast(int, b.va_addr), round_up(b.size, 0x1000), paddrs, system=True, snooped=snooped, uncached=uncached)

class APLPCIIfaceBase(LNXPCIIfaceBase):
  def __init__(self, dev, dev_id, vendor, devices, bars, vram_bar, va_start, va_size):
    self.pci_dev, self.dev, self.vram_bar = APLPCIDevice(dev.__class__.__name__[:2], pcibus=f'usb4:{dev_id}', bars=bars), dev, vram_bar
    assert (read_vendor:=self.pci_dev.read_config(0x00, 2)) == vendor, f"Vendor ID mismatch: expected {vendor:#x}, got {read_vendor:#x}"
  def map(self, b:HCQBuffer): raise RuntimeError(f"map failed: {b.owner} -> {self.dev}")

PCIIfaceBase:type = APLPCIIfaceBase if OSX else LNXPCIIfaceBase
