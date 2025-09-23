import os, mmap, array, functools, ctypes, select, contextlib, dataclasses, sys
from typing import cast, ClassVar
from tinygrad.helpers import round_up, to_mv, getenv, OSX, temp
from tinygrad.runtime.autogen import libc, vfio
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface, HCQBuffer
from tinygrad.runtime.support.memory import MemoryManager, VirtMapping

MAP_FIXED, MAP_LOCKED, MAP_POPULATE, MAP_NORESERVE = 0x10, 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000), 0x400

class _System:
  def reserve_hugepages(self, cnt): os.system(f"sudo sh -c 'echo {cnt} > /proc/sys/vm/nr_hugepages'")

  def memory_barrier(self): lib.atomic_thread_fence(__ATOMIC_SEQ_CST:=5) if (lib:=self.atomic_lib()) is not None else None

  def lock_memory(self, addr:int, size:int):
    if libc.mlock(ctypes.c_void_p(addr), size): raise RuntimeError(f"Failed to lock memory at {addr:#x} with size {size:#x}")

  def system_paddrs(self, vaddr:int, size:int) -> list[int]:
    self.pagemap().seek(vaddr // mmap.PAGESIZE * 8)
    return [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', self.pagemap().read(size//mmap.PAGESIZE*8, binary=True))]

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False, data:bytes|None=None) -> tuple[int, list[int]]:
    assert not contiguous or size <= (2 << 20), "Contiguous allocation is only supported for sizes up to 2MB"
    flags = (libc.MAP_HUGETLB if contiguous and (size:=round_up(size, mmap.PAGESIZE)) > 0x1000 else 0) | (MAP_FIXED if vaddr else 0)
    va = FileIOInterface.anon_mmap(vaddr, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS|MAP_POPULATE|MAP_LOCKED|flags, 0)

    if data is not None: to_mv(va, len(data))[:] = data
    return va, self.system_paddrs(va, size)

  def pci_reset(self, gpu): os.system(f"sudo sh -c 'echo 1 > /sys/bus/pci/devices/{gpu}/reset'")
  def pci_scan_bus(self, target_vendor:int, target_devices:list[int]) -> list[str]:
    result = []
    for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
      vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
      device = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
      if vendor == target_vendor and device in target_devices: result.append(pcibus)
    return sorted(result)

  @functools.cache
  def atomic_lib(self): return ctypes.CDLL(ctypes.util.find_library('atomic')) if sys.platform == "linux" else None

  @functools.cache
  def pagemap(self) -> FileIOInterface:
    if FileIOInterface(reloc_sysfs:="/proc/sys/vm/compact_unevictable_allowed", os.O_RDONLY).read()[0] != "0":
      os.system(cmd:=f"sudo sh -c 'echo 0 > {reloc_sysfs}'")
      assert FileIOInterface(reloc_sysfs, os.O_RDONLY).read()[0] == "0", f"Failed to disable migration of locked pages. Please run {cmd} manually."
    return FileIOInterface("/proc/self/pagemap", os.O_RDONLY)

  @functools.cache
  def vfio(self) -> FileIOInterface|None:
    try:
      if not FileIOInterface.exists("/sys/module/vfio"): os.system("sudo modprobe vfio-pci disable_idle_d3=1")

      FileIOInterface("/sys/module/vfio/parameters/enable_unsafe_noiommu_mode", os.O_RDWR).write("1")
      vfio_fd = FileIOInterface("/dev/vfio/vfio", os.O_RDWR)
      vfio.VFIO_CHECK_EXTENSION(vfio_fd, vfio.VFIO_NOIOMMU_IOMMU)

      return vfio_fd
    except OSError: return None

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
  def __init__(self, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    self.pcibus, self.irq_poller = pcibus, None

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", os.O_WRONLY).write(self.pcibus)

    for i in resize_bars or []:
      if FileIOInterface.exists(rpath:=f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize"):
        try: FileIOInterface(rpath, os.O_RDWR).write(str(int(FileIOInterface(rpath, os.O_RDONLY).read(), 16).bit_length() - 1))
        except OSError as e: raise RuntimeError(f"Cannot resize BAR {i}: {e}. Ensure the resizable BAR option is enabled on your system.") from e

    if getenv("VFIO", 0) and (vfio_fd:=System.vfio()) is not None:
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

    bar_info = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource", os.O_RDONLY).read().splitlines()
    self.bar_info = {j:(int(start,16), int(end,16), int(flgs,16)) for j,(start,end,flgs) in enumerate(l.split() for l in bar_info)}

  def read_config(self, offset:int, size:int): return int.from_bytes(self.cfg_fd.read(size, binary=True, offset=offset), byteorder='little')
  def write_config(self, offset:int, value:int, size:int): self.cfg_fd.write(value.to_bytes(size, byteorder='little'), binary=True, offset=offset)
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    fd, sz = self.bar_fds[bar], size or (self.bar_info[bar][1] - self.bar_info[bar][0] + 1)
    libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
    return MMIOInterface(loc, sz, fmt=fmt)

class PCIDevImplBase:
  mm: MemoryManager

@dataclasses.dataclass
class PCIAllocationMeta: mapping:VirtMapping; has_cpu_mapping:bool; hMemory:int=0 # noqa: E702

class PCIIfaceBase:
  dev_impl:PCIDevImplBase
  gpus:ClassVar[list[str]] = []

  def __init__(self, dev, dev_id, vendor, devices, bars, vram_bar, va_start, va_size):
    if len((cls:=type(self)).gpus) == 0:
      cls.gpus = System.pci_scan_bus(vendor, devices)
      visible_devices = [int(x) for x in (getenv('VISIBLE_DEVICES', '')).split(',') if x.strip()]
      cls.gpus = [cls.gpus[x] for x in visible_devices] if visible_devices else cls.gpus

      # Acquire va range to avoid collisions.
      FileIOInterface.anon_mmap(va_start, va_size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, 0)
    self.pci_dev, self.dev, self.vram_bar = PCIDevice(cls.gpus[dev_id], bars=bars, resize_bars=[vram_bar]), dev, vram_bar
    self.p2p_base_addr = self.pci_dev.bar_info[vram_bar][0]

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, **kwargs) -> HCQBuffer:
    if host or (uncached and cpu_access): # host or gtt-like memory.
      vaddr = self.dev_impl.mm.alloc_vaddr(size:=round_up(size, mmap.PAGESIZE), align=mmap.PAGESIZE)
      paddrs = [(paddr, mmap.PAGESIZE) for paddr in System.alloc_sysmem(size, vaddr=vaddr, contiguous=contiguous)[1]]
      mapping = self.dev_impl.mm.map_range(vaddr, size, paddrs, system=True, snooped=True, uncached=True)
      return HCQBuffer(vaddr, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=True, hMemory=paddrs[0][0]),
        view=MMIOInterface(mapping.va_addr, size, fmt='B'), owner=self.dev)

    mapping = self.dev_impl.mm.valloc(size:=round_up(size, 4 << 10), uncached=uncached, contiguous=cpu_access)
    if cpu_access: self.pci_dev.map_bar(bar=self.vram_bar, off=mapping.paddrs[0][0], addr=mapping.va_addr, size=mapping.size)
    return HCQBuffer(mapping.va_addr, size, view=MMIOInterface(mapping.va_addr, size, fmt='B') if cpu_access else None,
      meta=PCIAllocationMeta(mapping, has_cpu_mapping=cpu_access, hMemory=mapping.paddrs[0][0]), owner=self.dev)

  def free(self, b:HCQBuffer):
    for dev in b.mapped_devs[1:]: dev.iface.dev_impl.mm.unmap_range(b.va_addr, b.size)
    if not b.meta.mapping.system: self.dev_impl.mm.vfree(b.meta.mapping)
    if b.owner == self.dev and b.meta.has_cpu_mapping: FileIOInterface.munmap(b.va_addr, b.size)

  def map(self, b:HCQBuffer):
    if b.owner is not None and b.owner._is_cpu():
      System.lock_memory(cast(int, b.va_addr), b.size)
      paddrs, snooped, uncached = [(x, 0x1000) for x in System.system_paddrs(cast(int, b.va_addr), round_up(b.size, 0x1000))], True, False
    elif (ifa:=getattr(b.owner, "iface", None)) is not None and isinstance(ifa, PCIIfaceBase):
      paddrs = [(paddr if b.meta.mapping.system else (paddr + ifa.p2p_base_addr), size) for paddr,size in b.meta.mapping.paddrs]
      snooped, uncached = b.meta.mapping.snooped, b.meta.mapping.uncached
    else: raise RuntimeError(f"map failed: {b.owner} -> {self.dev}")

    self.dev_impl.mm.map_range(cast(int, b.va_addr), round_up(b.size, 0x1000), paddrs, system=True, snooped=snooped, uncached=uncached)
