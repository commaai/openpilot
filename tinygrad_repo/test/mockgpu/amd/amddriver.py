import pathlib, re, ctypes, mmap, collections, functools, copy, os
import tinygrad.runtime.autogen.kfd as kfd
import tinygrad.runtime.autogen.am.am as am
from tinygrad.helpers import from_mv
from test.mockgpu.driver import VirtDriver, VirtFileDesc, TextFileDesc, DirFileDesc, VirtFile
from test.mockgpu.amd.amdgpu import AMDGPU, gpu_props

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

def ioctls_from_header():
  # hdrpy = (pathlib.Path(__file__).parent.parent.parent.parent / "tinygrad" / "runtime" / "autogen" / "kfd.py").read_text()
  # pattern = r'# (AMDKFD_IOC_[A-Z0-9_]+)\s=\s_(IOW?R?).*\(( 0x[0-9a-fA-F]+) ,\s+struct\s([A-Za-z0-9_]+)\s+\)'
  # matches = re.findall(pattern, hdrpy, re.MULTILINE)
  hdr = (pathlib.Path(__file__).parent.parent.parent.parent / "extra" / "hip_gpu_driver" / "kfd_ioctl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(AMDKFD_IOC_[A-Z0-9_]+)\s+AMDKFD_(IOW?R?)\((0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)
  return type("KFD_IOCTLS", (object, ), {name: int(nr, 0x10) for name, _, nr, _ in matches}), \
         {int(nr, 0x10): getattr(kfd, "struct_"+sname) for name, idir, nr, sname in matches}
kfd_ioctls, kfd_headers = ioctls_from_header()

class KFDFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.kfd_ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return offset

class DRMFileDesc(VirtFileDesc):
  def __init__(self, fd, driver, gpu):
    super().__init__(fd)
    self.driver, self.gpu = driver, gpu

  def mmap(self, start, sz, prot, flags, fd, offset): return libc.mmap(start, sz, prot, flags|mmap.MAP_ANONYMOUS, -1, 0)

class AMDDriver(VirtDriver):
  def __init__(self, gpus=6):
    super().__init__()

    # NOTE: gpu ids start from one (id 0 is skipped in KFDIface._is_usable_gpu)
    self.tracked_files += [VirtFile('/dev/kfd', functools.partial(KFDFileDesc, driver=self))] + \
      [VirtFile('/sys/devices/virtual/kfd/kfd/topology/nodes', functools.partial(DirFileDesc, child_names=[str(i+1) for i in range(gpus)]))]

    self.gpus = {}
    self.next_fd = (1 << 30)
    self.next_handle = 1
    self.next_event = 1

    self.object_by_handle = {}
    self.doorbells = {}
    self.next_doorbell = collections.defaultdict(int)
    self.mmu_event_ids = []

    for i in range(gpus): self._prepare_gpu(i+1)

  def _alloc_fd(self):
    my_fd = self.next_fd
    self.next_fd = self.next_fd + 1
    return my_fd

  def _alloc_handle(self):
    handle = self.next_handle
    self.next_handle += 1
    return handle

  def _alloc_next_event_slot(self):
    ev = self.next_event
    self.next_event += 1
    return ev

  def _alloc_doorbell(self, gpu_id):
    x = ctypes.addressof(from_mv(self.doorbells[gpu_id])) + self.next_doorbell[gpu_id] * 8
    self.next_doorbell[gpu_id] += 1
    return x

  def _prepare_gpu(self, gpu_id):
    self.doorbells[gpu_id] = memoryview(bytearray(0x2000))
    self.gpus[gpu_id] = AMDGPU(gpu_id)
    self.tracked_files += [
      VirtFile('/sys/module/amdgpu', functools.partial(TextFileDesc, text="1")),
      VirtFile('/sys/module/amdgpu/parameters/ppfeaturemask', functools.partial(TextFileDesc, text="0xffff3fff")),
      VirtFile(f'/sys/devices/virtual/kfd/kfd/topology/nodes/{gpu_id}', functools.partial(DirFileDesc, child_names=['gpu_id', 'properties'])),
      VirtFile(f'/sys/devices/virtual/kfd/kfd/topology/nodes/{gpu_id}/gpu_id', functools.partial(TextFileDesc, text=f"{gpu_id}")),
      VirtFile(f'/sys/devices/virtual/kfd/kfd/topology/nodes/{gpu_id}/properties',
        functools.partial(TextFileDesc, text=gpu_props.format(drm_render_minor=gpu_id))),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0',
               functools.partial(DirFileDesc, child_names=[str(am.GC_HWID), str(am.SDMA0_HWID), str(am.NBIF_HWID)])),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.GC_HWID}/0/major', functools.partial(TextFileDesc, text='11')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.GC_HWID}/0/minor', functools.partial(TextFileDesc, text='0')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.GC_HWID}/0/revision', functools.partial(TextFileDesc, text='0')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.GC_HWID}/0/base_addr',
               functools.partial(TextFileDesc, text='0x00001260\n0x0000A000\n0x0001C000\n0x02402C00')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.SDMA0_HWID}/0/major', functools.partial(TextFileDesc, text='6')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.SDMA0_HWID}/0/minor', functools.partial(TextFileDesc, text='0')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.SDMA0_HWID}/0/revision', functools.partial(TextFileDesc, text='0')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.SDMA0_HWID}/0/base_addr',
               functools.partial(TextFileDesc, text='0x00001260\n0x0000A000\n0x0001C000\n0x02402C00')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.NBIF_HWID}/0/major', functools.partial(TextFileDesc, text='4')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.NBIF_HWID}/0/minor', functools.partial(TextFileDesc, text='3')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.NBIF_HWID}/0/revision', functools.partial(TextFileDesc, text='0')),
      VirtFile(f'/sys/class/drm/renderD{gpu_id}/device/ip_discovery/die/0/{am.NBIF_HWID}/0/base_addr',
               functools.partial(TextFileDesc, text='0x00000000\n0x00000014\n0x00000D20\n0x00010400\n0x0241B000\n0x04040000')),
      VirtFile(f'/dev/dri/renderD{gpu_id}', functools.partial(DRMFileDesc, driver=self, gpu=f"{self.gpus[gpu_id]}")),
    ]

  def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())

  def kfd_ioctl(self, req, argp):
    nr = req & 0xFF
    struct = kfd_headers[nr].from_address(argp)

    if nr == kfd_ioctls.AMDKFD_IOC_ACQUIRE_VM: pass
    elif nr == kfd_ioctls.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
      if struct.gpu_id not in self.gpus: return -1
      struct.handle = self._alloc_handle()
      self.object_by_handle[struct.handle] = copy.deepcopy(struct) # save memory struct to know what mem it is
    elif nr == kfd_ioctls.AMDKFD_IOC_FREE_MEMORY_OF_GPU:
      self.object_by_handle.pop(struct.handle)
    elif nr == kfd_ioctls.AMDKFD_IOC_MAP_MEMORY_TO_GPU:
      dev_ids = (ctypes.c_int32 * struct.n_devices).from_address(struct.device_ids_array_ptr)
      for i in range(struct.n_devices):
        gpu = self.gpus[dev_ids[i]]
        mem_obj = self.object_by_handle[struct.handle]
        gpu.map_range(mem_obj.va_addr, mem_obj.size)
        struct.n_success = i + 1
    elif nr == kfd_ioctls.AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
      dev_ids = (ctypes.c_int32 * struct.n_devices).from_address(struct.device_ids_array_ptr)
      for i in range(struct.n_devices):
        gpu = self.gpus[dev_ids[i]]
        mem_obj = self.object_by_handle[struct.handle]
        gpu.unmap_range(mem_obj.va_addr, mem_obj.size)
        struct.n_success = i + 1
    elif nr == kfd_ioctls.AMDKFD_IOC_CREATE_EVENT:
      struct.event_slot_index = self._alloc_next_event_slot()
      struct.event_id = struct.event_slot_index

      if struct.event_type == kfd.KFD_IOC_EVENT_MEMORY: self.mmu_event_ids.append(struct.event_id)
    elif nr == kfd_ioctls.AMDKFD_IOC_CREATE_QUEUE:
      gpu = self.gpus[struct.gpu_id]
      if struct.queue_type == kfd.KFD_IOC_QUEUE_TYPE_SDMA:
        gpu.add_sdma_queue(struct.ring_base_address, struct.ring_size, struct.read_pointer_address, struct.write_pointer_address)
      elif struct.queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE:
        gpu.add_pm4_queue(struct.ring_base_address, struct.ring_size, struct.read_pointer_address, struct.write_pointer_address)
      else: raise RuntimeError("Unsuported, queue")

      # Track writes to doorbell, calling callback
      struct.doorbell_offset = self._alloc_doorbell(struct.gpu_id)
      self.track_address(struct.doorbell_offset, struct.doorbell_offset + 8, lambda mv,off: None, lambda mv, off: self._emulate_execute())
    elif nr == kfd_ioctls.AMDKFD_IOC_WAIT_EVENTS:
      evs = (kfd.struct_kfd_event_data * struct.num_events).from_address(struct.events_ptr)
      for ev in evs:
        if ev.event_id in self.mmu_event_ids and "MOCKGPU_EMU_FAULTADDR" in os.environ:
          ev.memory_exception_data.gpu_id = 1
          ev.memory_exception_data.va = int(os.environ["MOCKGPU_EMU_FAULTADDR"], 16)
          ev.memory_exception_data.failure.NotPresent = 1
    else:
      name = "unknown"
      for k,v in kfd_ioctls.__dict__.items():
        if nr == v: name = k
      assert False, f"unknown kfd ioctl, {nr} {name}"
      exit(1)
    return 0

  def _emulate_execute(self):
    any_progress = True
    while any_progress:
      any_progress = False
      for gpu in self.gpus.values():
        for q in gpu.queues:
          if q.executing: any_progress |= q.execute() > 0
