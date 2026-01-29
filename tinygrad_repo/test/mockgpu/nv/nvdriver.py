import ctypes, mmap, collections, functools, os
from tinygrad.runtime.autogen import nv_570 as nv_gpu
from typing import cast, Any
from tinygrad.helpers import to_mv
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile
from test.mockgpu.nv.nvgpu import NVGPU

MAP_FIXED = 0x10
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

NVSubDevice = collections.namedtuple('NVSubDevice', ['device'])
NVUserMode = collections.namedtuple('NVUserMode', ['subdevice'])
NVVASpace = collections.namedtuple('NVVASpace', ['device'])
NVAllocation = collections.namedtuple('NVAllocation', ['device', 'size', 'is_signal'])
NVChannelGroup = collections.namedtuple('NVChannelGroup', ['device'])
NVContextShare = collections.namedtuple('NVContextShare', ['channel_group'])
NVGPFIFO = collections.namedtuple('NVGPFIFO', ['device', 'token'])
NVProfiler = collections.namedtuple('NVProfiler', ['subdevice'])

class NVCtlFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.ctl_ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return libc.mmap(start, sz, prot, flags|mmap.MAP_ANONYMOUS, -1, 0)

class NVUVMFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.uvm_ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return libc.mmap(start, sz, prot, flags|mmap.MAP_ANONYMOUS, -1, 0)

class NVDevFileDesc(VirtFileDesc):
  def __init__(self, fd, driver, gpu):
    super().__init__(fd)
    self.driver, self.gpu = driver, gpu
    self._mapping_userland = False
    self._mapping_signal = False

  def ioctl(self, fd, request, argp): return self.driver.dev_ioctl(self.gpu, request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset):
    start = libc.mmap(start, sz, prot, flags|mmap.MAP_ANONYMOUS, -1, 0)
    if self._mapping_userland or self._mapping_signal:
      self.driver.track_address(start, start+sz, lambda mv,off: None, lambda mv, off: self.driver._gpu_mmio_write(mv, off, self.gpu))
      self._mapping_signal = False
    return start

class NVDriver(VirtDriver):
  def __init__(self, gpus=6):
    super().__init__()

    self.tracked_files += [VirtFile('/dev/nvidiactl', functools.partial(NVCtlFileDesc, driver=self)),
                           VirtFile('/dev/nvidia-uvm', functools.partial(NVUVMFileDesc, driver=self))]

    self.root_handle = None

    self.gpus = {}
    self.next_fd = (1 << 29)
    self.next_handle = 1

    self.object_by_handle = {}
    self.opened_fds = {}
    self.next_doorbell = collections.defaultdict(int)
    self._executing = False  # re-entrancy guard for _gpu_mmio_write

    for i in range(gpus): self._prepare_gpu(i)

  def _alloc_fd(self):
    my_fd = self.next_fd
    self.next_fd = self.next_fd + 1
    return my_fd

  def _alloc_handle(self):
    handle = self.next_handle
    self.next_handle += 1
    return handle

  def _prepare_gpu(self, gpu_id):
    self.gpus[gpu_id] = NVGPU(gpu_id)
    self.tracked_files += [VirtFile(f'/dev/nvidia{gpu_id}', functools.partial(NVDevFileDesc, driver=self, gpu=self.gpus[gpu_id]))]

  def open(self, name, flags, mode, virtfile):
    cl = virtfile.fdcls(self._alloc_fd())
    self.opened_fds[cl.fd] = cl
    return cl

  def rm_alloc(self, argp):
    struct = nv_gpu.NVOS21_PARAMETERS.from_address(argp)
    params_ptr = cast(int, struct.pAllocParms)
    if struct.hClass == nv_gpu.NV01_ROOT_CLIENT: self.root_handle = struct.hObjectNew = self._alloc_handle()
    elif struct.hClass == nv_gpu.NV01_DEVICE_0:
      params:Any = nv_gpu.NV0080_ALLOC_PARAMETERS.from_address(params_ptr)
      assert params.hClientShare == self.root_handle
      struct.hObjectNew = self._alloc_handle()
      self.object_by_handle[struct.hObjectNew] = self.gpus[params.deviceId]
    elif struct.hClass == nv_gpu.NV20_SUBDEVICE_0:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVGPU)
      struct.hObjectNew = self._alloc_handle()
      self.object_by_handle[struct.hObjectNew] = NVSubDevice(self.object_by_handle[struct.hObjectParent])
    elif struct.hClass == nv_gpu.NV01_MEMORY_VIRTUAL:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVGPU)
      struct.hObjectNew = self._alloc_handle()
    elif struct.hClass == nv_gpu.TURING_USERMODE_A:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVSubDevice)
      struct.hObjectNew = self._alloc_handle()
      self.object_by_handle[struct.hObjectNew] = NVUserMode(self.object_by_handle[struct.hObjectParent])
    elif struct.hClass == nv_gpu.FERMI_VASPACE_A:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVGPU)
      struct.hObjectNew = self._alloc_handle()
      self.object_by_handle[struct.hObjectNew] = NVVASpace(self.object_by_handle[struct.hObjectParent])
    elif struct.hClass == nv_gpu.NV1_MEMORY_SYSTEM or struct.hClass == nv_gpu.NV1_MEMORY_USER:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVGPU)
      params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS.from_address(params_ptr)
      struct.hObjectNew = self._alloc_handle()
      is_signal = struct.hClass == nv_gpu.NV1_MEMORY_SYSTEM  # signal memory uses NV1_MEMORY_SYSTEM (uncached)
      self.object_by_handle[struct.hObjectNew] = NVAllocation(self.object_by_handle[struct.hObjectParent], params.size, is_signal)
    elif struct.hClass == nv_gpu.KEPLER_CHANNEL_GROUP_A:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVGPU)
      struct.hObjectNew = self._alloc_handle()
      self.object_by_handle[struct.hObjectNew] = NVChannelGroup(self.object_by_handle[struct.hObjectParent])
    elif struct.hClass == nv_gpu.FERMI_CONTEXT_SHARE_A:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVChannelGroup)
      struct.hObjectNew = self._alloc_handle()
      self.object_by_handle[struct.hObjectNew] = NVContextShare(self.object_by_handle[struct.hObjectParent])
    elif struct.hClass == nv_gpu.AMPERE_CHANNEL_GPFIFO_A:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVChannelGroup)
      struct.hObjectNew = self._alloc_handle()
      params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS.from_address(params_ptr)
      gpu = self.object_by_handle[struct.hObjectParent].device
      gpfifo_token = gpu.add_gpfifo(params.gpFifoOffset, params.gpFifoEntries)
      self.object_by_handle[struct.hObjectNew] = NVGPFIFO(gpu, gpfifo_token)
    elif struct.hClass == nv_gpu.AMPERE_DMA_COPY_B or struct.hClass == nv_gpu.ADA_COMPUTE_A:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVGPFIFO)
      struct.hObjectNew = self._alloc_handle()
    elif struct.hClass == nv_gpu.GT200_DEBUGGER:
      struct.hObjectNew = self._alloc_handle()
    elif struct.hClass == nv_gpu.MAXWELL_PROFILER_DEVICE:
      assert struct.hObjectParent in self.object_by_handle and isinstance(self.object_by_handle[struct.hObjectParent], NVSubDevice)
      struct.hObjectNew = self._alloc_handle()
      self.object_by_handle[struct.hObjectNew] = NVProfiler(self.object_by_handle[struct.hObjectParent])
    else: raise RuntimeError(f"Unknown {struct.hClass} to rm_alloc")
    return 0

  def rm_control(self, argp):
    struct = nv_gpu.NVOS54_PARAMETERS.from_address(argp)
    params_ptr = cast(int, struct.params)
    if struct.cmd == nv_gpu.NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2:
      params:Any = nv_gpu.NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS.from_address(params_ptr)
      params.deviceInstance = params.gpuId # emulate them to be the same
    elif struct.cmd == nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2 or struct.cmd == nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST:
      if struct.cmd == nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST:
        params = nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS.from_address(params_ptr)
      else:
        params = nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_V2_PARAMS.from_address(params_ptr)

      classes = [50021, 51607, 51648, 50543, 51125, 51125, 51125, 51125, 50529, 36967, 36909, 37105, 33868, 36978, 37095, 37094, 36980, 37014, 49270,
                 41068, 41088, 41280, 50025, 96, 112, 115, 125, 20608, 20640, 20539, 20540, 41089, 41092, 50034, 50810, 50811, 50814, 51056, 51057,
                 51059, 51069, 51071, 51632, 51639, 51639, 51706, 52019, 222, 50287, 50273, 50031, 50017] # from ada102
      params.numClasses = len(classes)
      if struct.cmd == nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST:
        if params.classList and params.numClasses > 0:
          clslist = to_mv(params.classList, params.numClasses * 4).cast('I')
          for i,c in enumerate(classes): clslist[i] = c
        else: params.numClasses = len(classes)
      else:
        for i,c in enumerate(classes): params.classList[i] = c
    elif struct.cmd == nv_gpu.NV2080_CTRL_CMD_GR_GET_INFO:
      info = {nv_gpu.NV2080_CTRL_GR_INFO_INDEX_SM_VERSION: nv_gpu.NV2080_CTRL_GR_INFO_SM_VERSION_3_5,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS: 1,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC: 1,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC: 1,
        nv_gpu.NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM: 1,
      }

      params = nv_gpu.NV2080_CTRL_GR_GET_INFO_PARAMS.from_address(params_ptr)
      reqlist = (nv_gpu.NV2080_CTRL_GR_INFO * params.grInfoListSize).from_address(params.grInfoList)
      for i in range(params.grInfoListSize): reqlist[i].data = info[reqlist[i].index]
    elif struct.cmd == nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO:
      assert struct.hObject in self.object_by_handle and isinstance(self.object_by_handle[struct.hObject], NVSubDevice)
      gpu = self.object_by_handle[struct.hObject].device
      params = nv_gpu.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS.from_address(params_ptr)
      if params.flags != nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY: raise RuntimeError("Unknown format")
      bts = gpu.gpu_uuid(sz=params.length)
      for i in range(params.length): params.data[i] = bts[i]
    elif struct.cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
      assert struct.hObject in self.object_by_handle and isinstance(self.object_by_handle[struct.hObject], NVGPFIFO)
      params = nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS.from_address(params_ptr)
      gpu_fifo = self.object_by_handle[struct.hObject]
      params.workSubmitToken = gpu_fifo.token
    elif struct.cmd == nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE: pass
    elif struct.cmd == nv_gpu.NV2080_CTRL_CMD_PERF_BOOST: pass
    elif struct.cmd == nv_gpu.NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE: pass
    elif struct.cmd == nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES:
      params = nv_gpu.NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS.from_address(params_ptr)
      params.mmuFault.valid = bool("MOCKGPU_EMU_FAULTADDR" in os.environ)
    elif struct.cmd == nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO:
      params = nv_gpu.struct_NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS.from_address(params_ptr)
      params.count = 1
      params.mmuFaultInfoList[0].faultAddress = int(os.environ['MOCKGPU_EMU_FAULTADDR'], base=16)
      params.mmuFaultInfoList[0].faultType = 1
      params.mmuFaultInfoList[0].accessType = 1
    elif struct.cmd == nv_gpu.NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2:
      params = nv_gpu.NV0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS.from_address(params_ptr)
      params.driverVersionBuffer = b"570.00.00\0"
    elif struct.cmd == nv_gpu.NV2080_CTRL_CMD_GR_GET_TPC_MASK:
      params = nv_gpu.NV2080_CTRL_GR_GET_TPC_MASK_PARAMS.from_address(params_ptr)
      params.tpcMask = 0x1  # one TPC
    # Profiler commands - just pass through for mockgpu
    elif struct.cmd in (nv_gpu.NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES, nv_gpu.NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM,
                        nv_gpu.NVB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY, nv_gpu.NVB0CC_CTRL_CMD_RESERVE_PM_AREA_PC_SAMPLER,
                        nv_gpu.NVB0CC_CTRL_CMD_BIND_PM_RESOURCES, nv_gpu.NVB0CC_CTRL_CMD_SET_HS_CREDITS,
                        nv_gpu.NVB0CC_CTRL_CMD_EXEC_REG_OPS, nv_gpu.NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT): pass
    else: raise RuntimeError(f"Unknown {struct.cmd} to rm_control")
    return 0

  def ctl_ioctl(self, req, argp):
    nr = req & 0xff
    if nr == nv_gpu.NV_ESC_RM_ALLOC: return self.rm_alloc(argp)
    elif nr == nv_gpu.NV_ESC_RM_CONTROL: return self.rm_control(argp)
    elif nr == nv_gpu.NV_ESC_RM_MAP_MEMORY:
      st:Any = nv_gpu.nv_ioctl_nvos33_parameters_with_fd.from_address(argp)
      obj = self.object_by_handle.get(st.params.hMemory)
      file = self.opened_fds.get(st.fd)
      if isinstance(obj, NVUserMode) and isinstance(file, NVDevFileDesc):
        file._mapping_userland = True
      elif isinstance(obj, NVAllocation) and obj.is_signal and isinstance(file, NVDevFileDesc):
        file._mapping_signal = True
    elif nr == nv_gpu.NV_ESC_RM_FREE:
      st = nv_gpu.NVOS00_PARAMETERS.from_address(argp)
      self.object_by_handle.pop(st.hObjectOld)
    elif nr == nv_gpu.NV_ESC_RM_MAP_MEMORY_DMA:
      pass # mappings are same as uvm
    elif nr == nv_gpu.NV_ESC_CARD_INFO:
      for i,gpu in enumerate(self.gpus.values()):
        st = nv_gpu.nv_ioctl_card_info_t.from_address(argp + i * ctypes.sizeof(nv_gpu.nv_ioctl_card_info_t))
        st.gpu_id = gpu.gpuid
        st.pci_info.device_id = 0x2684
        st.valid = True
    else: raise RuntimeError(f"Unknown {nr} to nvidiactl")
    return 0
  def uvm_ioctl(self, nr, argp):
    if nr == nv_gpu.UVM_INITIALIZE: pass
    elif nr == nv_gpu.UVM_MM_INITIALIZE: pass
    elif nr == nv_gpu.UVM_REGISTER_GPU:
      st:Any = nv_gpu.UVM_REGISTER_GPU_PARAMS.from_address(argp)
      assert any(all(st.gpu_uuid.uuid[i] == gpu.gpu_uuid()[i] for i in range(16)) for gpu in self.gpus.values())
    elif nr == nv_gpu.UVM_REGISTER_GPU_VASPACE: pass
    elif nr == nv_gpu.UVM_ENABLE_PEER_ACCESS: pass # uvm and shared spaced are setup already, no emulation for now
    elif nr == nv_gpu.UVM_CREATE_EXTERNAL_RANGE:
      st = nv_gpu.UVM_CREATE_EXTERNAL_RANGE_PARAMS.from_address(argp)
      libc.mmap(st.base, st.length, mmap.PROT_READ|mmap.PROT_WRITE, MAP_FIXED|mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
    elif nr == nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION:
      st = nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS.from_address(argp)
      for gpu_attr_id in range(st.gpuAttributesCount):
        gpu = None
        for _gpu in self.gpus.values():
          if all(st.perGpuAttributes[gpu_attr_id].gpuUuid.uuid[i] == _gpu.gpu_uuid()[i] for i in range(16)):
            gpu = _gpu
            break
        if gpu is None: return -1
        gpu.map_range(st.base, st.length)
    elif nr == nv_gpu.UVM_REGISTER_CHANNEL: pass
    elif nr == nv_gpu.UVM_FREE:
      st = nv_gpu.UVM_FREE_PARAMS.from_address(argp)
      libc.munmap(st.base, st.length)
    else: raise RuntimeError(f"Unknown {nr} to nvidia-uvm")
    return 0

  def dev_ioctl(self, dev, req, argp):
    nr = req & 0xff
    # Handle NV_ESC_RM_ALLOC_MEMORY for host/signal memory
    if nr == nv_gpu.NV_ESC_RM_ALLOC_MEMORY:
      st:Any = nv_gpu.nv_ioctl_nvos02_parameters_with_fd.from_address(argp)
      # Track host memory (signal memory) - progress queues when written to
      if st.params.hClass == nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR:
        self.track_address(st.params.pMemory, st.params.pMemory + st.params.limit + 1,
                           lambda mv,off: None, lambda mv, off: self._gpu_mmio_write(mv, off, None))
    return 0
  def _gpu_mmio_write(self, mv, off, gpu):
    if self._executing: return  # prevent re-entrancy
    self._executing = True
    try:
      any_progress = True
      while any_progress:
        any_progress = False
        for gpu in self.gpus.values():
          for q in gpu.queues:
            if q.ctrl.GPGet != q.ctrl.GPPut:
              any_progress |= q.execute()
    finally:
      self._executing = False
