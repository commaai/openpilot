from __future__ import annotations
import ctypes
from tinygrad.helpers import DEBUG, getenv
from extra.nv_pma.cupti import cupti

def stall_reason_name(reason: int) -> str:
  name = cupti.CUpti_ActivityPCSamplingStallReason.get(reason)
  return name.replace("CUPTI_ACTIVITY_PC_SAMPLING_STALL_", "").lower() if name else str(reason)

class CUPTIProfiler:
  def __init__(self):
    self.initialized = False
    self.pc_sampling_enabled = False
    self.buffers: list[ctypes.Array] = []
    self.kernel_stalls: dict[int, dict[int, int]] = {}
    self.raw_buffers: list[bytes] = []
    self.pc_samples: list[dict] = []

  def _check_cupti(self, status, soft=False):
    if status != cupti.CUPTI_SUCCESS:
      if soft: return False
      raise RuntimeError(f"CUPTI Error {status}")
    return True

  def init(self, ctx, device_id: int = 0, profile_level: int = 2):
    if self.initialized: return

    # Initialize profiler API
    init_params = cupti.CUpti_Profiler_Initialize_Params()
    init_params.structSize = 16
    cupti.cuptiProfilerInitialize(ctypes.byref(init_params))

    # Register buffer callbacks for Activity API
    self._buf_req_cb = cupti.CUpti_BuffersCallbackRequestFunc(self._buffer_requested)
    self._buf_comp_cb = cupti.CUpti_BuffersCallbackCompleteFunc(self._buffer_completed)
    self._check_cupti(cupti.cuptiActivityRegisterCallbacks(self._buf_req_cb, self._buf_comp_cb))

    # PROFILE=1: kernel timing, PROFILE=2: PC sampling with stall reasons
    if profile_level >= 2:
      # PC sampling for stall analysis (requires elevated privileges)
      if DEBUG >= 1: print("  CUPTI: PC sampling mode (before)")
      pc_status = cupti.cuptiActivityEnable(cupti.CUPTI_ACTIVITY_KIND_PC_SAMPLING)
      if pc_status == cupti.CUPTI_SUCCESS:
        config = cupti.CUpti_ActivityPCSamplingConfig()
        config.size, config.samplingPeriod = 16, cupti.CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN
        cfg_status = cupti.dll.cuptiActivityConfigurePCSampling(ctx, ctypes.byref(config))
        if cfg_status == cupti.CUPTI_SUCCESS:
          if DEBUG >= 1: print("  CUPTI: PC sampling mode (before stall analysis)")
          cupti.cuptiActivityEnable(cupti.CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO)
          self.pc_sampling_enabled = True
          if DEBUG >= 1: print("  CUPTI: PC sampling mode (stall analysis)")
        elif cfg_status == 35:
          if DEBUG >= 1: print("  CUPTI: PC sampling needs: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0'|sudo tee /etc/modprobe.d/nvidia.conf && sudo reboot")
      # Fall back to kernel timing if PC sampling setup failed
      if not self.pc_sampling_enabled:
        self._check_cupti(cupti.cuptiActivityEnable(cupti.CUPTI_ACTIVITY_KIND_KERNEL))
    else:
      # Kernel activity tracing for timing
      self._check_cupti(cupti.cuptiActivityEnable(cupti.CUPTI_ACTIVITY_KIND_KERNEL))

    self.initialized = True

  def _buffer_requested(self, buffer, size, max_num_records):
    buf = (ctypes.c_uint8 * 1024 * 1024)()  # 1MB buffer
    self.buffers.append(buf)
    buffer[0] = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
    size[0] = ctypes.sizeof(buf)
    max_num_records[0] = 0

  def _buffer_completed(self, ctx, stream_id, buffer, size, valid_size):
    if valid_size > 0:
      record = ctypes.POINTER(cupti.CUpti_Activity)()
      while cupti.cuptiActivityGetNextRecord(buffer, valid_size, ctypes.byref(record)) == cupti.CUPTI_SUCCESS:
        kind = record.contents.kind
        if kind == cupti.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
          kernel = ctypes.cast(record, ctypes.POINTER(cupti.CUpti_ActivityKernel9)).contents
          name = ctypes.string_at(kernel.name).decode() if kernel.name else "unknown"
          duration_us = (kernel.end - kernel.start) / 1000.0
          grid, block = (kernel.gridX, kernel.gridY, kernel.gridZ), (kernel.blockX, kernel.blockY, kernel.blockZ)
          print(f"  CUPTI: {name[:40]:40s} | {duration_us:10.2f} us | grid={grid} block={block} | regs={kernel.registersPerThread:3d} smem={kernel.staticSharedMemory + kernel.dynamicSharedMemory:6d}B")
        elif kind == cupti.CUPTI_ACTIVITY_KIND_PC_SAMPLING:
          pc = ctypes.cast(record, ctypes.POINTER(cupti.CUpti_ActivityPCSampling3)).contents
          cid = pc.correlationId
          if cid not in self.kernel_stalls: self.kernel_stalls[cid] = {}
          self.kernel_stalls[cid][pc.stallReason] = self.kernel_stalls[cid].get(pc.stallReason, 0) + pc.samples
          self.pc_samples.append({
            'correlationId': pc.correlationId, 'pcOffset': pc.pcOffset, 'stallReason': pc.stallReason,
            'samples': pc.samples, 'latencySamples': pc.latencySamples, 'functionId': pc.functionId, 'sourceLocatorId': pc.sourceLocatorId
          })
          if DEBUG >= 3:
            print(f"    PC {pc.pcOffset:#x} stall={stall_reason_name(pc.stallReason)} samples={pc.samples} latency={pc.latencySamples} func={pc.functionId} src={pc.sourceLocatorId}")
        elif kind == cupti.CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO:
          info = ctypes.cast(record, ctypes.POINTER(cupti.CUpti_ActivityPCSamplingRecordInfo)).contents
          cid = info.correlationId
          if cid in self.kernel_stalls:
            stalls = self.kernel_stalls[cid]
            total = sum(stalls.values())
            if total > 0:
              top = sorted(stalls.items(), key=lambda x: -x[1])[:5]
              stall_str = " ".join(f"{stall_reason_name(r)}:{100*c//total}%" for r,c in top if c > 0)
              print(f"  CUPTI stalls (corr={cid}): {total} samples | {stall_str}")
            del self.kernel_stalls[cid]
        else: print(f"  CUPTI: Unhandled activity kind {kind}")

  def flush(self):
    if not self.initialized: return
    self._check_cupti(cupti.cuptiActivityFlushAll(0))

# Module-level profiler instance
_profiler: CUPTIProfiler | None = None

def get_profiler() -> CUPTIProfiler | None:
  return _profiler

def get_cupti_raw_buffers() -> list[bytes]:
  return _profiler.raw_buffers if _profiler else []

def clear_cupti_raw_buffers():
  if _profiler: _profiler.raw_buffers.clear()

def get_cupti_pc_samples() -> list[dict]:
  return _profiler.pc_samples if _profiler else []

def clear_cupti_pc_samples():
  if _profiler: _profiler.pc_samples.clear()

# Raw PMA buffer access (from ioctl interception)
def get_pma_raw_dumps() -> list[bytes]:
  try:
    from extra.nv_gpu_driver.nv_ioctl import get_pma_raw_dumps as _get
    return _get()
  except ImportError: return []

def clear_pma_raw_dumps():
  try:
    from extra.nv_gpu_driver.nv_ioctl import clear_pma_raw_dumps as _clear
    _clear()
  except ImportError: pass

def enable(profile_level:int=2):
  global _profiler
  if _profiler is not None: return

  _profiler = CUPTIProfiler()

  # Patch CUDADevice to initialize CUPTI profiler
  from tinygrad.runtime.ops_cuda import CUDADevice
  _orig_init = CUDADevice.__init__
  _orig_sync = CUDADevice.synchronize

  def _patched_init(self, device: str):
    _orig_init(self, device)
    device_id = int(device.split(":")[1]) if ":" in device else 0
    _profiler.init(self.context, device_id, profile_level)

  def _patched_sync(self):
    _orig_sync(self)
    if _profiler: _profiler.flush()

  CUDADevice.__init__ = _patched_init
  CUDADevice.synchronize = _patched_sync

def enable_auto():
  if (profile_level:=getenv("PROFILE", 0)) > 0: enable(profile_level)
