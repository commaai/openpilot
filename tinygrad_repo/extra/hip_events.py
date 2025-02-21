import ctypes
import tinygrad.runtime.autogen.hip as hip
from tinygrad.runtime.ops_hip import check
from tinygrad.helpers import init_c_var

if __name__ == "__main__":
  check(hip.hipSetDevice(0))
  evt = init_c_var(hip.hipEvent_t(), lambda x: check(hip.hipEventCreate(ctypes.byref(x))))
  check(hip.hipSetDevice(1))
  check(hip.hipStreamWaitEvent(None, evt, 0))
  check(hip.hipSetDevice(0))
  check(hip.hipEventRecord(evt, None))