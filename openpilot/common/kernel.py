import os
from functools import cache


@cache
def upstream_kernel() -> bool:
  """Whether this comma device runs the upstream kernel rather than the downstream 4.9 one. It's decided by the running
  kernel's release, which every process sees the same from the start of boot, unlike device nodes, which appear as their
  drivers probe. Always False off AGNOS.

  C++ makes the same check: HardwareComma::upstream_kernel() in hardware.h, and msgq's VisionIPC picks its buffer
  backend by it (visionbuf_backend_name() is "dma_heap" exactly on the upstream kernel). tests/test_kernel.py checks
  that msgq agrees. The AGNOS updater imports this module from a self-contained zipapp, so it can't use msgq's"""
  return os.path.isfile("/AGNOS") and int(os.uname().release.split(".")[0]) > 4
