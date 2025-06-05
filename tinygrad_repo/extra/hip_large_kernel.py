from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes, _to_np_dtype

dev = Device.default
mbin = dev.compiler.compile("""
typedef long unsigned int size_t;
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 1))) write_ones(signed char* data0) {
  int gidx0 = __ockl_get_group_id(0); /* 16 */
  int gidx1 = __ockl_get_group_id(1); /* 1026048 */
  *(data0+(gidx0+gidx1*1)) = 1;
}
""")
dev.compiler.disassemble(mbin)
buf0 = Buffer(Device.DEFAULT, 1*65537, dtypes.uint8).ensure_allocated()

prg = dev.runtime("write_ones", mbin)
prg(buf0._buf, global_size=(1,65537,1), local_size=(1,1,1), wait=True)

import numpy as np
def to_np(buf): return np.frombuffer(buf.as_buffer().cast(buf.dtype.base.fmt), dtype=_to_np_dtype(buf.dtype.base))

big = to_np(buf0)
print(big)
print((big-1).nonzero())
