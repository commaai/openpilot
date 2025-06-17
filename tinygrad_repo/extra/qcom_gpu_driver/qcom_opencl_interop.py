import ctypes, array
from hexdump import hexdump
from tinygrad.runtime.ops_gpu import GPUDevice
from tinygrad.helpers import getenv, to_mv, mv_address
from tinygrad.dtype import dtypes
from tinygrad import Tensor, TinyJit
from tinygrad.runtime.autogen import opencl as cl
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

# create raw opencl buffer.
gdev = GPUDevice()
cl_buf = cl.clCreateBuffer(gdev.context, cl.CL_MEM_READ_WRITE, 0x100, None, status := ctypes.c_int32())
assert status.value == 0

# fill it with something for fun
data = memoryview(array.array('I', [i for i in range(64)]))
cl.clEnqueueWriteBuffer(gdev.queue, cl_buf, False, 0, 0x100, mv_address(data), 0, None, None)
cl.clFinish(gdev.queue) # wait writes to complete

# get raw gpu pointer from opencl buffer.

## get buf desc
hexdump(to_mv(ctypes.addressof(cl_buf), 0x40))
cl_buf_desc_ptr = to_mv(ctypes.addressof(cl_buf), 8).cast('Q')[0]

## get buf device ptr
hexdump(to_mv(cl_buf_desc_ptr, 0x100))
rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20] # offset 0xA0 is a raw gpu pointer.

# create QCOM tensor with the externally managed buffer
x = Tensor.from_blob(rawbuf_ptr, (8, 8), dtype=dtypes.int, device='QCOM')
y = (x + 1).numpy()
print(y)

# all calculations are done, save to free the object
cl.clReleaseMemObject(cl_buf)

# all together with jit
@TinyJit
def calc(x): return x + 2

for i in range(4):
  cl_buf = cl.clCreateBuffer(gdev.context, cl.CL_MEM_READ_WRITE, 2*2*4, None, status := ctypes.c_int32())
  assert status.value == 0
  data = memoryview(array.array('I', [x+i for x in range(2*2)]))
  cl.clEnqueueWriteBuffer(gdev.queue, cl_buf, False, 0, 2*2*4, mv_address(data), 0, None, None)
  cl.clFinish(gdev.queue) # wait writes to complete

  cl_buf_desc_ptr = to_mv(ctypes.addressof(cl_buf), 8).cast('Q')[0]
  rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20]

  y = calc(x = Tensor.from_blob(rawbuf_ptr, (2, 2), dtype=dtypes.int, device='QCOM')).numpy()
  print(f'jit {i}\n', y)

  # all calculations are done, save to free the object
  cl.clReleaseMemObject(cl_buf)

# now images!

h, w = 128, 128
cl_img = cl.clCreateImage2D(gdev.context, cl.CL_MEM_READ_WRITE, cl.cl_image_format(cl.CL_RGBA, cl.CL_FLOAT), w, h, 0, None, status := ctypes.c_int32())
assert status.value == 0

# fill it with something for fun
data = memoryview(array.array('f', [i for i in range(h*w*4)]))
cl.clEnqueueWriteImage(gdev.queue, cl_img, False, (ctypes.c_size_t * 3)(0,0,0), (ctypes.c_size_t * 3)(w,h,1), 0, 0, mv_address(data), 0, None, None)
cl.clFinish(gdev.queue) # wait writes to complete

# get raw gpu pointer from opencl buffer.

## get buf desc
hexdump(to_mv(ctypes.addressof(cl_img), 0x40))
cl_buf_desc_ptr = to_mv(ctypes.addressof(cl_img), 8).cast('Q')[0]

## get buf device ptr
hexdump(to_mv(cl_buf_desc_ptr, 0x100))
rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20] # offset 0xA0 is a raw gpu pointer.

# create QCOM tensor with the externally managed buffer
# dtypes.imageh = cl.cl_image_format(cl.CL_RGBA, cl.CL_HALF_FLOAT)
# dtypes.imagef = cl.cl_image_format(cl.CL_RGBA, cl.CL_FLOAT)
x = Tensor.from_blob(rawbuf_ptr, (h*w*4,), dtype=dtypes.imagef((h,w)), device='QCOM')
y = (x + 1).numpy()
print(y)

# all calculations are done, save to free the object
cl.clReleaseMemObject(cl_img)
