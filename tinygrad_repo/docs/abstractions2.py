# tinygrad is a tensor library, and as a tensor library it has multiple parts
# 1. a "runtime". this allows buffer management, compilation, and running programs
# 2. a "Device" that uses the runtime but specifies compute in an abstract way for all
# 3. a "UOp" that fuses the compute into kernels, using memory only when needed
# 4. a "Tensor" that provides an easy to use frontend with autograd ".backward()"


print("******** first, the runtime ***********")

from tinygrad.runtime.ops_cpu import ClangJITCompiler, MallocAllocator, CPUProgram

# allocate some buffers
out = MallocAllocator.alloc(4)
a = MallocAllocator.alloc(4)
b = MallocAllocator.alloc(4)

# load in some values (little endian)
MallocAllocator._copyin(a, memoryview(bytearray([2,0,0,0])))
MallocAllocator._copyin(b, memoryview(bytearray([3,0,0,0])))

# compile a program to a binary
lib = ClangJITCompiler().compile("void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }")

# create a runtime for the program
fxn = CPUProgram("add", lib)

# run the program
fxn(out, a, b)

# check the data out
print(val := MallocAllocator._as_buffer(out).cast("I").tolist()[0])
assert val == 5


print("******** second, the Device ***********")

DEVICE = "CPU"   # NOTE: you can change this!

import struct
from tinygrad.dtype import dtypes
from tinygrad.device import Buffer, Device
from tinygrad.uop.ops import UOp, Ops
from tinygrad.shape.shapetracker import ShapeTracker

# allocate some buffers + load in values
out = Buffer(DEVICE, 1, dtypes.int32).allocate()
a = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 2))))
b = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 3))))
# NOTE: a._buf is the same as the return from MallocAllocator.alloc

# describe the computation
buf_1 = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 1)
buf_2 = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 2)
ld_1 = UOp(Ops.LOAD, dtypes.int32, (buf_1.view(ShapeTracker.from_shape((1,))),))
ld_2 = UOp(Ops.LOAD, dtypes.int32, (buf_2.view(ShapeTracker.from_shape((1,))),))
alu = ld_1 + ld_2
output_buf = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), (), 0)
st_0 = UOp(Ops.STORE, dtypes.void, (output_buf.view(ShapeTracker.from_shape((1,))), alu))
s = UOp(Ops.SINK, dtypes.void, (st_0,))

# convert the computation to a "linearized" format (print the format)
from tinygrad.engine.realize import get_kernel, CompiledRunner
kernel = get_kernel(Device[DEVICE].renderer, s).linearize()

# compile a program (and print the source)
fxn = CompiledRunner(kernel.to_program())
print(fxn.p.src)
# NOTE: fxn.clprg is the CPUProgram

# run the program
fxn.exec([out, a, b])

# check the data out
assert out.as_buffer().cast('I')[0] == 5


print("******** third, the UOp ***********")

from tinygrad.engine.realize import run_schedule
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.grouper import get_kernelize_map

# allocate some values + load in values
a = UOp.new_buffer(DEVICE, 1, dtypes.int32)
b = UOp.new_buffer(DEVICE, 1, dtypes.int32)
a.buffer.allocate().copyin(memoryview(bytearray(struct.pack("I", 2))))
b.buffer.allocate().copyin(memoryview(bytearray(struct.pack("I", 3))))

# describe the computation
out = a + b
s = UOp(Ops.SINK, dtypes.void, (out,))

# group the computation into kernels
becomes_map = get_kernelize_map(s)

# the compute maps to an assign
assign = becomes_map[a+b]

# the first source is the output buffer (data)
assert assign.src[0].op is Ops.BUFFER
# the second source is the kernel (compute)
assert assign.src[1].op is Ops.KERNEL

# schedule the kernel graph in a linear list
s = UOp(Ops.SINK, dtypes.void, (assign,))
sched, _ = create_schedule_with_vars(s)
assert len(sched) == 1

# DEBUGGING: print the compute ast
print(sched[-1].ast)
# NOTE: sched[-1].ast is the same as st_0 above

# the output will be stored in a new buffer
out = assign.buf_uop
assert out.op is Ops.BUFFER and not out.buffer.is_allocated()
print(out)

# run that schedule
run_schedule(sched)

# check the data out
assert out.is_realized and out.buffer.as_buffer().cast('I')[0] == 5


print("******** fourth, the Tensor ***********")

from tinygrad import Tensor

a = Tensor([2], dtype=dtypes.int32, device=DEVICE)
b = Tensor([3], dtype=dtypes.int32, device=DEVICE)
out = a + b

# check the data out
print(val:=out.item())
assert val == 5
