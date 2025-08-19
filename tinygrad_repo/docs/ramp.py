#!/usr/bin/env python3

# this file is a "ramp" for people new to tinygrad to think about how to approach it
# it is runnable and editable.
# whenever you see stuff like DEBUG=2 or CPU=1 discussed, these are environment variables
# in a unix shell like bash `DEBUG=2 CPU=1 python docs/ramp.py`

# this pip installs tinygrad master for the system
# the -e allows you to edit the tinygrad folder and update system tinygrad
# tinygrad is pure Python, so you are encouraged to do this
# git pull in the tinygrad directory will also get you the latest
"""
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
"""

# %% ********
print("******* PART 1 *******")

# we start with a Device.
# a Device is where Tensors are stored and compute is run
# tinygrad autodetects the best device on your system and makes it the DEFAULT
from tinygrad import Device
print(Device.DEFAULT)  # on Mac, you can see this prints METAL

# now, lets create a Tensor
from tinygrad import Tensor, dtypes
t = Tensor([1,2,3,4])

# you can see this Tensor is on the DEFAULT device with int dtype and shape (4,)
assert t.device == Device.DEFAULT
assert t.dtype == dtypes.int
assert t.shape == (4,)

# unlike in torch, if we print it, it doesn't print the contents
# this is because tinygrad is lazy
# this Tensor has not been computed yet
print(t)
# <Tensor <UOp METAL (4,) int (<Ops.COPY: 7>, None)> on METAL with grad None>

# the ".uop" property on Tensor contains the specification of how to compute it
print(t.uop)
"""
UOp(Ops.COPY, dtypes.int, arg=None, src=(
  UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=0, src=()),
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),
  UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),))
"""
# as you can see, it's specifying a copy from PYTHON device
# which is where the [1,2,3,4] array lives

# UOps are the specification language in tinygrad
# they are immutable and form a DAG
# they have a "Ops", a "dtype", a tuple of srcs (parents), and an arg

t.realize()
# if we want to "realize" a tensor, we can with the "realize" method
# now when we look at the uop, it's changed
print(t.uop)
"""
UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
  UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),
  UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),))
"""
# the copy was actually run, and now the "uop" of the Tensor is just a BUFFER
# if you run this script with DEBUG=2 in the environment, you can see the copy happen
# *** METAL      1 copy       16,   METAL <- PYTHON ...

# now let's do some compute
# we look at the uop to see the specification of the compute
t_times_2 = t * 2
print(t_times_2.uop)
"""
UOp(Ops.MUL, dtypes.int, arg=None, src=(
  UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),
    x2:=UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),
  UOp(Ops.EXPAND, dtypes.int, arg=(4,), src=(
    UOp(Ops.RESHAPE, dtypes.int, arg=(1,), src=(
      UOp(Ops.CONST, dtypes.int, arg=2, src=(
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
           x2,)),)),)),)),))
"""
# the BUFFER from above is being multiplied by a CONST 2
# it's RESHAPEd and EXPANDed to broadcast the CONST to the BUFFER

# we can check the result with
assert t_times_2.tolist() == [2, 4, 6, 8]

# UOps are both immutable and globally unique
# if i multiply the Tensor by 4 twice, these result Tensors will have the same uop specification
t_times_4_try_1 = t * 4
t_times_4_try_2 = t * 4
assert t_times_4_try_1.uop is t_times_4_try_2.uop
# the specification isn't just the same, it's the exact same Python object
assert t_times_4_try_1 is not t_times_4_try_2
# the Tensor is a different Python object

# if we realize `t_times_4_try_1` ...
t_times_4_try_1.realize()
print(t_times_4_try_2.uop)
"""
UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
  UOp(Ops.UNIQUE, dtypes.void, arg=4, src=()),
  UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),))
"""
# ... `t_times_4_try_2` also becomes the same BUFFER
assert t_times_4_try_1.uop is t_times_4_try_2.uop
# so this print doesn't require any computation, just a copy back to the CPU so we can print it
print("** only the copy start")
print(t_times_4_try_2.tolist())  # [4, 8, 12, 16]
print("** only the copy end")
# you can confirm this with DEBUG=2, seeing what's printed in between the "**" prints

# tinygrad has an auto differentiation engine that operates according to these same principles
# the derivative of "log(x)" is "1/x", and you can see this on line 20 of gradient.py
t_float = Tensor([3.0])
t_log = t_float.log()
t_log_grad, = t_log.sum().gradient(t_float)
# due to how log is implemented, this gradient contains a lot of UOps
print(t_log_grad.uop)
# ...not shown here...
# but if you run with DEBUG=4 (CPU=1 used here for simpler code), you can see the generated code
"""
void E_(float* restrict data0, float* restrict data1) {
  float val0 = *(data1+0);
  *(data0+0) = (1/val0);
}
"""
# the derivative is close to 1/3
assert (t_log_grad.item() - 1/3) < 1e-6

# %% ********
print("******* PART 2 *******")

# we redefine the same t here so this cell can run on it's own
from tinygrad import Tensor
t = Tensor([1,2,3,4])

# what's above gives you enough of an understanding to go use tinygrad as a library
# however, a lot of the beauty of tinygrad is in how easy it is to interact with the internals
# NOTE: the APIs here are subject to change

t_plus_3_plus_4 = t + 3 + 4
print(t_plus_3_plus_4.uop)
"""
UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.ADD, dtypes.int, arg=None, src=(
    UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
      UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),
      x3:=UOp(Ops.DEVICE, dtypes.void, arg='CPU', src=()),)),
    UOp(Ops.EXPAND, dtypes.int, arg=(4,), src=(
      UOp(Ops.RESHAPE, dtypes.int, arg=(1,), src=(
        UOp(Ops.CONST, dtypes.int, arg=3, src=(
          x7:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
             x3,)),)),)),)),)),
  UOp(Ops.EXPAND, dtypes.int, arg=(4,), src=(
    UOp(Ops.RESHAPE, dtypes.int, arg=(1,), src=(
      UOp(Ops.CONST, dtypes.int, arg=4, src=(
         x7,)),)),)),))
"""
# you can see it's adding both 3 and 4

# but by the time we are actually running the code, it's adding 7
# `kernelize` will simplify and group the operations in the graph into kernels
t_plus_3_plus_4.kernelize()
print(t_plus_3_plus_4.uop)
"""
UOp(Ops.ASSIGN, dtypes.int, arg=None, src=(
  x0:=UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=7, src=()),
    x2:=UOp(Ops.DEVICE, dtypes.void, arg='CPU', src=()),)),
  UOp(Ops.KERNEL, dtypes.void, arg=<Kernel 12 SINK(<Ops.STORE: 48>,) (__add__,)>, src=(
     x0,
    UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
      UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),
       x2,)),)),))
"""
# ASSIGN has two srcs, src[0] is the BUFFER that's assigned to, and src[1] is the thing to assign
# src[1] is the GPU Kernel that's going to be run
# we can get the ast of the Kernel as follows
kernel_ast = t_plus_3_plus_4.uop.src[1].arg.ast

# almost everything in tinygrad functions as a rewrite of the UOps
# the codegen rewrites the ast to a simplified form ready for "rendering"
from tinygrad.codegen import full_rewrite_to_sink
rewritten_ast = full_rewrite_to_sink(kernel_ast)
print(rewritten_ast)
"""
UOp(Ops.SINK, dtypes.void, arg=None, src=(
  UOp(Ops.STORE, dtypes.void, arg=None, src=(
    UOp(Ops.INDEX, dtypes.int.ptr(4), arg=None, src=(
      UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(4), arg=0, src=()),
      x3:=UOp(Ops.SPECIAL, dtypes.int, arg=('gidx0', 4), src=()),)),
    UOp(Ops.ADD, dtypes.int, arg=None, src=(
      UOp(Ops.LOAD, dtypes.int, arg=None, src=(
        UOp(Ops.INDEX, dtypes.int.ptr(4), arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(4), arg=1, src=()),
           x3,)),)),
      UOp(Ops.CONST, dtypes.int, arg=7, src=()),)),)),))
"""
# you can see at this point we are adding 7, not 3 and 4

# with DEBUG=4, we can see the code.
# since optimizations are on, it UPCASTed the operation, explicitly writing out all 4 +7s
t_plus_3_plus_4.realize()
"""
void E_4n2(int* restrict data0, int* restrict data1) {
  int val0 = *(data1+0);
  int val1 = *(data1+1);
  int val2 = *(data1+2);
  int val3 = *(data1+3);
  *(data0+0) = (val0+7);
  *(data0+1) = (val1+7);
  *(data0+2) = (val2+7);
  *(data0+3) = (val3+7);
}
"""
# the function name E_4n2 is "E" for elementwise op (as opposed to "r" for reduce op)
# "4" for the size, and "n2" for name deduping (it's the 3rd function with the same E and 4 in this session)
# when you print the name with DEBUG=2, you'll see the 4 is yellow, meaning that it's upcasted
# if you run with NOOPT=1 ...
"""
void E_4n2(int* restrict data0, int* restrict data1) {
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int val0 = *(data1+ridx0);
    *(data0+ridx0) = (val0+7);
  }
}
"""
# ... you get this unoptimized code with a loop and the 4 is blue (for global). the color code is in kernel.py

# %% ********
print("******* PART 3 *******")

# now, we go even lower and understand UOps better and how the graph rewrite engine works.
# it's much simpler than what's in LLVM or MLIR

from tinygrad import dtypes
from tinygrad.uop.ops import UOp, Ops

# first, we'll construct some const UOps
a = UOp(Ops.CONST, dtypes.int, arg=2)
b = UOp(Ops.CONST, dtypes.int, arg=2)

# if you have been paying attention, you should know these are the same Python object
assert a is b

# UOps support normal Python math operations, so a_plus_b expresses the spec for 2 + 2
a_plus_b = a + b
print(a_plus_b)
"""
UOp(Ops.ADD, dtypes.int, arg=None, src=(
  x0:=UOp(Ops.CONST, dtypes.int, arg=2, src=()),
   x0,))
"""

# we could actually render this 2+2 into a language like c and run it
# or, we can use tinygrad's graph rewrite engine to "constant fold"

from tinygrad.uop.ops import graph_rewrite, UPat, PatternMatcher

# a `PatternMatcher` is a list of tuples. for each element in the list:
# [0] is the pattern to match, and [1] is the function to run.
# this function can return either a UOp to replace the pattern with, or None to not replace
simple_pm = PatternMatcher([
  (UPat(Ops.ADD, src=(UPat(Ops.CONST, name="c1"), UPat(Ops.CONST, name="c2"))),
   lambda c1,c2: UOp(Ops.CONST, dtype=c1.dtype, arg=c1.arg+c2.arg)),
])
# this pattern matches the addition of two CONST and rewrites it into a single CONST UOp

# to actually apply the pattern to a_plus_b, we use graph_rewrite
a_plus_b_simplified = graph_rewrite(a_plus_b, simple_pm)
print(a_plus_b_simplified)
"""
UOp(Ops.CONST, dtypes.int, arg=4, src=())
"""
# 2+2 is in fact, 4

# we can also use syntactic sugar to write the pattern nicer
simpler_pm = PatternMatcher([
  (UPat.cvar("c1")+UPat.cvar("c2"), lambda c1,c2: c1.const_like(c1.arg+c2.arg))
])
assert graph_rewrite(a_plus_b, simple_pm) is graph_rewrite(a_plus_b, simpler_pm)
# note again the use of is, UOps are immutable and globally unique

# %% ********

# that brings you to an understanding of the most core concepts in tinygrad
# you can run this with VIZ=1 to use the web based graph rewrite explorer
# hopefully now you understand it. the nodes in the graph are just UOps
