#!/usr/bin/env python3
import numpy as np
import time
import sys
np.set_printoptions(linewidth=160)
np.set_printoptions(linewidth=1000, threshold=10000000000, suppress=False)
from tinygrad.runtime.ops_llvm import LLVMDevice, LLVMProgram, LLVMCompiler
from llvmlite import ir  # type: ignore
from tinygrad.helpers import flat_mv
from tinygrad.device import MallocAllocator

# https://github.com/corsix/amx/blob/main/Instructions.md
# 12 lines for AMX support
from functools import partialmethod
class AMX:
  @staticmethod
  def nop_op_imm5(op, imm5, builder): builder.asm(ir.FunctionType(ir.VoidType(), []), f".word (0x201000 + ({op} << 5) + {imm5}); amx op {op} imm {imm5}", "", tuple(), True)
  @staticmethod
  def op_gpr(op, builder, gpr): builder.asm(ir.FunctionType(ir.VoidType(), [ir.IntType(64)]), f".word (0x201000 + ({op} << 5) + 0$0 - ((0$0 >> 4) * 6)); amx op {op} reg $0", "r", (gpr,), True)
  set, clr = partialmethod(nop_op_imm5, 17, 0), partialmethod(nop_op_imm5, 17, 1)
  ldx, ldy, stx, sty = partialmethod(op_gpr, 0), partialmethod(op_gpr, 1), partialmethod(op_gpr, 2), partialmethod(op_gpr, 3)
  ldz, stz, ldzi, stzi = partialmethod(op_gpr, 4), partialmethod(op_gpr, 5), partialmethod(op_gpr, 6), partialmethod(op_gpr, 7)
  extrx, extry = partialmethod(op_gpr, 8), partialmethod(op_gpr, 9)
  fma64, fms64, fma32, fms32 = partialmethod(op_gpr, 10), partialmethod(op_gpr, 11), partialmethod(op_gpr, 12), partialmethod(op_gpr, 13)
  mac16, fma16, fms16 = partialmethod(op_gpr, 14), partialmethod(op_gpr, 15), partialmethod(op_gpr, 16)
  vecint, vecfp, matint, matfp, genlut = partialmethod(op_gpr, 18), partialmethod(op_gpr, 19), partialmethod(op_gpr, 20), partialmethod(op_gpr, 21), partialmethod(op_gpr, 22)

def int_const(x): return ir.Constant(ir.IntType(64), x)


N = 4096
# N = 1024
# N = 64

BW = N*N*4

# matrix is 64M, max load bandwidth is 57 GB/s
# cache line looks like 256 bytes (64 floats)

na = np.zeros((256), dtype=np.float32)
# na = np.zeros((N, N), dtype=np.float32)
nb = np.random.randn(N, N).astype(np.float32)
nc = np.random.randn(N, N).astype(np.float32)

ns = nb.reshape(-1, 32).sum(axis=0)

a = MallocAllocator.alloc(na.size * np.dtype(np.float32).itemsize)
b = MallocAllocator.alloc(nb.size * np.dtype(np.float32).itemsize)
c = MallocAllocator.alloc(nc.size * np.dtype(np.float32).itemsize)

MallocAllocator._copyin(b, flat_mv(nb.data))
MallocAllocator._copyin(c, flat_mv(nc.data))

module = ir.Module(name=__file__)
func = ir.Function(module, ir.FunctionType(ir.IntType(64), [ir.FloatType().as_pointer()]*3), name='exec')

# load all
entry = ir.IRBuilder(func.append_basic_block(name="entry"))
zm, xm, ym = [entry.ptrtoint(func.args[i], ir.IntType(64)) for i in range(3)]

loop_1 = ir.IRBuilder(func.append_basic_block(name="loop_y"))
loop_1_exit = ir.IRBuilder(func.append_basic_block(name="loop_y_exit"))
exit = ir.IRBuilder(func.append_basic_block(name="exit"))

y = loop_1.phi(ir.IntType(64), name="y")
y.add_incoming(int_const(0), entry._block)
yp = loop_1_exit.add(y, int_const(32*2))
y.add_incoming(yp, loop_1_exit._block)

prefetch_function = ir.Function(module, ir.FunctionType(ir.VoidType(), [ir.PointerType(ir.FloatType()), ir.IntType(32), ir.IntType(32), ir.IntType(32)]), name="llvm.prefetch")

xptr = y
addr = loop_1_exit.add(xm, loop_1_exit.mul(int_const(4), xptr))

#prefetch_ptr = loop_1_exit.inttoptr(loop_1_exit.add(addr, int_const(128)), ir.PointerType(ir.FloatType()))
#loop_1_exit.call(prefetch_function, [prefetch_ptr, ir.IntType(32)(0), ir.IntType(32)(2), ir.IntType(32)(1)])

AMX.ldx(loop_1_exit, loop_1_exit.add(int_const(1<<62), addr))
xptr = loop_1_exit.add(xptr, int_const(32))
AMX.ldy(loop_1_exit, loop_1_exit.add(int_const(1<<62), loop_1_exit.add(xm, loop_1_exit.mul(int_const(4), xptr))))

AMX.fma32(loop_1_exit, int_const(1 << 63 | 1 << 28))
AMX.fma32(loop_1_exit, int_const(1 << 63 | 1 << 28 | 1 << 20 | (16*4)<<10))
AMX.fma32(loop_1_exit, int_const(1 << 63 | 1 << 29))
AMX.fma32(loop_1_exit, int_const(1 << 63 | 1 << 29 | 1 << 20 | (16*4)))

AMX.set(entry)

AMX.stz(exit, exit.add(zm, int_const(1 << 62 | (0 << 56) | 0)))
AMX.clr(exit)

entry.branch(loop_1._block)
loop_1.branch(loop_1_exit._block)
loop_1_exit.cbranch(loop_1_exit.icmp_unsigned("==", yp, int_const(N*N)), exit._block, loop_1._block)
exit.ret(int_const(0))

device = LLVMDevice("llvm")
prog = LLVMProgram(device, "exec", LLVMCompiler(device).compile(str(module)))

"""
loop_1 = ir.IRBuilder(func.append_basic_block(name="loop_y"))
loop_2 = ir.IRBuilder(func.append_basic_block(name="loop_x"))
loop_3 = ir.IRBuilder(func.append_basic_block(name="loop_k"))
loop_3_exit = ir.IRBuilder(func.append_basic_block(name="loop_k_exit"))
loop_2_exit = ir.IRBuilder(func.append_basic_block(name="loop_x_exit"))
loop_1_exit = ir.IRBuilder(func.append_basic_block(name="loop_y_exit"))

y = loop_1.phi(ir.IntType(64), name="y")
x = loop_2.phi(ir.IntType(64), name="x")
k = loop_3.phi(ir.IntType(64), name="k")

exit = ir.IRBuilder(func.append_basic_block(name="exit"))

AMX.set(loop_2)

# stride
xptr = loop_3_exit.add(x, loop_3_exit.mul(k, int_const(N)))
yptr = loop_3_exit.add(y, loop_3_exit.mul(k, int_const(N)))

# if you are okay with the wrong answer, this is faster
#xptr = loop_3_exit.add(x, loop_3_exit.mul(k, int_const(32)))
#yptr = loop_3_exit.add(y, loop_3_exit.mul(k, int_const(32)))

# double loads load 32 floats
AMX.ldx(loop_3_exit, loop_3_exit.add(int_const(1<<62), loop_3_exit.add(xm, loop_3_exit.mul(int_const(4), xptr))))
AMX.ldy(loop_3_exit, loop_3_exit.add(int_const(1<<62), loop_3_exit.add(ym, loop_3_exit.mul(int_const(4), yptr))))

# <Z row> <X offset> <Y offset>
AMX.fma32(loop_3_exit, int_const(0<<20 | (0*16*4)<<10 | (0*16*4)))
AMX.fma32(loop_3_exit, int_const(1<<20 | (1*16*4)<<10 | (0*16*4)))
AMX.fma32(loop_3_exit, int_const(2<<20 | (0*16*4)<<10 | (1*16*4)))
AMX.fma32(loop_3_exit, int_const(3<<20 | (1*16*4)<<10 | (1*16*4)))

# store
gptr = loop_2_exit.mul(loop_2_exit.add(loop_2.mul(y, int_const(N)), x), int_const(4))
zmp = loop_2_exit.add(zm, gptr)
for j in range(2):
  for r in range(16):
    z_row = j*2
    ptr = ((j*16)+r)*N
    AMX.stz(loop_2_exit, loop_2_exit.add(zmp, int_const(1 << 62 | ((r*4+z_row) << 56) | ptr*4)))
AMX.clr(loop_2_exit)

yp = loop_1_exit.add(y, int_const(32))
xp = loop_2_exit.add(x, int_const(32))
kp = loop_3_exit.add(k, int_const(1))

y.add_incoming(int_const(0), entry._block)
x.add_incoming(int_const(0), loop_1._block)
k.add_incoming(int_const(0), loop_2._block)
y.add_incoming(yp, loop_1_exit._block)
x.add_incoming(xp, loop_2_exit._block)
k.add_incoming(kp, loop_3_exit._block)

entry.branch(loop_1._block)
loop_1.branch(loop_2._block)
loop_2.branch(loop_3._block)
loop_3.branch(loop_3_exit._block)
loop_3_exit.cbranch(loop_3_exit.icmp_unsigned("==", kp, int_const(N)), loop_2_exit._block, loop_3._block)
loop_2_exit.cbranch(loop_2_exit.icmp_unsigned("==", xp, int_const(N)), loop_1_exit._block, loop_2._block)
loop_1_exit.cbranch(loop_1_exit.icmp_unsigned("==", yp, int_const(N)), exit._block, loop_1._block)
exit.ret(int_const(0))

device = LLVMDevice("llvm")
prog = LLVMProgram(device, "exec", LLVMCompiler(device).compile(str(module)))
"""

def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  return time.perf_counter() - st

tm = min([timeit(lambda: prog(a, b, c, N**2)) for _ in range(20)])
MallocAllocator._copyout(flat_mv(na.data), a)
print(f"{N*N:10d} {tm*1e6:9.2f} us, {BW*1e-9/tm:.2f} GB/s")

np.testing.assert_allclose(na[:ns.shape[0]], ns, atol=1e-4, rtol=1e-4)

# comp = (nb.T @ nc).T
# np.testing.assert_allclose(na, comp, atol=1e-4, rtol=1e-5)