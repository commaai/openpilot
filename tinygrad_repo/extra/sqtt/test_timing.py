import os
os.environ["PYTHONPATH"] = "."
os.environ["SQTT"] = "1"
if "DEV" not in os.environ: os.environ["DEV"] = "AMD"
os.environ["VIZ"] = "1"
os.environ["AMD_LLVM"] = "0"

import unittest
import sys, contextlib
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.renderer import ProgramSpec
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AddrSpace
from tinygrad.engine.realize import CompiledRunner
from tinygrad.device import Device, ProfileDeviceEvent

from extra.sqtt.roc import decode, InstExec, PrgExec

dev = Device[os.environ["DEV"]]

def custom(arg:str, s:UOp|None=None) -> UOp: return UOp(Ops.CUSTOM, src=(s,) if s is not None else (), arg=arg)

def asm_kernel(instrs:list[str], l:int=1, g:int=1) -> Tensor:
  name = sys._getframe(1).f_code.co_name
  def fxn(_):
    L = UOp.special(l, "lidx0")
    G = UOp.special(g, "gidx0")
    op = custom("asm volatile (")
    for inst in instrs: op = custom(f'  "{inst}\\n\\t"', op)
    op = custom(");", op)
    return UOp.sink(op, L, G, arg=KernelInfo(name=name))
  k = Tensor.custom_kernel(Tensor.empty(1), fxn=fxn)[0]
  return k

@contextlib.contextmanager
def save_sqtt():
  # clear the old traces
  dev.profile_events.clear()
  sqtt:dict[PrgExec, list[InstExec]] = {}
  yield sqtt
  # decode sqtt
  if os.environ["DEV"] == "AMD":
    rctx = decode(dev.profile_events+[ProfileDeviceEvent("AMD", props=dev.device_props())])
    assert len(rctx.inst_execs) > 0, "empty sqtt output"
    sqtt.update(rctx.inst_execs)

class TestTiming(unittest.TestCase):
  def test_v_add(self):
    with save_sqtt() as sqtt:
      asm_kernel([f"v_add_f32 v{10+i} v{10+i+1} {10+i}" for i in range(3)]).realize()
    wave = list(sqtt.values())[0][:-1]
    assert all(s.dur == 1 for s in wave)
    assert all(s.stall == 0 for s in wave)

  def test_chain_v_add_1l(self):
    with save_sqtt() as sqtt:
      asm_kernel([
        "v_add_f32_e32 v1 v0 v0",
        "v_add_f32_e32 v2 v1 v1",
      ]).realize()
    wave = list(sqtt.values())[0][:-1]
    assert all(s.dur == 1 for s in wave)
    assert all(s.stall == 0 for s in wave)

  def test_multi_cycle_inst(self):
    def custom_vrcp(A, B):
      op = custom("float a = 0.0;")
      op = custom("float b = (*(data1_1+0));", op)
      #op = custom('asm volatile("v_mul_f32_e32 %2 %2 %1" : "+v"(a) : "v"(b));', op)
      op = custom('asm volatile("v_rcp_f32_e32 %2 %1" : "+v"(a) : "v"(b));', op)
      op = custom('asm volatile("v_add_f32_e64 %1 %1 1.0" : "+v"(a));', op)
      op = custom("*(data0_1+0) = a;", op)
      return UOp.sink(op, A, B, arg=KernelInfo(name="custom_vrcp"))
    out = Tensor([0.]).realize()
    inp = Tensor([-2.0]).realize()
    with save_sqtt() as sqtt:
      Tensor.custom_kernel(out, inp, fxn=custom_vrcp)[0].realize()

    wave = list(sqtt.values())[0][0]
    for i in range(len(wave.insts)):
      if wave.insts[i].inst.startswith("global_store"):
        print(f"store diff {wave.insts[i].time-(wave.insts[i-1].time)}")
    self.assertEqual(out.item(), 0.5)

  def test_wmma(self):
    with save_sqtt() as sqtt:
      asm_kernel([
        "v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]",
        "v_add_f32_e32 v0 v16 v0",
      ], l=32*4).realize()
    assert len(sqtt) == 2, f"expected two waves, got {len(sqtt)} {list(sqtt.keys())}"
    wmma = list(sqtt.values())[0][0]
    self.assertGreater(wmma.dur, 1) # rgp says 32 clocks

  def test_sleep(self):
    n = 1
    def sleep_kernel(data0):
      assert data0.dtype.base == dtypes.ulong
      op = custom("unsigned long long t0 = __builtin_readcyclecounter();")
      op = custom(f"__builtin_amdgcn_s_sleep({n});", op)
      op = custom(f"unsigned long long t1 = __builtin_readcyclecounter();", op)
      op = custom(f"data0_{data0.size}[0] = t1 - t0;", op)
      return UOp.sink(data0, op, arg=KernelInfo(name=f"sleep_{n}"))
    diff_hw_reg = Tensor.empty(1, dtype=dtypes.ulong)
    diff_hw_reg = Tensor.custom_kernel(diff_hw_reg, fxn=sleep_kernel)[0]
    with save_sqtt() as sqtt:
      diff_hw_reg.realize()
    diff_sqtt = list(sqtt.values())[0][2]
    self.assertEqual(diff_sqtt.dur, diff_hw_reg.item()-1) # 1 cycle for reading the counter register

if __name__ == "__main__":
  unittest.main()
