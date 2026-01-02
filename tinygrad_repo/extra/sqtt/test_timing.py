import os
os.environ["PYTHONPATH"] = "."
os.environ["SQTT"] = "1"
if "DEV" not in os.environ: os.environ["DEV"] = "AMD"
os.environ["PROFILE"] = "1"
# VIZ=1 to launch server
# os.environ["VIZ"] = "1"
os.environ["AMD_LLVM"] = "0"

import unittest
import sys, contextlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.device import Device, ProfileDeviceEvent

from extra.sqtt.roc import decode, WaveExec

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
  sqtt:dict[str, list[WaveExec]] = {}
  yield sqtt
  # decode sqtt
  if os.environ["DEV"] != "AMD": return
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
      for tc in dev.renderer.get_tensor_cores(dev.arch):
        M, K, N = tc.dims
        s = 32
        a = Tensor.empty(M*s, K*s, dtype=tc.dtype_in)@Tensor.empty(K*s, N*s, dtype=tc.dtype_in)
        a.realize()
        print(a)
    for p,waves in sqtt.items():
      for e in waves[0].insts:
        if (e.inst.startswith("v_wmma")):
          instruction = e.inst.split(" ")[0]
          print(f"{instruction:<29} : {e.dur} cycles")

  def test_sleep(self):
    n = 1
    def sleep_kernel(data0):
      assert data0.dtype.base == dtypes.ulong
      op = custom("unsigned long long t0 = __builtin_readcyclecounter();")
      op = custom(f"__builtin_amdgcn_s_sleep({n});", op)
      op = custom("unsigned long long t1 = __builtin_readcyclecounter();", op)
      op = custom(f"data0_{data0.size}[0] = t1 - t0;", op)
      return UOp.sink(data0, op, arg=KernelInfo(name=f"sleep_{n}"))
    diff_hw_reg = Tensor.empty(1, dtype=dtypes.ulong)
    diff_hw_reg = Tensor.custom_kernel(diff_hw_reg, fxn=sleep_kernel)[0]
    with save_sqtt() as sqtt:
      diff_hw_reg.realize()
    sleep = next((e for e in sqtt[f"sleep_{n}"][0].insts if e.inst.startswith("s_sleep")))
    # cycles = sleep dur + overhead of storing hi/lo REG_SHADER_CYCLES
    self.assertGreaterEqual(diff_hw_reg.item(), sleep.dur)

  def test_nop(self):
    with save_sqtt() as sqtt:
      asm_kernel(["s_nop 1"]*10).realize()
    wave = list(sqtt.values())[0][0]
    for e in wave.insts:
      print(f"{e.inst} {e.dur=} {e.stall=}")

  def test_wave_sched(self):
    num_waves = getenv("NUM_WAVES", 16)
    num_wgps = getenv("NUM_WGPS", 2)
    num_vgpr = getenv("NUM_VGPR", 256)
    with save_sqtt() as sqtt:
      # 1 cycle decode, no stall
      asm_kernel([f"v_mov_b32_e32 v{i} {i}" for i in range(num_vgpr)], l=32*num_waves, g=num_wgps).realize()
    waves = list(sqtt.values())[0]
    print(len(waves), "waves decoded")
    for w in waves:
      print(f"{w.wave_id:<2} {w.simd=} {w.cu=} {w.se=} @ clk {w.begin_time}")

  def test_ones(self):
    N = getenv("N", 4096)
    CNT = getenv("CNT", 2)
    with save_sqtt() as sqtt:
      for _ in range(CNT):
        Tensor.ones(N, N).contiguous().realize()
    self.assertEqual(len(sqtt), CNT)

if __name__ == "__main__":
  unittest.main()
