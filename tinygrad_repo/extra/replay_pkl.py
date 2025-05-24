import pickle, sys
from dataclasses import replace
from tinygrad import Device, Context, Tensor, GlobalCounters
from tinygrad.device import Buffer
from tinygrad.helpers import getenv, BEAM
from tinygrad.engine.jit import TinyJit
from tinygrad.engine.realize import CompiledRunner, ExecItem, ScheduleItem, lower_schedule_item
from tinygrad.renderer import ProgramSpec
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.codegen.heuristic import hand_coded_optimizations
import numpy as np

def move_jit_captured_to_dev(captured, device="DSP"):
  captured.expected_st_vars_dtype_device = [x[:3] + (device,) for x in captured.expected_st_vars_dtype_device]

  assign = {}
  def move_buffer(b):
    if b in assign: return assign[b]

    if b._base is not None:
      newbuf = Buffer(device, b.size, b.dtype, base=move_buffer(b._base), offset=b.offset)
    else:
      newbuf = Buffer(device, b.size, b.dtype)
      if b.is_allocated(): newbuf.ensure_allocated().copyin(b.as_buffer())
    assign[b] = newbuf
    return assign[b]

  for item in captured.jit_cache:
    for b in item.bufs:
      if b is not None: move_buffer(b)
  captured.jit_cache = [ExecItem(item.prg, [assign.get(b,b) for b in item.bufs]) for item in captured.jit_cache]
  return captured

if __name__ == "__main__":
  with Context(DEBUG=0):
    with open(sys.argv[1], "rb") as f:
      fxn: TinyJit = pickle.load(f)
      print(f"{f.tell()/1e6:.2f}M loaded")
    print(type(fxn))

  # Move all buffers to DSP device.
  fxn.captured = move_jit_captured_to_dev(fxn.captured, "DSP")
  new_jit = []

  knum = 1
  for ei in fxn.captured.jit_cache:
    # skip the copy and the first kernel
    if isinstance(ei.prg, CompiledRunner) and all(x is not None for x in ei.bufs):
      if knum == (pknum:=getenv("KNUM", 0)) or pknum == 0:
        p: ProgramSpec = ei.prg.p
        k = Kernel(p.ast, Device["DSP"].renderer)

        if getenv("VALIDATE"):
          with Context(NOOPT=1):
            lower_schedule_item(ScheduleItem(p.ast, ei.bufs)).run()
            correct = ei.bufs[0].numpy()
            ei.bufs[0].copyin(memoryview(bytearray(b'\x00'*ei.bufs[0].nbytes)))
            GlobalCounters.kernel_count -= 1

        if not getenv("NOOPT"): k.apply_opts(hand_coded_optimizations(k))
        p2 = k.to_program()
        new_ei = replace(ei, prg=CompiledRunner(p2))
        new_ei.run()
        new_jit.append(new_ei)
        test = ei.bufs[0].numpy()

        if getenv("VALIDATE"):
          import numpy as np
          np.testing.assert_allclose(correct, test, rtol=1e-3, atol=1e-3)
      knum += 1

  if getenv("RUN_JIT", 0):
    fxn.captured.free_intermediates()
    fxn.captured.jit_cache = new_jit
    fxn(input=Tensor(np.zeros((1, 3, 224, 224), dtype=np.float32), device="DSP"))
