from tinygrad import Tensor, dtypes, GlobalCounters
from tinygrad.engine.realize import get_program

if __name__ == "__main__":
  t = Tensor.empty(81920, 4096, dtype=dtypes.half)
  GlobalCounters.reset()
  t.softmax(-1, dtype="half").realize()
  GlobalCounters.reset()
  t.softmax(-1, dtype="half", _single_kernel=True).realize()

  from tinygrad.codegen.opt.kernel import Kernel, Opt, OptOps
  from tinygrad.helpers import get_single_element
  GlobalCounters.reset()
  si = get_single_element(t.softmax(-1, dtype="half", _single_kernel=True).schedule())
  k = Kernel(si.ast)
  #k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
  k.apply_opt(Opt(OptOps.UPCAST, 1, 4))
  k.apply_opt(Opt(OptOps.LOCAL, 1, 32))
  #k.apply_opt(Opt(OptOps.LOCAL, 0, 8))
  k.apply_opt(Opt(OptOps.UNROLL, 1, 4))
  k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
  #k.apply_opt(Opt(OptOps.GROUP, 1, 256))
  #k.apply_opt(Opt(OptOps.GROUP, 0, 32))
  #k.apply_opt(Opt(OptOps.GROUP, 1, 32))
  #k.apply_opt(Opt(OptOps.GROUP, 0, 32))
  from tinygrad.engine.realize import CompiledRunner, ExecItem
  run = CompiledRunner(prg:=get_program(k.get_optimized_ast(), k.opts))
  ExecItem(run, si.bufs).run()
