import pickle, sys
from dataclasses import replace
from tinygrad import Device
from tinygrad.helpers import getenv
from tinygrad.engine.jit import TinyJit
from tinygrad.engine.realize import CompiledRunner
from tinygrad.renderer import ProgramSpec
from tinygrad.codegen.kernel import Kernel, Opt, OptOps

if __name__ == "__main__":
  with open(sys.argv[1], "rb") as f:
    fxn: TinyJit = pickle.load(f)
    print(f"{f.tell()/1e6:.2f}M loaded")
  print(type(fxn))

  knum = 1
  for ei in fxn.captured.jit_cache:
    # skip the copy and the first kernel
    if isinstance(ei.prg, CompiledRunner) and all(x is not None for x in ei.bufs):
      if knum == (pknum:=getenv("KNUM", 0)) or pknum == 0:
        p: ProgramSpec = ei.prg.p
        k = Kernel(p.ast, Device["DSP"].renderer)
        if not getenv("NOOPT"):
          if knum == 2:
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=1, arg=0))
            k.apply_opt(Opt(op=OptOps.UNROLL, axis=0, arg=0))
            k.apply_opt(Opt(OptOps.PADTO, 2, 128))
            k.apply_opt(Opt(OptOps.UPCAST, 2, 128))
          else:
            k.hand_coded_optimizations()
        p2 = k.to_program()
        new_ei = replace(ei, prg=CompiledRunner(p2))
        new_ei.run()
      knum += 1

