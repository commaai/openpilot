from extra.models.resnet import ResNet50
from tinygrad import Tensor, nn, Device
from tinygrad.helpers import Profiling, Timing, getenv
from tinygrad.uop.ops import Ops
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.uop.spec import type_verify, program_spec

if __name__ == "__main__":
  mdl = ResNet50()
  for p in nn.state.get_parameters(mdl): p.replace(Tensor.empty(p.shape))
  img = Tensor.empty(64, 3, 224, 224)

  PROFILE = getenv("PYPROFILE", 0)
  FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
  SCHEDULE_ONLY = getenv("SCHEDULE_ONLY", 0)
  LINEARIZE = bool(getenv("LINEARIZE", 1))

  with Timing("all "):
    with Timing("***** model tensor in    "):
      out = mdl(img)

    if not FORWARD_ONLY:
      with Timing("***** model schedule in  "):
        with Profiling(PROFILE >= 3):
          sched = out.schedule()

      if not SCHEDULE_ONLY:
        asts = list({x.ast.key:x.ast for x in sched if x.ast.op is Ops.SINK}.values())
        if (restrict_kernel := getenv("RESTRICT_KERNEL", -1)) != -1: asts = asts[restrict_kernel:restrict_kernel+1]

        with Profiling(PROFILE, fn="/tmp/rewrite.prof"):
          with Timing("***** model rewrite in   "):
            rewritten_uops = []
            for u in asts:
              rewritten_uops.append(full_rewrite_to_sink(u, ren=Device.default.renderer))

        if LINEARIZE:
          with Timing("***** model linearize in "):
            uops_line = []
            for u in rewritten_uops:
              uops_line.append(linearize(u))
          with Timing("***** model verify in    "):
            for u in uops_line: type_verify(u, program_spec)
          print(sum(len(u) for u in uops_line))
