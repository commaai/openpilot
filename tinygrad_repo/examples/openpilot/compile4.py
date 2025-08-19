import sys
from tinygrad import Tensor, fetch, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp
from tinygrad.frontend.onnx import OnnxRunner
from tinygrad.schedule.kernelize import get_kernelize_map
from tinygrad.engine.schedule import create_schedule_with_vars
from tinygrad.engine.realize import run_schedule

# NOLOCALS=1 GPU=1 IMAGE=2 FLOAT16=1 VIZ=1 DEBUG=2 python3 examples/openpilot/compile4.py

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/openpilot.pkl"

if __name__ == "__main__":
  onnx_file = fetch(OPENPILOT_MODEL)
  run_onnx = OnnxRunner(onnx_file)

  inputs = run_onnx.get_empty_input_data("npy", dtypes.float32)
  out: Tensor = next(iter(run_onnx({k:v.to(None) for k,v in inputs.items()}).values())).to('cpu')
  root = out.uop
  targets = [x.uop for x in inputs.values()]
  print(targets)

  # TODO: abstract this from gradient?

  # compute the target path (top down)
  in_target_path: dict[UOp, bool] = {}
  for u in root.toposort(): in_target_path[u] = any(x in targets or in_target_path[x] for x in u.src)
  independent_set = {}
  for u in root.toposort():
    if in_target_path[u]:
      for s in u.src:
        if not in_target_path[s]:
          independent_set[s] = None
  independent = UOp.sink(*independent_set.keys())
  kernelized = get_kernelize_map(independent)
  independent = independent.substitute(kernelized)
  schedule, var_vals = create_schedule_with_vars(independent)
  run_schedule(schedule)

  print("**** real ****")
  GlobalCounters.reset()
  out.uop = root.substitute(kernelized)
  out.kernelize()

  # realize
  out.realize()
