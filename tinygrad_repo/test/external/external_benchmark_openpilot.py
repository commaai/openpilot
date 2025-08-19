import time, sys, hashlib
from pathlib import Path
from tinygrad.frontend.onnx import OnnxRunner
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import IMAGE, GlobalCounters, fetch, colored, getenv, trange
import numpy as np
from extra.bench_log import BenchEvent, WallTimeEvent

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

if __name__ == "__main__":
  run_onnx = OnnxRunner(fetch(OPENPILOT_MODEL))

  Tensor.manual_seed(100)
  input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
  input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}
  new_inputs = {k:Tensor.randn(*shp, dtype=input_types[k]).mul(8).realize() for k,shp in input_shapes.items()}
  new_inputs_junk = {k:Tensor.randn(*shp, dtype=input_types[k]).mul(8).realize() for k,shp in input_shapes.items()}
  new_inputs_junk_numpy = {k:v.numpy() for k,v in new_inputs_junk.items()}

  # benchmark
  for _ in range(5):
    GlobalCounters.reset()
    st = time.perf_counter_ns()
    ret = next(iter(run_onnx(new_inputs_junk).values())).cast(dtypes.float32).numpy()
    print(f"unjitted: {(time.perf_counter_ns() - st)*1e-6:7.4f} ms")

  # NOTE: the inputs to a JIT must be first level arguments
  run_onnx_jit = TinyJit(lambda **kwargs: run_onnx(kwargs), prune=True)
  for _ in range(20):
    GlobalCounters.reset()
    st = time.perf_counter_ns()
    with WallTimeEvent(BenchEvent.STEP):
      # Need to cast non-image inputs from numpy, this is only realistic way to run model
      inputs = {**{k:v for k,v in new_inputs_junk.items() if 'img' in k},
                **{k:Tensor(v) for k,v in new_inputs_junk_numpy.items() if 'img' not in k}}
      ret = next(iter(run_onnx_jit(**inputs).values())).cast(dtypes.float32).numpy()
    print(f"jitted:  {(time.perf_counter_ns() - st)*1e-6:7.4f} ms")

  suffix = ""
  if IMAGE.value < 2: suffix += f"_image{IMAGE.value}" # image=2 has no suffix for compatibility
  if getenv("FLOAT16") == 1: suffix += "_float16"
  path = Path(__file__).parent / "openpilot" / f"{hashlib.md5(OPENPILOT_MODEL.encode()).hexdigest()}{suffix}.npy"

  # validate if we have records
  tinygrad_out = next(iter(run_onnx_jit(**new_inputs).values())).cast(dtypes.float32).numpy()
  if getenv("SAVE_OUTPUT"):
    np.save(path, tinygrad_out)
    print(f"saved output to {path}!")
  elif getenv("FUZZ") and path.exists():
    known_good_out = np.load(path)
    for _ in trange(1000):
      ret = next(iter(run_onnx_jit(**new_inputs).values())).cast(dtypes.float32).numpy()
      np.testing.assert_allclose(known_good_out, ret, atol=1e-2, rtol=1e-2)
    print(colored("fuzz validated!", "green"))
  elif path.exists():
    known_good_out = np.load(path)
    np.testing.assert_allclose(known_good_out, tinygrad_out, atol=1e-2, rtol=1e-2)
    print(colored("outputs validated!", "green"))
  else:
    print(colored("skipping validation", "yellow"))
