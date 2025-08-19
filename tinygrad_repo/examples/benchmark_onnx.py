import sys, time
from tinygrad import TinyJit, GlobalCounters, fetch, getenv
from tinygrad.frontend.onnx import OnnxRunner
from extra.onnx_helpers import get_example_inputs, validate

def load_onnx_model(onnx_file):
  run_onnx = OnnxRunner(onnx_file)
  run_onnx_jit = TinyJit(lambda **kwargs: next(iter(run_onnx({k:v.to(None) for k,v in kwargs.items()}).values())), prune=True, optimize=True)
  return run_onnx_jit, run_onnx.graph_inputs

if __name__ == "__main__":
  onnx_file = fetch(sys.argv[1])
  run_onnx_jit, input_specs = load_onnx_model(onnx_file)
  print("loaded model")

  for i in range(3):
    new_inputs = get_example_inputs(input_specs)
    GlobalCounters.reset()
    print(f"run {i}")
    run_onnx_jit(**new_inputs)

  # run 20 times
  for _ in range(20):
    new_inputs = get_example_inputs(input_specs)
    GlobalCounters.reset()
    st = time.perf_counter()
    out = run_onnx_jit(**new_inputs)
    mt = time.perf_counter()
    val = out.numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")

  if getenv("ORT"):
    validate(onnx_file, new_inputs, rtol=1e-3, atol=1e-3)
    print("model validated")
