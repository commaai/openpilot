import os, sys, pickle, time, re
import numpy as np
if "JIT_BATCH_SIZE" not in os.environ: os.environ["JIT_BATCH_SIZE"] = "0"

from tinygrad import fetch, Tensor, TinyJit, Context, GlobalCounters, Device, dtypes
from tinygrad.helpers import DEBUG, getenv
from tinygrad.engine.realize import CompiledRunner
from tinygrad.nn.onnx import OnnxRunner

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/openpilot.pkl"

def compile(onnx_file):
  run_onnx = OnnxRunner(onnx_file)
  print("loaded model")

  input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
  input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}

  # Float inputs and outputs to tinyjits for openpilot are always float32
  # TODO this seems dumb
  input_types = {k:(dtypes.float32 if v is dtypes.float16 else v) for k,v in input_types.items()}
  Tensor.manual_seed(100)
  inputs = {k:Tensor(Tensor.randn(*shp, dtype=input_types[k]).mul(8).realize().numpy(), device='NPY') for k,shp in sorted(input_shapes.items())}
  if not getenv("NPY_IMG"):
    inputs = {k:Tensor(v.numpy(), device=Device.DEFAULT).realize() if 'img' in k else v for k,v in inputs.items()}
  print("created tensors")

  run_onnx_jit = TinyJit(lambda **kwargs:
                         next(iter(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values())).cast('float32'), prune=True)
  for i in range(3):
    GlobalCounters.reset()
    print(f"run {i}")
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
      ret = run_onnx_jit(**inputs).numpy()
    # copy i == 1 so use of JITBEAM is okay
    if i == 1: test_val = np.copy(ret)
  print(f"captured {len(run_onnx_jit.captured.jit_cache)} kernels")
  np.testing.assert_equal(test_val, ret, "JIT run failed")
  print("jit run validated")

  # check gated read_image usage
  kernel_count = 0
  read_image_count = 0
  gated_read_image_count = 0
  for ei in run_onnx_jit.captured.jit_cache:
    if isinstance(ei.prg, CompiledRunner):
      kernel_count += 1
      read_image_count += ei.prg.p.src.count("read_image")
      gated_read_image_count += ei.prg.p.src.count("?read_image")
      for v in [m.group(1) for m in re.finditer(r'(val\d+)\s*=\s*read_imagef\(', ei.prg.p.src)]:
        if len(re.findall(fr'[\?\:]{v}\.[xyzw]', ei.prg.p.src)) > 0: gated_read_image_count += 1
  print(f"{kernel_count=},  {read_image_count=}, {gated_read_image_count=}")
  if (allowed_kernel_count:=getenv("ALLOWED_KERNEL_COUNT", -1)) != -1:
    assert kernel_count == allowed_kernel_count, f"different kernels! {kernel_count=}, {allowed_kernel_count=}"
  if (allowed_read_image:=getenv("ALLOWED_READ_IMAGE", -1)) != -1:
    assert read_image_count == allowed_read_image, f"different read_image! {read_image_count=}, {allowed_read_image=}"
  if (allowed_gated_read_image:=getenv("ALLOWED_GATED_READ_IMAGE", -1)) != -1:
    assert gated_read_image_count == allowed_gated_read_image, f"different gated read_image! {gated_read_image_count=}, {allowed_gated_read_image=}"

  with open(OUTPUT, "wb") as f:
    pickle.dump(run_onnx_jit, f)
  mdl_sz = os.path.getsize(onnx_file)
  pkl_sz = os.path.getsize(OUTPUT)
  print(f"mdl size is {mdl_sz/1e6:.2f}M")
  print(f"pkl size is {pkl_sz/1e6:.2f}M")
  print("**** compile done ****")
  return inputs, test_val

def test_vs_compile(run, inputs, test_val=None):

  # run 20 times
  step_times = []
  for _ in range(20):
    st = time.perf_counter()
    out = run(**inputs)
    mt = time.perf_counter()
    val = out.numpy()
    et = time.perf_counter()
    step_times.append((et-st)*1e3)
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")

  if (assert_time:=getenv("ASSERT_MIN_STEP_TIME")):
    min_time = min(step_times)
    assert min_time < assert_time, f"Speed regression, expected min step time of < {assert_time} ms but took: {min_time} ms"

  if test_val is not None: np.testing.assert_equal(test_val, val)
  print("**** test done ****")

  # test that changing the numpy changes the model outputs
  inputs_2x = {k: Tensor(v.numpy()*2, device=v.device) for k,v in inputs.items()}
  out = run(**inputs_2x)
  changed_val = out.numpy()
  np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, val, changed_val)
  return val

def test_vs_onnx(new_inputs, test_val, onnx_file, tol):
  import onnx
  import onnxruntime as ort
  
  onnx_inputs = {k:v.numpy() for k,v in new_inputs.items()}
  onnx_model = onnx.load(onnx_file)

  ORT_TO_NP_DTYPES: dict[str, np.dtype] = {
    'tensor(float)': np.dtype('float32'),
    'tensor(float16)': np.dtype('float16'),
    'tensor(uint8)': np.dtype('uint8'),
  }

  timings = []
  onnx_session = ort.InferenceSession(onnx_file)
  onnx_types = {x.name: ORT_TO_NP_DTYPES[x.type] for x in onnx_session.get_inputs()}
  onnx_inputs = {k:onnx_inputs[k].astype(onnx_types[k]) for k in onnx_inputs}

  for _ in range(1 if test_val is not None else 5):
    st = time.perf_counter()
    onnx_output = onnx_session.run([onnx_model.graph.output[0].name], onnx_inputs)
    timings.append(time.perf_counter() - st)

  np.testing.assert_allclose(onnx_output[0].reshape(test_val.shape), test_val, atol=tol, rtol=tol)
  print("test vs onnx passed")
  return timings

def bench(run, inputs):
  from extra.bench_log import WallTimeEvent, BenchEvent
  for _ in range(10):
    with WallTimeEvent(BenchEvent.STEP):
      run(**inputs).numpy()

if __name__ == "__main__":
  onnx_file = fetch(OPENPILOT_MODEL)
  inputs, outputs = compile(onnx_file)

  with open(OUTPUT, "rb") as f: pickle_loaded = pickle.load(f)

  test_vs_compile(pickle_loaded, inputs, outputs)
  if getenv("SELFTEST"):
    test_vs_onnx(inputs, outputs, onnx_file, 1e-4)

  if getenv("BENCHMARK_LOG", ""):
    bench(pickle_loaded, inputs)
