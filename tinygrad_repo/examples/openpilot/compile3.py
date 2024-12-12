import os, sys, pickle, time
import numpy as np
if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"
if "JIT_BATCH_SIZE" not in os.environ: os.environ["JIT_BATCH_SIZE"] = "0"

from tinygrad import fetch, Tensor, TinyJit, Context, GlobalCounters, Device
from tinygrad.helpers import DEBUG, getenv
from tinygrad.tensor import _from_np_dtype

import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from extra.onnx import get_run_onnx   # TODO: port to main tinygrad

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/openpilot.pkl"

def compile():
  Tensor.no_grad = True
  Tensor.training = False

  onnx_bytes = fetch(OPENPILOT_MODEL)
  onnx_model = onnx.load(onnx_bytes)
  run_onnx = get_run_onnx(onnx_model)
  print("loaded model")

  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}
  if getenv("FLOAT16", 0) == 0: input_types = {k:(np.float32 if v==np.float16 else v) for k,v in input_types.items()}
  input_types = {inp.name: np.float32 for inp in onnx_model.graph.input}
  if 'input_img' in input_shapes:
    input_types['input_img'] = np.uint8
  else:
    input_types['input_imgs'] = np.uint8
    input_types['big_input_imgs'] = np.uint8
  Tensor.manual_seed(100)
  new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in sorted(input_shapes.items())}
  new_inputs_numpy = {k:v.numpy() for k,v in new_inputs.items()}
  print("created tensors")

  run_onnx_jit = TinyJit(lambda **kwargs:
                         next(iter(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values())).cast('float32'), prune=True)
  for i in range(3):
    GlobalCounters.reset()
    print(f"run {i}")
    inputs = {**{k:v.clone() for k,v in new_inputs.items() if 'imgs' in k},
              **{k:Tensor(v, device="NPY").realize() for k,v in new_inputs_numpy.items() if 'imgs' not in k}}
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
      ret = run_onnx_jit(**inputs).numpy()
    # copy i == 1 so use of JITBEAM is okay
    if i == 1: test_val = np.copy(ret)
  print(f"captured {len(run_onnx_jit.captured.jit_cache)} kernels")
  np.testing.assert_equal(test_val, ret, "JIT run failed")
  print("jit run validated")

  with open(OUTPUT, "wb") as f:
    pickle.dump(run_onnx_jit, f)
  mdl_sz = os.path.getsize(onnx_bytes)
  pkl_sz = os.path.getsize(OUTPUT)
  print(f"mdl size is {mdl_sz/1e6:.2f}M")
  print(f"pkl size is {pkl_sz/1e6:.2f}M")
  print("**** compile done ****")
  return test_val

def test(test_val=None):
  with open(OUTPUT, "rb") as f:
    run = pickle.load(f)

  # same randomness as above
  Tensor.manual_seed(100)
  new_inputs = {nm:Tensor.randn(*st.shape, dtype=dtype).mul(8).realize() for nm, (st, _, dtype, _) in
                sorted(zip(run.captured.expected_names, run.captured.expected_st_vars_dtype_device))}
  new_inputs_numpy = {k:v.numpy() for k,v in new_inputs.items()}

  # create fake "from_blob" tensors for the inputs, and wrapped NPY tensors for the numpy inputs (these have the same underlying memory)
  inputs = {**{k:v for k,v in new_inputs.items() if 'imgs' in k},
            **{k:Tensor(v, device="NPY").realize() for k,v in new_inputs_numpy.items() if 'imgs' not in k}}

  # run 20 times
  for _ in range(20):
    inputs_numpy = {k:v.numpy() for k,v in new_inputs.items()}
    st = time.perf_counter()
    out = run(**inputs)
    mt = time.perf_counter()
    val = out.numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")
  print(out, val.shape, val.dtype)
  if test_val is not None: np.testing.assert_equal(test_val, val)
  print("**** test done ****")

  # test that changing the numpy changes the model outputs
  for v in new_inputs_numpy.values(): v *= 2
  out = run(**inputs)
  changed_val = out.numpy()
  np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, val, changed_val)

if __name__ == "__main__":
  test_val = compile() if not getenv("RUN") else None
  test(test_val)

