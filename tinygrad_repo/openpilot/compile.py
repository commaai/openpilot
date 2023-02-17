#!/usr/bin/env python3
import os, time, io, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

if os.getenv("OPT", None) is None:
  os.environ['OPT'] = '99'
if os.getenv("GPU", None) is None:
  os.environ['OPENCL'] = '1'

ALLOWED_KERNEL_COUNT = int(os.getenv("ALLOWED_KERNEL_COUNT", 0))
DEBUGCL = int(os.getenv("DEBUGCL", 0))

import onnx
import numpy as np

import tinygrad.graph as graph
from tinygrad.ops import GlobalCounters

from tinygrad.llops.ops_gpu import CL
from extra.utils import fetch
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/6c5693e965b9c63f8678f52b9e9b5abe35f23feb/selfdrive/modeld/models/supercombo.onnx"

np.random.seed(1337)
def get_random_input_tensors(input_shapes):
  np_inputs = {
    "input_imgs": np.random.randn(*(1, 12, 128, 256))*256,
    "big_input_imgs": np.random.randn(*(1, 12, 128, 256))*256,
    "desire": np.zeros((1,100, 8)),
    "traffic_convention": np.array([[1., 0.]]),
    #"features_buffer": np.random.randn(*(1, 99, 128))
    "features_buffer": np.random.randn(*input_shapes['features_buffer'])
    #"initial_state": np.zeros((1, 768))
  }
  if int(os.getenv("ZERO_OUT", "0")):
    np_inputs = {k:v*0 for k,v in np_inputs.items()}

  for k,v in np_inputs.items():
    assert v.shape == input_shapes[k], f"{k} shape mismatch, {v.shape} {input_shapes[k]}"

  #import pickle
  #frames, big_frames, last_state, frame_inputs, policy_outs = pickle.load(open("openpilot/test/frame_0.pkl", "rb"))
  #np_inputs["input_imgs"] = frames
  #np_inputs["big_input_imgs"] = big_frames
  #np_inputs["initial_state"] = last_state[0]

  #for i,k in enumerate(np_inputs.keys()):
  #  dat = open("/home/batman/openpilot/xx/ml_tools/snpe/compile_test_data/dlc_input_%d" % i, "rb").read()
  #  np_inputs[k] = np.frombuffer(dat, np.float32).reshape(np_inputs[k].shape)

  np_inputs = {k:v.astype(np.float32) for k,v in np_inputs.items()}
  inputs = {k:Tensor(v.astype(np.float32), requires_grad=False) for k,v in np_inputs.items()}
  for _,v in inputs.items(): v.realize()
  return inputs, np_inputs

def compile(dat, output_fn):
  Tensor.no_grad = True
  using_graph = graph.GRAPH
  graph.GRAPH = False

  onnx_model = onnx.load(io.BytesIO(dat))
  run_onnx = get_run_onnx(onnx_model)

  input_shapes = {}
  for inp in onnx_model.graph.input:
    input_shapes[inp.name] = tuple(x.dim_value for x in inp.type.tensor_type.shape.dim)

  inputs, _ = get_random_input_tensors(input_shapes)

  # initial run(s) to load weights
  for _ in range(2):
    st = time.monotonic()
    tinygrad_out = next(iter(run_onnx(inputs).values()))
    mt = time.monotonic()
    tinygrad_out.realize()
    mt2 = time.monotonic()
    tinygrad_out = tinygrad_out.numpy()
    et = time.monotonic()
    print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(mt2-mt)*1000.0:.2f} ms for realize, {(et-mt2)*1000.0:.2f} ms for GPU queue")

    # realize all non GCed tensors (fix for batchnorm folding)
    import gc
    gc.collect()
    for x in [x for x in gc.get_objects() if isinstance(x, Tensor)]:
      x.realize()

  # real run
  inputs, np_inputs = get_random_input_tensors(input_shapes)
  print("***** REAL RUN *****")
  tinygrad_out = next(iter(run_onnx(inputs).values()))

  # note, since CL.CACHE is enabled, it doesn't actually run the kernels
  start_ops = GlobalCounters.global_ops
  CL.CACHE = []
  if using_graph: graph.GRAPH = True
  CL.kernel_count = -1
  tinygrad_out.realize()
  graph.GRAPH = False
  print("kernel count:", len(CL.CACHE))
  assert len(CL.CACHE) <= ALLOWED_KERNEL_COUNT or ALLOWED_KERNEL_COUNT == 0, "too many kernels!"
  used_ops = GlobalCounters.global_ops - start_ops

  from extra.thneed import Thneed
  t = Thneed(CL.CACHE, {k:inputs[k].lazydata.realized.cl for k in inputs.keys()})
  CL.CACHE = None
  if int(os.getenv("OPTWG", "0")):
    t.optimize_local_workgroup()

  # save thneed (before run)
  t.save(output_fn)

  print(f"buffers to save: {len(t.buffers_to_save)}, outputs: {t.outputs}")
  runtime = t.run()
  print(f"network using {used_ops/1e9:.2f} GOPS with runtime {runtime*1e3:.2f} ms that's {used_ops/runtime*1e-9:.2f} GFLOPS")

  # confirm thneed found the right output
  thneed_out = np.empty((t.outputs[0].size//4,), dtype=np.float32).reshape(tinygrad_out.shape)
  CL.enqueue_copy(thneed_out, t.outputs[0], is_blocking=True)
  np.testing.assert_allclose(thneed_out, tinygrad_out.numpy())

  # float32 only (fix this)
  FLOAT16 = int(os.getenv("FLOAT16", 0))
  if FLOAT16 == 0:
    try:
      from test.test_onnx import run_onnx_torch
      torch_out = run_onnx_torch(onnx_model, np_inputs).numpy()
      print(thneed_out, torch_out, "mse", np.sum((thneed_out-torch_out)**2), "max err", np.max(np.abs((thneed_out-torch_out))))
      np.testing.assert_allclose(torch_out, thneed_out, atol=1e-4, rtol=1e-2)

      # test loading/run thneed
      _, new_np_inputs = get_random_input_tensors(input_shapes)
      new_torch_out = run_onnx_torch(onnx_model, new_np_inputs).numpy()

      # try old thneed with a different input
      for k,v in t.inputs.items():
        CL.enqueue_copy(v, new_np_inputs[k], is_blocking=True)

      t.run()
      old_thneed_out = np.empty((t.outputs[0].size//4,), dtype=np.float32).reshape(tinygrad_out.shape)
      CL.enqueue_copy(old_thneed_out, t.outputs[0], is_blocking=True)

      # compare thneed (rerun) with torch
      np.testing.assert_allclose(new_torch_out, old_thneed_out, atol=1e-4, rtol=1e-2)

      # load thneed and try that
      _, new_np_inputs = get_random_input_tensors(input_shapes)
      new_torch_out = run_onnx_torch(onnx_model, new_np_inputs).numpy()
      nt = Thneed()
      nt.load(output_fn)

      # inputs
      for k,v in nt.inputs.items():
        CL.enqueue_copy(v, new_np_inputs[k], is_blocking=True)

      nt.run()
      new_thneed_out = np.empty((nt.outputs[0].size//4,), dtype=np.float32).reshape(tinygrad_out.shape)
      CL.enqueue_copy(new_thneed_out, nt.outputs[0], is_blocking=True)

      # compare torch to thneed
      np.testing.assert_allclose(new_torch_out, new_thneed_out, atol=1e-4, rtol=1e-2)
      print("thneed self-test passed!")
    except ModuleNotFoundError:
      pass
  


# UNSAFE_FLOAT4=1 DEBUGCL=1 FLOAT16=1 python3 openpilot/compile.py
# 22.59 ms
if __name__ == "__main__":
  if len(sys.argv) >= 3:
    with open(sys.argv[1], "rb") as f:
      dat = f.read()
    compile(dat, sys.argv[2])
  else:
    dat = fetch(OPENPILOT_MODEL)
    compile(dat, "/tmp/output.thneed")
