#!/usr/bin/env python3
import os
import pickle
import re
import sys
import time
from dataclasses import replace

import numpy as np

if "JIT_BATCH_SIZE" not in os.environ:
  os.environ["JIT_BATCH_SIZE"] = "0"

from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import DEBUG, getenv
from tinygrad.nn.onnx import OnnxNode, OnnxPBParser, OnnxRunner

OPENPILOT_MODEL = sys.argv[1]
OUTPUT = sys.argv[2]


class DMonitoringOnnxRunner(OnnxRunner):
  def __init__(self, model_path):
    model = OnnxPBParser(model_path, load_external_data=True).parse()
    if not getenv("DMONITORING_KEEP_FEATURES"):
      self._drop_unused_features(model["graph"])
    self._init_from_graph(model["graph"])

  @staticmethod
  def _drop_unused_features(graph):
    for node in graph["node"]:
      parsed_node: OnnxNode = node["parsed_node"]
      if parsed_node.op == "Concat" and parsed_node.outputs == ("outputs",):
        assert parsed_node.inputs[-1] == "linear", parsed_node
        node["input"] = list(parsed_node.inputs[:-1])
        node["parsed_node"] = replace(parsed_node, inputs=parsed_node.inputs[:-1])
        return
    raise RuntimeError("failed to find dmonitoring output concat")


def compile(onnx_file):
  run_onnx = DMonitoringOnnxRunner(onnx_file)
  print("loaded model")

  input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
  input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}

  input_types = {k: (dtypes.float32 if v is dtypes.float16 else v) for k, v in input_types.items()}
  Tensor.manual_seed(100)
  inputs = {k: Tensor(Tensor.randn(*shp, dtype=input_types[k]).mul(8).realize().numpy(), device='NPY')
            for k, shp in sorted(input_shapes.items())}
  if not getenv("NPY_IMG"):
    inputs = {k: Tensor(v.numpy(), device=Device.DEFAULT).realize() if 'img' in k else v for k, v in inputs.items()}
  print("created tensors")

  run_onnx_jit = TinyJit(lambda **kwargs:
                         next(iter(run_onnx({k: v.to(Device.DEFAULT) for k, v in kwargs.items()}).values())).cast('float32'),
                         prune=True)
  for i in range(3):
    GlobalCounters.reset()
    print(f"run {i}")
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1), OPENPILOT_HACKS=1):
      ret = run_onnx_jit(**inputs).numpy()
    if i == 1:
      test_val = np.copy(ret)
  print(f"captured {len(run_onnx_jit.captured.jit_cache)} kernels")
  np.testing.assert_equal(test_val, ret, "JIT run failed")
  print("jit run validated")

  kernel_count = 0
  read_image_count = 0
  gated_read_image_count = 0
  for ei in run_onnx_jit.captured.jit_cache:
    src = getattr(getattr(getattr(ei, "prg", None), "p", None), "src", "")
    if not src:
      continue
    kernel_count += 1
    read_image_count += src.count("read_image")
    gated_read_image_count += src.count("?read_image")
    for v in [m.group(1) for m in re.finditer(r'(val\d+)\s*=\s*read_imagef\(', src)]:
      if len(re.findall(fr'[\?\:]{v}\.[xyzw]', src)) > 0:
        gated_read_image_count += 1
  print(f"{kernel_count=},  {read_image_count=}, {gated_read_image_count=}")

  with open(OUTPUT, "wb") as f:
    pickle.dump(run_onnx_jit, f)
  mdl_sz = os.path.getsize(onnx_file)
  pkl_sz = os.path.getsize(OUTPUT)
  print(f"mdl size is {mdl_sz / 1e6:.2f}M")
  print(f"pkl size is {pkl_sz / 1e6:.2f}M")
  print("**** compile done ****")
  return inputs, test_val


def test_vs_compile(run, inputs, test_val=None):
  step_times = []
  for _ in range(20):
    st = time.perf_counter()
    out = run(**inputs)
    mt = time.perf_counter()
    val = out.numpy()
    et = time.perf_counter()
    step_times.append((et - st) * 1e3)
    print(f"enqueue {(mt - st) * 1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")

  if (assert_time := getenv("ASSERT_MIN_STEP_TIME")):
    min_time = min(step_times)
    assert min_time < assert_time, f"Speed regression, expected min step time of < {assert_time} ms but took: {min_time} ms"

  if test_val is not None:
    np.testing.assert_equal(test_val, val)
  print("**** test done ****")

  if os.environ.get("DEV", "") != "CL":
    inputs_2x = {k: Tensor(v.numpy() * 2, device=v.device) for k, v in inputs.items()}
    out = run(**inputs_2x)
    changed_val = out.numpy()
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, val, changed_val)
  return val


if __name__ == "__main__":
  inputs, outputs = compile(OPENPILOT_MODEL)

  with open(OUTPUT, "rb") as f:
    pickle_loaded = pickle.load(f)

  test_vs_compile(pickle_loaded, inputs, outputs)
