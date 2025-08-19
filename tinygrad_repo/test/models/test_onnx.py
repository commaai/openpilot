#!/usr/bin/env python
import os
import time
import unittest
import numpy as np
try:
  import onnx
except ModuleNotFoundError:
  raise unittest.SkipTest("onnx not installed, skipping onnx test")
from tinygrad.frontend.onnx import OnnxRunner
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.helpers import CI, fetch, temp, Context

try:
  from extra.onnx_helpers import validate
  from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry
  HUGGINGFACE_AVAILABLE = True
except ModuleNotFoundError:
  HUGGINGFACE_AVAILABLE = False

def run_onnx_torch(onnx_model, inputs):
  import torch
  from onnx2torch import convert
  torch_model = convert(onnx_model).float()
  with torch.no_grad():
    torch_out = torch_model(*[torch.tensor(x) for x in inputs.values()])
  return torch_out

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

np.random.seed(1337)

class TestOnnxModel(unittest.TestCase):
  def test_benchmark_openpilot_model(self):
    onnx_model = fetch(OPENPILOT_MODEL)
    run_onnx = OnnxRunner(onnx_model)
    def get_inputs():
      np_inputs = {
        "input_imgs": np.random.randn(*(1, 12, 128, 256)),
        "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
        "desire": np.zeros((1, 100, 8)),
        "traffic_convention": np.array([[1., 0.]]),
        "nav_features": np.zeros((1, 256)),
        "features_buffer": np.zeros((1, 99, 128)),
    }
      inputs = {k:Tensor(v.astype(np.float32), requires_grad=False) for k,v in np_inputs.items()}
      return inputs

    for _ in range(7):
      inputs = get_inputs()
      st = time.monotonic()
      tinygrad_out = run_onnx(inputs)['outputs']
      mt = time.monotonic()
      tinygrad_out.realize()
      mt2 = time.monotonic()
      tinygrad_out = tinygrad_out.numpy()
      et = time.monotonic()
      if not CI:
        print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(mt2-mt)*1000.0:.2f} ms for realize, {(et-mt2)*1000.0:.2f} ms for GPU queue")

    if not CI:
      import cProfile
      import pstats
      inputs = get_inputs()
      pr = cProfile.Profile(timer=time.perf_counter_ns, timeunit=1e-6)
      pr.enable()
    tinygrad_out = run_onnx(inputs)['outputs']
    tinygrad_out.realize()
    tinygrad_out = tinygrad_out.numpy()
    if not CI:
      pr.disable()
      stats = pstats.Stats(pr)
      stats.dump_stats(temp("net.prof"))
      os.system(f"flameprof {temp('net.prof')} > {temp('prof.svg')}")
      ps = stats.sort_stats(pstats.SortKey.TIME)
      ps.print_stats(30)

  def test_openpilot_model(self):
    onnx_model = fetch(OPENPILOT_MODEL)
    run_onnx = OnnxRunner(onnx_model)
    print("got run_onnx")
    inputs = {
      "input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "desire": np.zeros((1, 100, 8)),
      "traffic_convention": np.array([[1., 0.]]),
      "nav_features": np.zeros((1, 256)),
      "features_buffer": np.zeros((1, 99, 128)),
    }
    inputs = {k:v.astype(np.float32) for k,v in inputs.items()}

    st = time.monotonic()
    print("****** run onnx ******")
    tinygrad_out = run_onnx(inputs)['outputs']
    mt = time.monotonic()
    print("****** realize ******")
    tinygrad_out.realize()
    mt2 = time.monotonic()
    tinygrad_out = tinygrad_out.numpy()
    et = time.monotonic()
    print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(mt2-mt)*1000.0:.2f} ms for realize, {(et-mt2)*1000.0:.2f} ms for GPU queue")

    onnx_model = onnx.load(fetch(OPENPILOT_MODEL))
    torch_out = run_onnx_torch(onnx_model, inputs).numpy()
    print(tinygrad_out, torch_out)
    np.testing.assert_allclose(tinygrad_out, torch_out, atol=1e-4, rtol=1e-2)

  @unittest.skip("slow")
  def test_efficientnet(self):
    input_name, input_new = "images:0", True
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"),
      input_name, input_new)

  @unittest.skip("TODO: FIX THIS IT CAUSES SEGFAULT")
  def test_shufflenet(self):
    input_name, input_new = "gpu_0/data_0", False
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx"),
      input_name, input_new)

  @unittest.skip("test is very slow")
  def test_resnet(self):
    # NOTE: many onnx models can't be run right now due to max pool with strides != kernel_size
    input_name, input_new = "data", False
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx"),
      input_name, input_new)

  def _test_model(self, fn, input_name, input_new, debug=False):
    run_onnx = OnnxRunner(fn)
    print("onnx loaded")
    from test.models.test_efficientnet import chicken_img, car_img, preprocess, _LABELS

    def run(img):
      inputs = {input_name: preprocess(img, new=input_new)}
      tinygrad_out = list(run_onnx(inputs, debug=debug).values())[0].numpy()
      return tinygrad_out.argmax()

    cls = run(chicken_img)
    print(cls, _LABELS[cls])
    assert _LABELS[cls] == "hen" or _LABELS[cls] == "cock"
    cls = run(car_img)
    print(cls, _LABELS[cls])
    assert "car" in _LABELS[cls] or _LABELS[cls] == "convertible"

@unittest.skipUnless(HUGGINGFACE_AVAILABLE and Device.DEFAULT == "METAL", "only run on METAL")
class TestHuggingFaceOnnxModels(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls._ctx = Context(MAX_BUFFER_SIZE=0)
    cls._ctx.__enter__()

  @classmethod
  def tearDownClass(cls):
    cls._ctx.__exit__()

  def _validate(self, repo_id, model_file, custom_inputs, rtol=1e-4, atol=1e-4):
    onnx_model_path = snapshot_download_with_retry(
      repo_id=repo_id,
      allow_patterns=["*.onnx", "*.onnx_data"],
      cache_dir=str(DOWNLOADS_DIR)
    )
    onnx_model_path = onnx_model_path / model_file
    file_size = onnx_model_path.stat().st_size
    print(f"Validating model: {repo_id}/{model_file} ({file_size/1e6:.2f}M)")
    validate(onnx_model_path, custom_inputs, rtol=rtol, atol=atol)

  def test_xlm_roberta_large(self):
    repo_id = "FacebookAI/xlm-roberta-large"
    model_file = "onnx/model.onnx"
    custom_inputs = {
      "input_ids": np.random.randint(0, 250002, (1, 11), dtype=np.int64),
      "attention_mask": np.ones((1, 11), dtype=np.int64),
    }
    self._validate(repo_id, model_file, custom_inputs)

if __name__ == "__main__":
  unittest.main()
