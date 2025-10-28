#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.frontend.onnx import OnnxRunner
from tinygrad.device import Device
from tinygrad.helpers import fetch, Context

from extra.onnx_helpers import validate
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry

def run_onnx_torch(onnx_model, inputs):
  import torch
  from onnx2torch import convert
  torch_model = convert(onnx_model).float()
  with torch.no_grad():
    torch_out = torch_model(*[torch.tensor(x) for x in inputs.values()])
  return torch_out

np.random.seed(1337)

class TestOnnxModel(unittest.TestCase):
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

@unittest.skipUnless(Device.DEFAULT == "METAL", "only run on METAL")
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
