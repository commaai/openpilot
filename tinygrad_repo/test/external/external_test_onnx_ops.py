# inputs, attributes, and outputs for tests are found here:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md
# https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md

from typing import Any
import unittest, onnx, tempfile
import numpy as np
from extra.onnx_helpers import validate

class TestOnnxOps(unittest.TestCase):
  DOMAIN = None
  def helper_test_single_op(self, op:str, inps:dict[str, np.ndarray], opts:dict[str, Any], outs:list[str], rtol=1e-3, atol=1e-6):
    onnx_inputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
    onnx_outputs = [onnx.helper.make_empty_tensor_value_info(name) for name in outs]
    nodes = [onnx.helper.make_node(op, list(inps), list(outs), domain=self.DOMAIN, **opts)]
    graph = onnx.helper.make_graph(nodes, f"test_{op.lower()}", onnx_inputs, onnx_outputs)
    model = onnx.helper.make_model(graph, producer_name=f"test_{op.lower()}")
    with tempfile.NamedTemporaryFile() as tmp:
      onnx.save(model, tmp.name)
      validate(tmp.name, inps, rtol, atol)

class TestMainOnnxOps(TestOnnxOps):
  DOMAIN = ""
  def test_reshape(self):
    inputs = {"in": np.arange(6, dtype=np.float32), "shape": np.array([2,3], dtype=np.int64)}
    attributes = {}
    outputs = ["out"]
    self.helper_test_single_op("Reshape", inputs, attributes, outputs)

  def test_qlinear_conv(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      for b in (np.ones([32], dtype=np.int32), np.zeros([32], dtype=np.int32)):
        with self.subTest(dtype=dtype, zero_point=zero_point):
          dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
          inputs = {
            "x": np.random.randint(dtype_min, dtype_max + 1, [1, 3, 224, 224], dtype=dtype),
            "x_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
            "x_zero_point": np.array(zero_point, dtype=dtype),
            "w": np.random.randint(dtype_min, dtype_max + 1, [32, 3, 3, 3], dtype=dtype),
            "w_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
            "w_zero_point": np.array(zero_point, dtype=dtype),
            "y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
            "y_zero_point": np.array(zero_point, dtype=dtype),
            "b": b
          }
          attributes = {'auto_pad': 'NOTSET', 'dilations': (1, 1), 'group': 1, 'kernel_shape': (3, 3), 'pads': (1, 1, 1, 1), 'strides': (2, 2)}
          outputs = ["out"]
          self.helper_test_single_op("QLinearConv", inputs, attributes, outputs, atol=1)

  def test_qlinear_matmul(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        inputs = {
          "A": np.random.randint(dtype_min, dtype_max + 1, [10, 10], dtype=dtype),
          "A_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "A_zero_point": np.array(zero_point, dtype=dtype),
          "B": np.random.randint(dtype_min, dtype_max + 1, [10, 10], dtype=dtype),
          "B_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "B_zero_point": np.array(zero_point, dtype=dtype),
          "Y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "Y_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {}
        outputs = ["Y"]
        self.helper_test_single_op("QLinearMatMul", inputs, attributes, outputs, atol=1)

class TestContribOnnxOps(TestOnnxOps):
  DOMAIN = "com.microsoft"

  def test_attention(self):
    batch_size, seq_len, input_hidden_size = 2, 8, 256
    num_heads, head_size = 4, 64
    hidden_size = num_heads * head_size
    v_hidden_size = hidden_size

    # for mask_index
    right_padding_mask = np.random.randint(1, seq_len + 1, size=(batch_size,), dtype=np.int32)
    end_positions = np.random.randint(1, seq_len + 1, size=(batch_size,), dtype=np.int32)
    start_positions = np.array([np.random.randint(0, end) for end in end_positions], dtype=np.int32)
    left_padding_mask = np.concatenate([end_positions, start_positions])

    base_inps = {
      "input": np.random.randn(batch_size, seq_len, input_hidden_size).astype(np.float32),
      "weights": np.random.randn(input_hidden_size, hidden_size * 3).astype(np.float32),
      # bias is required in ORT (segfaults otherwise), eventhough docs says it's optional
      "bias": np.random.randn(hidden_size * 2 + v_hidden_size).astype(np.float32),
    }
    base_opts = {"num_heads": num_heads}

    test_cases = [
      ({}, {}),
      ({}, {"scale": 0.1}),
      ({}, {"scale": 1.0}),
      ({}, {"unidirectional": 1}),
      ({"mask_index": right_padding_mask}, {}),
      ({"mask_index": left_padding_mask}, {}),
      ({"mask_index": np.random.randint(0, seq_len, size=(batch_size, seq_len), dtype=np.int32)}, {"mask_filter_value": -5000.0}),
      ({"mask_index": np.random.randint(0, seq_len, size=(batch_size, seq_len, seq_len), dtype=np.int32)}, {"mask_filter_value": -np.inf}),
      # BUG: when `mask_index` is used with `unidirectional`, the first value must be True
      # otherwise this will trigger a different ORT behavior where start consecutive Falses will be turned True
      # e.g. mask_index = [[0, 0, 1, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1, 1]]
      # will need mask[:, :, 0:1, 0:1] = True
      ({"mask_index": np.array([[1, 0, 1, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1, 1, 1]], dtype=np.int32)}, {"unidirectional": 1}),
      ({ "weights": np.random.randn(input_hidden_size, hidden_size + hidden_size + 128).astype(np.float32),
         "bias": np.random.randn(hidden_size + hidden_size + 128).astype(np.float32)},
       {"qkv_hidden_sizes": [hidden_size, hidden_size, 128]}),
      # TODO: past is not tested. ORT gives type error for input
    ]

    for i, (extra_inps, extra_opts) in enumerate(test_cases):
      with self.subTest(f"test_attention_{i}"):
        inps = {**base_inps, **extra_inps}
        opts = {**base_opts, **extra_opts}
        outputs = ["output", "present"] if "past" in inps else ["output"]
        self.helper_test_single_op("Attention", inps, opts, outputs, atol=1e-4)

  def test_skip_layer_normalization(self):
    shape = (2, 8, 32)
    for has_beta in [True, False]:
      for has_bias in [True, False]:
        with self.subTest(has_beta=has_beta, has_bias=has_bias):
          hidden_size = shape[-1]
          inputs = {
            "input": np.random.randn(*shape).astype(np.float32),
            "skip": np.random.randn(*shape).astype(np.float32),
            "gamma": np.random.randn(hidden_size).astype(np.float32),
          }
          if has_beta: inputs["beta"] = np.random.randn(hidden_size).astype(np.float32)
          if has_bias: inputs["bias"] = np.random.randn(hidden_size).astype(np.float32)
          attributes = {"epsilon": 1e-12}
          outputs = ["output", "mean", "inv_std_var", "input_skip_bias_sum"]
          self.helper_test_single_op("SkipLayerNormalization", inputs, attributes, outputs)

  def test_bias_gelu(self):
    shape = (2,3,4)
    inputs = {
      "A": np.random.randn(*shape).astype(np.float32),
      "B": np.random.randn(shape[-1]).astype(np.float32)
    }
    attributes = {}
    outputs = ["C"]
    self.helper_test_single_op("BiasGelu", inputs, attributes, outputs)

  def test_qlinear_add(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        inputs = {
          "A": np.random.randint(dtype_min, dtype_max + 1, [10, 10], dtype=dtype),
          "A_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "A_zero_point": np.array(zero_point, dtype=dtype),
          "B": np.random.randint(dtype_min, dtype_max + 1, [10, 10], dtype=dtype),
          "B_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "B_zero_point": np.array(zero_point, dtype=dtype),
          "C_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "C_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {}
        outputs = ["C"]
        self.helper_test_single_op("QLinearAdd", inputs, attributes, outputs, atol=1)

  def test_qlinear_global_average_pool(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        inputs = {
          "X": np.random.randint(dtype_min, dtype_max + 1, [1, 3, 32, 32], dtype=dtype),
          "x_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "x_zero_point": np.array(zero_point, dtype=dtype),
          "y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "y_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {"channels_last": 0}
        outputs = ["C"]
        self.helper_test_single_op("QLinearGlobalAveragePool", inputs, attributes, outputs, atol=1)

if __name__ == "__main__":
  unittest.main()