import unittest, onnx, tempfile
from tinygrad import dtypes
from tinygrad.frontend.onnx import OnnxRunner, onnx_load
from tinygrad.device import is_dtype_supported
from extra.onnx import data_types
from hypothesis import given, settings, strategies as st
import numpy as np

data_types.pop(16) # TODO: this is bf16, need to support double parsing first.
device_supported_dtypes = [odt for odt, dtype in data_types.items() if is_dtype_supported(dtype)]
device_unsupported_dtypes = [odt for odt, dtype in data_types.items() if not is_dtype_supported(dtype)]

class TestOnnxRunnerDtypes(unittest.TestCase):
  def _test_input_spec_dtype(self, onnx_data_type, tinygrad_dtype):
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx_data_type, ())
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, ())
    node = onnx.helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile(suffix='.onnx')
    onnx.save(model, tmp.name)
    tmp.flush()
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(runner.graph_inputs['input'].dtype, tinygrad_dtype)

  def _test_initializer_dtype(self, onnx_data_type, tinygrad_dtype):
    arr = np.array([0, 1], dtype=onnx.helper.tensor_dtype_to_np_dtype(onnx_data_type))
    initializer = onnx.helper.make_tensor('initializer', onnx_data_type, arr.shape, arr.tobytes(), raw=True)
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx_data_type, ())
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, ())
    node = onnx.helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor], [initializer])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile(suffix='.onnx')
    onnx.save(model, tmp.name)
    tmp.flush()
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(runner.graph_values['initializer'].dtype, tinygrad_dtype)

  def _test_node_attribute_dtype(self, onnx_data_type, tinygrad_dtype):
    arr = np.array([0, 1], dtype=onnx.helper.tensor_dtype_to_np_dtype(onnx_data_type))
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, arr.shape)
    value_tensor = onnx.helper.make_tensor('value', onnx_data_type, arr.shape, arr.tobytes(), raw=True)
    node = onnx.helper.make_node('Constant', inputs=[], outputs=['output'], value=value_tensor)
    graph = onnx.helper.make_graph([node], 'attribute_test', [], [output_tensor])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile(suffix='.onnx')
    tmp.flush()
    onnx.save(model, tmp.name)
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(runner.graph_nodes[0].opts['value'].dtype, tinygrad_dtype)

  @settings(deadline=1000) # TODO investigate unreliable timing
  @given(onnx_data_type=st.sampled_from(device_supported_dtypes))
  def test_supported_dtype_spec(self, onnx_data_type):
    tinygrad_dtype = data_types[onnx_data_type]
    self._test_input_spec_dtype(onnx_data_type, tinygrad_dtype)
    self._test_initializer_dtype(onnx_data_type, tinygrad_dtype)
    self._test_node_attribute_dtype(onnx_data_type, tinygrad_dtype)

  @unittest.skipUnless(device_unsupported_dtypes, "No unsupported dtypes for this device to test.")
  @settings(deadline=1000) # TODO investigate unreliable timing
  @given(onnx_data_type=st.sampled_from(device_unsupported_dtypes))
  def test_unsupported_dtype_spec(self, onnx_data_type):
    true_dtype = data_types[onnx_data_type]
    default_dtype = dtypes.default_int if dtypes.is_int(true_dtype) else dtypes.default_float
    self._test_input_spec_dtype(onnx_data_type, true_dtype)
    self._test_initializer_dtype(onnx_data_type, default_dtype)
    self._test_node_attribute_dtype(onnx_data_type, default_dtype)

if __name__ == '__main__':
  unittest.main()