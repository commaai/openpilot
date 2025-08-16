import unittest, onnx, tempfile, pathlib
import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.uop.ops import Ops
from tinygrad.device import is_dtype_supported
from extra.onnx import data_types
from tinygrad.frontend.onnx import OnnxRunner
from hypothesis import given, strategies as st

# copied from test_const_folding.py
def _check_ast_count(desired_count:int, t:Tensor):
  # NOTE: this has side effect because everything can be scheduled only once
  schedule = t.schedule()
  asts = [s for s in schedule if s.ast.op is Ops.SINK]
  assert len(asts) == desired_count, f"{len(asts)} != {desired_count}"

def build_onnx(nodes, from_disk:bool=True, **kwargs):
  """Helper to build and return an OnnxRunner from ONNX nodes."""
  graph = onnx.helper.make_graph(nodes, 'test', kwargs.get('inputs', []), kwargs.get('outputs', []), kwargs.get('initializers', []))
  model = onnx.helper.make_model(graph)
  if from_disk:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      model_path = tmp_path / "model.onnx"
      onnx.save(model, model_path)
      runner = OnnxRunner(model_path)
  else:
    # use the in-memory method
    runner = OnnxRunner(Tensor(model.SerializeToString(), device="PYTHON"))
  return runner

class TestOnnxRunner(unittest.TestCase):
  def _test_const_fold_unary_op(self, from_disk:bool):
    runner = build_onnx(
        nodes=[
          onnx.helper.make_node('Expand', ['inp', 'shape'], ['expanded']),
          onnx.helper.make_node('Exp', ['expanded'], ['output'])
        ],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, (5,))],
        initializers=[
          onnx.helper.make_tensor('inp', onnx.TensorProto.FLOAT, (), [1.0]),
          onnx.helper.make_tensor('shape', onnx.TensorProto.INT64, (1,), [5])
        ],
        from_disk=from_disk)
    output = runner({'inp': Tensor([1.0])})['output']
    _check_ast_count(0, output)

  def _test_const_fold_binary_op(self, from_disk:bool):
    runner = build_onnx(
        nodes=[onnx.helper.make_node('Add', ['inp', 'const'], ['output'])],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, (4,))],
        initializers=[
          onnx.helper.make_tensor('inp', onnx.TensorProto.FLOAT, (4,), [1, 2, 3, 4]),
          onnx.helper.make_tensor('const', onnx.TensorProto.FLOAT, (), [0])
        ],
        from_disk=from_disk)
    output = runner({'inp': Tensor([1, 2, 3, 4])})['output']
    _check_ast_count(0, output)

  def test_const_fold_from_disk(self):
    self._test_const_fold_unary_op(True)
    self._test_const_fold_binary_op(True)

  def test_const_fold_from_memory(self):
    self._test_const_fold_unary_op(False)
    # TODO: understand this and fix this, bitcast related
    # self._test_const_fold_binary_op(False)

  def test_external_data_loading(self):
    weights = np.arange(4, dtype=np.float32)
    tensor_with_data = onnx.helper.make_tensor('weights', onnx.TensorProto.FLOAT, weights.shape, weights.tobytes(), raw=True)
    graph = onnx.helper.make_graph(
        nodes=[onnx.helper.make_node('Add', ['inp', 'weights'], ['output'])],
        name='test_external',
        inputs=[onnx.helper.make_tensor_value_info('inp', onnx.TensorProto.FLOAT, (1,))],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, weights.shape)],
        initializer=[tensor_with_data]
    )
    model = onnx.helper.make_model(graph)

    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      model_path = tmp_path / "model.onnx"
      onnx.save_model(model, model_path, save_as_external_data=True, all_tensors_to_one_file=True, size_threshold=0, location="weights.onnx_data")
      runner = OnnxRunner(model_path)
      output = runner({'inp': Tensor([1])})['output']
      np.testing.assert_equal(output.numpy(), weights + 1)

all_dtypes = list(data_types.keys())
device_supported_dtypes = {odt for odt, dtype in data_types.items() if is_dtype_supported(dtype)}

class TestOnnxRunnerDtypes(unittest.TestCase):
  """
  Internal tensors (initializers, attributes) fallback to default dtype if unsupported by device.
  External tensors (inputs) preserve their original dtype - user must ensure compatibility with device.
  """
  def _get_expected_dtype(self, onnx_dtype: int, is_input: bool):
    true_dtype = data_types[onnx_dtype]
    # inputs always preserve their true dtype.
    if is_input:
      return true_dtype
    # supported types are always themselves.
    if onnx_dtype in device_supported_dtypes:
      return true_dtype
    # otherwise it's an unsupported dtype that's internal to the ONNX model, which should fallback to default.
    return dtypes.default_int if dtypes.is_int(true_dtype) else dtypes.default_float

  @given(onnx_dtype=st.sampled_from(all_dtypes))
  def test_input_dtype(self, onnx_dtype: int):
    expected_dtype = self._get_expected_dtype(onnx_dtype, True)
    runner = build_onnx(
        nodes=[onnx.helper.make_node('Identity', ['input'], ['output'])],
        inputs=[onnx.helper.make_tensor_value_info('input', onnx_dtype, ())],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx_dtype, ())],
        from_disk=False)
    self.assertEqual(runner.graph_inputs['input'].dtype, expected_dtype)

  @given(onnx_dtype=st.sampled_from(all_dtypes))
  def test_initializer_dtype(self, onnx_dtype: int):
    expected_dtype = self._get_expected_dtype(onnx_dtype, False)
    runner = build_onnx(
        nodes=[onnx.helper.make_node('Identity', ['initializer'], ['output'])],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx_dtype, (2,))],
        initializers=[onnx.helper.make_tensor('initializer', onnx_dtype, (2,), [1, 2])],
        from_disk=False)
    self.assertEqual(runner.graph_values['initializer'].dtype, expected_dtype)

  @given(onnx_dtype=st.sampled_from(all_dtypes))
  def test_node_attribute_dtype(self, onnx_dtype: int):
    expected_dtype = self._get_expected_dtype(onnx_dtype, False)
    value_tensor = onnx.helper.make_tensor('value', onnx_dtype, (2,), [1, 2])
    runner = build_onnx(
        nodes=[onnx.helper.make_node('Constant', [], ['output'], value=value_tensor)],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx_dtype, (2,))],
        from_disk=False)
    self.assertEqual(runner.graph_nodes[0].opts['value'].dtype, expected_dtype)

if __name__ == '__main__':
  unittest.main()