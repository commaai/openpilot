import unittest, onnx, tempfile, pathlib
import numpy as np
from tinygrad import Tensor
from tinygrad.uop.ops import Ops
from typing import Any
from tinygrad.nn.onnx import OnnxRunner, OnnxPBParser, OnnxDataType
from hypothesis import given, strategies as st

# copied from test_const_folding.py
def _check_ast_count(desired_count:int, t:Tensor):
  # NOTE: this has side effect because everything can be scheduled only once
  linear = t.schedule_linear()
  asts = [call for call in linear.src if call.src[0].op is Ops.SINK]
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

  @unittest.skip("const folding is removed")
  def test_const_fold_from_disk(self):
    self._test_const_fold_unary_op(True)
    self._test_const_fold_binary_op(True)

  @unittest.skip("const folding is removed")
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

all_dtypes = list(OnnxDataType)

class TestOnnxRunnerDtypes(unittest.TestCase):
  """
  Internal tensors (initializers, attributes) fallback to default dtype if unsupported by device.
  External tensors (inputs) preserve their original dtype - user must ensure compatibility with device.
  """
  def _get_expected_dtype(self, onnx_dtype: int, is_input: bool): return OnnxDataType(onnx_dtype).to_dtype()

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

# from openpilot selfdrive/modeld/get_model_metadata.py
class MetadataOnnxPBParser(OnnxPBParser):
  def _parse_ModelProto(self) -> dict:
    obj: dict[str, Any] = {"graph": {"input": [], "output": []}, "metadata_props": []}
    for fid, wire_type in self._parse_message(self.reader.len):
      match fid:
        case 7: obj["graph"] = self._parse_GraphProto()
        case 14: obj["metadata_props"].append(self._parse_StringStringEntryProto())
        case _: self.reader.skip_field(wire_type)
    return obj

class TestOnnxMetadata(unittest.TestCase):
  def test_metadata_props(self):
    graph = onnx.helper.make_graph(
      nodes=[onnx.helper.make_node('Identity', ['input'], ['output'])],
      name='test',
      inputs=[onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, (1, 3))],
      outputs=[onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, (1, 3))],
    )
    model = onnx.helper.make_model(graph)
    model.metadata_props.append(onnx.StringStringEntryProto(key="model_checkpoint", value="v1.0"))
    model.metadata_props.append(onnx.StringStringEntryProto(key="output_slices", value="dGVzdA=="))

    with tempfile.TemporaryDirectory() as tmpdir:
      model_path = pathlib.Path(tmpdir) / "model.onnx"
      onnx.save(model, model_path)
      parsed = MetadataOnnxPBParser(model_path).parse()

    # metadata_props should be accessible as dicts with "key" and "value"
    self.assertEqual(len(parsed["metadata_props"]), 2)
    self.assertEqual(parsed["metadata_props"][0]["key"], "model_checkpoint")
    self.assertEqual(parsed["metadata_props"][0]["value"], "v1.0")
    self.assertEqual(parsed["metadata_props"][1]["key"], "output_slices")
    self.assertEqual(parsed["metadata_props"][1]["value"], "dGVzdA==")

if __name__ == '__main__':
  unittest.main()