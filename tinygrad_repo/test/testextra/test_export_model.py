import unittest
from extra.export_model import export_model, EXPORT_SUPPORTED_DEVICE
from tinygrad.tensor import Tensor, Device
from tinygrad import dtypes
import json

class MockMultiInputModel:
  def forward(self, x1, x2, x3):
    return x1 + x2 + x3

class MockMultiOutputModel:
  def __call__(self, x1):
    return x1 + 2.0, x1.pad(((0, 0), (0, 1))) + 1.0

# TODO: move compile_efficientnet tests here
@unittest.skipUnless(Device.DEFAULT in EXPORT_SUPPORTED_DEVICE, f"Model export is not supported on {Device.DEFAULT}")
class TextModelExport(unittest.TestCase):
  def test_multi_input_model_export(self):
    model = MockMultiInputModel()
    inputs = [Tensor.rand(2,2), Tensor.rand(2,2), Tensor.rand(2,2)]
    prg, inp_sizes, _, _ = export_model(model, "", *inputs)
    prg = json.loads(prg)

    assert len(inputs) == len(prg["inputs"]) == len(inp_sizes), f"Model and exported inputs don't match: mdl={len(inputs)}, prg={len(prg['inputs'])}, inp_sizes={len(inp_sizes)}"  # noqa: E501

    for i in range(len(inputs)):
      assert f"input{i}" in inp_sizes, f"input{i} not captured in inp_sizes"
      assert f"input{i}" in prg["buffers"], f"input{i} not captured in exported buffers"

    for i, exported_input in enumerate(prg["inputs"]):
      assert inputs[i].dtype.name == exported_input["dtype"], f"Model and exported input dtype don't match: mdl={inputs[i].dtype.name}, prg={exported_input['dtype']}"  # noqa: E501

  def test_multi_output_model_export(self):
    model = MockMultiOutputModel()
    input_tensor = Tensor.rand(2,2)
    outputs = model(input_tensor)
    prg, _, out_sizes, _ = export_model(model, "", input_tensor)
    prg = json.loads(prg)

    assert len(outputs) == len(prg["outputs"]) == len(out_sizes), f"Model and exported outputs don't match: mdl={len(outputs)}, prg={len(prg['outputs'])}, inp_sizes={len(out_sizes)}"  # noqa: E501

    for i in range(len(outputs)):
      assert f"output{i}" in out_sizes, f"output{i} not captured in out_sizes"
      assert f"output{i}" in prg["buffers"], f"output{i} not captured in exported buffers"

    for i, exported_output in enumerate(prg["outputs"]):
      assert outputs[i].dtype.name == exported_output["dtype"], f"Model and exported output dtype don't match: mdl={outputs[i].dtype.name}, prg={exported_output['dtype']}"  # noqa: E501

@unittest.skipUnless(Device.DEFAULT == "WEBGPU", "Testing WebGPU specific model export behavior")
class TextModelExportWebGPU(unittest.TestCase):
  def test_exported_input_output_dtypes(self):
    class MyModel:
      def forward(self, *inputs): return tuple([(inp+2).cast(inp.dtype) for inp in inputs])
    model = MyModel()
    # [:-1] because "ulong" and "long" is not supported
    inputs = [Tensor.randn(2, dtype=dt) for dt in dtypes.uints[:-1] + dtypes.sints[:-1] + (dtypes.bool, dtypes.float)]
    prg, _, _, _ = export_model(model, "webgpu", *inputs)
    expected_buffer_types = ["Uint"]*len(dtypes.uints[:-1]) + ["Int"]*len(dtypes.sints[:-1]) + ["Int", "Float"]
    for i, expected_buffer_type in enumerate(expected_buffer_types):
      dt = inputs[i].dtype
      expected_arr_prefix = f"{expected_buffer_type}{dt.itemsize*8}"
      # test input buffers
      self.assertIn(f"new {expected_arr_prefix}Array(gpuWriteBuffer{i}.getMappedRange()).set(_input{i});", prg)
      # test output buffers
      self.assertIn(f"const resultBuffer{i} = new {expected_arr_prefix}Array(gpuReadBuffer{i}.size/{dt.itemsize});", prg)
      self.assertIn(f"resultBuffer{i}.set(new {expected_arr_prefix}Array(gpuReadBuffer{i}.getMappedRange()));", prg)

if __name__ == '__main__':
  unittest.main()
