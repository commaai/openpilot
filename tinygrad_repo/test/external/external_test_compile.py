import tempfile, unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from examples.openpilot.load_pickle import make_inputs
from tinygrad.helpers import Context, DEV, fetch, getenv
from tinygrad.engine.realize import lower_and_compile
from tinygrad.uop.ops import Ops
from examples.openpilot.helpers import load_pickle, make_retargetable


class TestCompile(unittest.TestCase):
  def test_metadata(self):
    import onnx
    from examples.openpilot.compile_onnx import onnx_metadata
    graph = onnx.helper.make_graph([onnx.helper.make_node('Identity', ['input'], ['output'])], 'metadata',
      [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, (1, 3))],
      [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, (1, 3))])
    model = onnx.helper.make_model(graph)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'metadata.onnx'
      for metadata in ({}, {'model_checkpoint': 'v1.0', 'output_slices': 'dGVzdA=='}):
        onnx.helper.set_model_props(model, metadata)
        onnx.save(model, path)
        properties, output_shapes = onnx_metadata(path)
        self.assertEqual(properties, metadata)
        self.assertEqual(output_shapes, {'output': (1, 3)})

  def test_warp_layouts(self):
    from tinygrad import Tensor
    from examples.openpilot.compile_warp import NV12Frame, make_luma_warp, make_frame_prepare
    frame = NV12Frame(8, 6, 12, 8, 4, 144)
    data = np.zeros((12, 12), dtype=np.uint8)
    data[:6, :8] = np.arange(48, dtype=np.uint8).reshape(6, 8)
    data[8:, 0:8:2], data[8:, 1:8:2] = 100, 200
    image = Tensor(data.reshape(-1)).realize()
    transform = np.eye(3, dtype=np.float32)
    luma = make_luma_warp(frame, 4, 4, border_fill=16)
    np.testing.assert_array_equal(luma(input_frame=image, M_inv=Tensor(transform, device='NPY')).numpy(), data[:4, :4].reshape(1, 16))
    transform[0, 2] = 1000
    np.testing.assert_array_equal(luma(input_frame=image, M_inv=Tensor(transform, device='NPY')).numpy(), np.full((1, 16), 16, dtype=np.uint8))
    yuv = make_frame_prepare(frame, 4, 4)
    expected = [[[0, 2], [16, 18]], [[8, 10], [24, 26]], [[1, 3], [17, 19]], [[9, 11], [25, 27]], [[100]*2]*2, [[200]*2]*2]
    np.testing.assert_array_equal(yuv(input_frame=image, M_inv=Tensor(np.eye(3, dtype=np.float32), device='NPY')).numpy(), expected)


@unittest.skipUnless(getenv("MODEL_PKL", ""), "requires an artifact from python examples/openpilot/compile_onnx.py")
class TestCompiledModel(unittest.TestCase):
  def setUp(self):
    self.artifact = load_pickle(getenv("MODEL_PKL", ""), out_of_band=bool(getenv("PICKLE_OOB")))
    self.model = self.artifact['run']

  def test_retarget(self):
    from tinygrad.runtime.autogen.libc import Elf64_Ehdr, EM_AARCH64
    make_retargetable(self.model)
    with Context(DEV=replace(DEV.target("CPU"), arch="arm64,generic")): linear = lower_and_compile(self.model.captured._linear)
    for p in linear.toposort(enter_calls=True):
      if p.op is Ops.PROGRAM and p.arg.target.device == "CPU":
        self.assertEqual(Elf64_Ehdr.from_buffer_copy(p.src[-1].arg).e_machine, EM_AARCH64)

  def test_inputs(self):
    def run(seed):
      inputs = make_inputs(self.artifact, seed)
      self.model(**inputs)
      return [t.numpy().copy() for t in inputs['output_buffers'].values()]
    original, changed, repeated = run(100), run(101), run(100)
    for before, after, again in zip(original, changed, repeated, strict=True):
      self.assertTrue(np.isfinite(before).all() and np.isfinite(after).all())
      np.testing.assert_array_equal(before, again)
    self.assertFalse(all(np.array_equal(a, b) for a, b in zip(original, changed, strict=True)))

  @unittest.skipUnless(getenv("MODEL_ONNX", ""), "requires the source ONNX model")
  def test_onnx(self):
    import onnxruntime as ort
    session = ort.InferenceSession(str(fetch(getenv("MODEL_ONNX", ""))), providers=['CPUExecutionProvider'])
    input_types = {v.name: v.type.removeprefix('tensor(').removesuffix(')') for v in session.get_inputs()}
    for seed in (100, 101):
      inputs = make_inputs(self.artifact, seed)
      reference = session.run(None, {k: v.numpy().astype({'float': 'float32', 'double': 'float64'}.get(input_types[k], input_types[k]))
                                    for k, v in inputs.items() if k in input_types})
      self.model(**inputs)
      for expected, actual in zip(reference, inputs['output_buffers'].values(), strict=True):
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-4, rtol=1e-4)


if __name__ == '__main__': unittest.main()
