from tinygrad import Tensor
from tinygrad.tensor import _to_np_dtype
from extra.onnx import OnnxRunner, OnnxValue
import onnx
import numpy as np
import onnxruntime as ort

def get_example_inputs(graph_inputs:dict[str, OnnxValue]):
  ret: dict[str, Tensor] = {}
  for name, spec in graph_inputs.items():
    assert not spec.is_optional and not spec.is_sequence, "only allow tensor input for now"
    shape = tuple(dim if isinstance(dim, int) else 1 for dim in spec.shape)
    value = Tensor(np.random.uniform(size=shape).astype(_to_np_dtype(spec.dtype)) * 8).realize()
    ret.update({name:value})
  return ret

def validate(onnx_file, inputs, rtol=1e-5, atol=1e-5):
  run_onnx = OnnxRunner(onnx.load(onnx_file))
  tinygrad_out = run_onnx(inputs)

  ort_options = ort.SessionOptions()
  ort_options.log_severity_level = 3
  ort_sess = ort.InferenceSession(onnx_file, ort_options, ["CPUExecutionProvider"])
  np_inputs = {k:v.numpy() if isinstance(v, Tensor) else v for k,v in inputs.items()}
  out_names = list(run_onnx.graph_outputs)
  out_values = ort_sess.run(out_names, np_inputs)
  ort_out = dict(zip(out_names, out_values))

  assert len(tinygrad_out) == len(ort_out) and tinygrad_out.keys() == ort_out.keys()
  for k in tinygrad_out.keys():
    tiny_v, onnx_v = tinygrad_out[k], ort_out[k]
    if tiny_v is None: assert tiny_v == onnx_v
    else: np.testing.assert_allclose(tiny_v.numpy(), onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tinygrad_out.keys()}")