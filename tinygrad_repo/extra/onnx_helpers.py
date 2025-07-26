from tinygrad import Tensor
from tinygrad.tensor import _to_np_dtype
from tinygrad.frontend.onnx import OnnxRunner
from extra.onnx import OnnxValue
import numpy as np
import onnxruntime as ort

def get_example_inputs(graph_inputs:dict[str, OnnxValue], config={}):
  def _get_shape(onnx_shape: tuple[str|int]):
    shape = []
    for onnx_dim in onnx_shape:
      match onnx_dim:
        case int(): shape.append(onnx_dim)
        case "width" | "height":
          size = config.get("size", {})
          shape.append(size) if isinstance(size, int) else shape.append(size.get(onnx_dim, 224))
        case "sequence" | "sequence_length" | "decoder_sequence_length": shape.append(64)
        case "encoder_sequence_length": shape.append(config.get("nb_max_frames", 64))
        case "past_decoder_sequence_length" | "encoder_sequence_length_out": shape.append(64)
        case "encoder_sequence_length / 2": shape.append(32)
        case "batch_size": shape.append(1)
        case "num_channels": shape.append(config.get("in_channels", 3))
        case "num_channels_latent": shape.append(config.get("latent_channels", 4))
        case "height_latent" | "width_latent": shape.append(config.get("sample_size", 1024) // 8)
        case "feature_size": shape.append(config.get("num_mel_bins", 128))
        case _: shape.append(1)
    return shape
  def _get_value(name, shape, dtype):
    match name:
      case "input_ids":
        vocab_size = config.get("text_config", {}).get("vocab_size") or config.get("vocab_size", 32)
        val = np.random.randint(0, vocab_size-1, shape)
      case "attention_mask": val = np.random.randint(0, 2, size=shape)
      case "token_type_ids": val = np.random.randint(0, config.get("type_vocab_size", 2), shape)
      case "image_tensor": val = np.random.randint(0, 256, shape)
      case "task_id": return Tensor(0, dtype=dtype)
      case _: val = np.random.uniform(size=shape) * 8
    return Tensor(val.astype(_to_np_dtype(dtype))).realize()

  ret: dict[str, Tensor] = {}
  for name, spec in graph_inputs.items():
    assert not spec.is_optional and not spec.is_sequence, "only allow tensor input for now"
    shape = _get_shape(spec.shape)
    value = _get_value(name, shape, spec.dtype)
    ret.update({name:value})
  return ret

def validate(onnx_file, inputs, rtol=1e-5, atol=1e-5):
  run_onnx = OnnxRunner(onnx_file)

  ort_options = ort.SessionOptions()
  ort_options.log_severity_level = 3
  ort_sess = ort.InferenceSession(onnx_file, ort_options, ["CPUExecutionProvider"])
  np_inputs = {k:v.numpy() if isinstance(v, Tensor) else v for k,v in inputs.items()}
  out_names = list(run_onnx.graph_outputs)
  out_values = ort_sess.run(out_names, np_inputs)
  ort_out = dict(zip(out_names, out_values))

  tinygrad_out = run_onnx(inputs)

  assert tinygrad_out.keys() == ort_out.keys()
  for k in tinygrad_out.keys():
    tiny_v, onnx_v = tinygrad_out[k], ort_out[k]
    if tiny_v is None: assert onnx_v is None, f"{k}: {tiny_v=}, {onnx_v=}"
    else: np.testing.assert_allclose(tiny_v.numpy(), onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tinygrad_out.keys()}")