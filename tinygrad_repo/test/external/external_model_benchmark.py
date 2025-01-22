import csv, pathlib, time, numpy as np
from os import getenv
import torch
torch.set_num_threads(1)
import onnx
from onnx.helper import tensor_dtype_to_np_dtype
import onnxruntime as ort
from onnx2torch import convert
from extra.onnx import get_run_onnx
from tinygrad.helpers import OSX, DEBUG, fetch
from tinygrad import Tensor, Device

MODELS = {
  "resnet50": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx",
  "openpilot": "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx",
  "efficientnet": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
  "shufflenet": "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx",
  "commavq": "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/gpt2m.onnx",
  "dm": "https://github.com/commaai/openpilot/raw/ba7f840a06dbc8ae3c45b3b4976c88a21895aed0/selfdrive/modeld/models/dmonitoring_model.onnx",

  # broken in torch MPS
  # "zfnet": "https://github.com/onnx/models/raw/main/archive/vision/classification/zfnet-512/model/zfnet512-9.onnx",
  # TypeError: BatchNormalization() got an unexpected keyword argument 'is_test'
  # "densenet": "https://github.com/onnx/models/raw/main/archive/vision/classification/densenet-121/model/densenet-3.onnx",
  # AssertionError: only onnx version >= 10 supported for slice
  # "bert": "https://github.com/onnx/models/raw/main/archive/text/machine_comprehension/bert-squad/model/bertsquad-8.onnx",
  # really slow
  # "resnet18": "https://github.com/onnx/models/raw/main/archive/vision/classification/resnet/model/resnet18-v2-7.onnx",
}

CSV = {}
open_csv = None

def benchmark(mnm, nm, fxn):
  tms = []
  for _ in range(3):
    st = time.perf_counter_ns()
    ret = fxn()
    tms.append(time.perf_counter_ns() - st)
  print(f"{mnm:15s} {nm:25s} {min(tms)*1e-6:7.2f} ms")
  CSV[nm] = min(tms)*1e-6
  return min(tms), ret

#BASE = pathlib.Path(__file__).parents[2] / "weights" / "onnx"
BASE = pathlib.Path("/tmp/onnx")
def benchmark_model(m, devices, validate_outs=False):
  torch.manual_seed(1)
  global open_csv, CSV
  CSV = {"model": m}

  fn = fetch(MODELS[m])
  onnx_model = onnx.load(fn)
  output_names = [out.name for out in onnx_model.graph.output]
  excluded = {inp.name for inp in onnx_model.graph.initializer}
  input_shapes = {inp.name:tuple(x.dim_value if x.dim_value != 0 else 1 for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input if inp.name not in excluded}  # noqa: E501
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input if inp.name not in excluded}
  #input_types = {k:v if v!=np.float16 else np.float32 for k,v in input_types.items()}  # cast
  np_inputs = {k:torch.randn(shp).numpy().astype(input_types[k]) for k,shp in input_shapes.items()}
  assert len(input_shapes) < 30, f"too many input shapes {len(input_shapes)}"

  # print input names
  if DEBUG >= 2: print([inp.name for inp in onnx_model.graph.input if inp.name not in excluded])
  for device in devices:
    try:
      Device.DEFAULT = device
      inputs = {k:Tensor(inp) for k,inp in np_inputs.items()}
      tinygrad_model = get_run_onnx(onnx_model)
      benchmark(m, f"tinygrad_{device.lower()}_jitless", lambda: {k:v.numpy() for k,v in tinygrad_model(inputs).items()})

      from tinygrad.engine.jit import TinyJit
      tinygrad_jitted_model = TinyJit(lambda **kwargs: {k:v.realize() for k,v in tinygrad_model(kwargs).items()})
      for _ in range(3): {k:v.numpy() for k,v in tinygrad_jitted_model(**inputs).items()}
      benchmark(m, f"tinygrad_{device.lower()}_jit", lambda: {k:v.numpy() for k,v in tinygrad_jitted_model(**inputs).items()}) # noqa: F821
      del inputs, tinygrad_model, tinygrad_jitted_model
    except RuntimeError as e:
      # TODO: we don't run the dm model on METAL for now
      if Device.DEFAULT == "METAL":
        assert "buffer count limit" in str(e)
        return
      else: raise e

  # convert model to torch
  try:
    torch_model = convert(onnx_model)
  except Exception as e:
    # model conversion failed
    print(f"{m:16s}onnx2torch {type(e).__name__:>25}")
  else:
    torch_inputs = [torch.tensor(x) for x in np_inputs.values()]
    try: benchmark(m, "torch_cpu", lambda: torch_model(*torch_inputs))
    except Exception as e: print(f"{m:16s}torch_cpu {type(e).__name__:>25}")

    torch_device = "mps" if OSX else "cuda"
    torch_mps_model = torch_model.to(torch_device)
    torch_mps_inputs = [x.to(torch_device) for x in torch_inputs]
    try: benchmark(m, f"torch_{torch_device}", lambda: torch_mps_model(*torch_mps_inputs))
    except Exception as e: print(f"{m:16s}torch_{torch_device} {type(e).__name__:>25}")

  # bench onnxruntime
  ort_options = ort.SessionOptions()
  ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  ort_options.log_severity_level = 3  # no warnings
  for backend in ["CPU", "CUDA" if not OSX else "CoreML"]:  # https://onnxruntime.ai/docs/execution-providers/
    provider = backend+"ExecutionProvider"
    if provider not in ort.get_available_providers(): continue
    ort_sess = ort.InferenceSession(str(fn), ort_options, [provider])
    try:
      benchmark(m, f"onnxruntime_{backend.lower()}", lambda: ort_sess.run(output_names, np_inputs))
    except Exception as e: print(f"{m:16s}onnxruntime_{backend.lower()} {type(e).__name__:>25}")
    del ort_sess

  if validate_outs:
    for device in devices:
      rtol, atol = 2e-3, 2e-3  # tolerance for fp16 models
      Device.DEFAULT = device
      inputs = {k:Tensor(inp) for k,inp in np_inputs.items()}
      tinygrad_model = get_run_onnx(onnx_model)
      tinygrad_out = tinygrad_model(inputs)

      ort_sess = ort.InferenceSession(str(fn), ort_options, ["CPUExecutionProvider"])
      onnx_out = ort_sess.run(output_names, np_inputs)
      onnx_out = dict([*list(zip(output_names, onnx_out))])

      assert_allclose(tinygrad_out, onnx_out, rtol=rtol, atol=atol)
      print(f"{m:16s}outputs validated on {device=} with rtol={rtol:.1e}, atol={atol:.1e}")

  if open_csv is None:
    open_csv = csv.DictWriter(open('onnx_inference_speed.csv', 'w', newline=''), fieldnames=list(CSV.keys()))
    open_csv.writeheader()
  open_csv.writerow(CSV)

def assert_allclose(tiny_out:dict, onnx_out:dict, rtol=1e-5, atol=1e-5):
  assert len(tiny_out) == len(onnx_out) and tiny_out.keys() == onnx_out.keys()
  for k in tiny_out.keys():
    tiny_v, onnx_v = tiny_out[k], onnx_out[k]
    if tiny_v is None: assert tiny_v == onnx_v
    else: np.testing.assert_allclose(tiny_v.numpy(), onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tiny_out.keys()}")

if __name__ == "__main__":
  devices = [Device.DEFAULT] if getenv("NOCLANG") else [Device.DEFAULT, "CLANG"]
  if getenv("MODEL", "") != "": benchmark_model(getenv("MODEL", ""), devices, True)
  else:
    for m in MODELS: benchmark_model(m, devices, True)
