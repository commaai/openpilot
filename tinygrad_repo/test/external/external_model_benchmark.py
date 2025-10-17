import csv, pathlib, time
import numpy as np
import torch
torch.set_num_threads(1)
import onnxruntime as ort
from onnx2torch import convert
from tinygrad.frontend.onnx import OnnxRunner
from tinygrad.helpers import OSX, DEBUG, fetch, getenv
from tinygrad.dtype import _to_np_dtype
from tinygrad import Tensor, Device, dtypes

MODELS = {
  "resnet50": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx",
  "openpilot": "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx",
  "efficientnet": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
  "shufflenet": "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx",
  # TODO: precision issue
  # "commavq": "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/gpt2m.onnx",
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
half_models = ["openpilot", "commavq"]

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
  runner = OnnxRunner(fn)
  output_names = runner.graph_outputs
  input_shapes = {name: tuple(s if isinstance(s, int) and s != 0 else 1 for s in spec.shape) for name, spec in runner.graph_inputs.items()}
  input_types = {name: spec.dtype for name, spec in runner.graph_inputs.items()}
  np_inputs = {k:torch.randn(shp).numpy().astype(_to_np_dtype(input_types[k])) for k,shp in input_shapes.items()}
  assert len(input_shapes) < 30, f"too many input shapes {len(input_shapes)}"

  # print input names
  if DEBUG >= 2: print(list(runner.graph_inputs))
  for device in devices:
    Device.DEFAULT = device
    inputs = {k:Tensor(inp) for k,inp in np_inputs.items()}
    tinygrad_model = runner.to(device)
    benchmark(m, f"tinygrad_{device.lower()}_jitless", lambda: {k:v.numpy() for k,v in tinygrad_model(inputs).items()})

    from tinygrad.engine.jit import TinyJit
    tinygrad_jitted_model = TinyJit(lambda **kwargs: {k:v.realize() for k,v in tinygrad_model(kwargs).items()})
    for _ in range(3): {k:v.numpy() for k,v in tinygrad_jitted_model(**inputs).items()}
    benchmark(m, f"tinygrad_{device.lower()}_jit", lambda: {k:v.numpy() for k,v in tinygrad_jitted_model(**inputs).items()}) # noqa: F821
    del inputs, tinygrad_model, tinygrad_jitted_model

  # convert model to torch
  try:
    torch_model = convert(fn)
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
      # force half inputs to float for numerical stability when validating
      # this will rely on automatic dtype promotion for converting half weights inside the graph
      if m in half_models:
        inputs = {k:Tensor(inp, dtype=dtypes.float32) if inp.dtype == np.float16 else Tensor(inp) for k,inp in np_inputs.items()}
      else:
        inputs = {k:Tensor(inp) for k,inp in np_inputs.items()}
      tinygrad_model = runner.to(device)
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

def assert_allclose(tiny_out:dict, onnx_out:dict, rtol, atol):
  assert tiny_out.keys() == onnx_out.keys()
  for k in tiny_out.keys():
    tiny_v, onnx_v = tiny_out[k], onnx_out[k]
    np.testing.assert_allclose(tiny_v.numpy(), onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tiny_out.keys()}")

if __name__ == "__main__":
  devices = [Device.DEFAULT] if getenv("NOCLANG") else [Device.DEFAULT, "CPU"]
  if (model:=getenv("MODEL", "")) != "": benchmark_model(model, devices, validate_outs=True)
  else:
    for m in MODELS: benchmark_model(m, devices, validate_outs=True)
