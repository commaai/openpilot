#!/usr/bin/env python3
import os, sys, io, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"
if "OPT" not in os.environ: os.environ["OPT"] = "99"
os.environ["PREREALIZE"] = "0"

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

import onnx
from typing import Tuple, List
from extra.utils import fetch
from extra.onnx import get_run_onnx
from tinygrad.graph import print_tree, log_schedule_item
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, partition, GlobalCounters, Context, DEBUG, getenv, ImageDType, GRAPH
from tinygrad.realize import run_schedule
from tinygrad.ops import LoadOps, Device, ScheduleItem
from tinygrad.features.image import fix_schedule_for_images
Device.DEFAULT = "GPU"

def get_schedule(onnx_data, supercombo_dtypes=False) -> Tuple[List[ScheduleItem], List[ScheduleItem]]:
  Tensor.no_grad = True
  Tensor.training = False

  # load the model
  onnx_model = onnx.load(io.BytesIO(onnx_data))
  run_onnx = get_run_onnx(onnx_model)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types_onnx = {inp.name:onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}

  # run the model
  input_types = {k:getattr(dtypes, input_types_onnx[k].name)for k in input_types_onnx.keys()}
  print(input_types)
  if supercombo_dtypes:
    input_types = {k:dtypes.float32 if 'img' not in k else dtypes.uint8 for k in input_types.keys()}
  print(input_types)

  inputs = {k:Tensor.empty(*shp, dtype=input_types[k]) for k,shp in input_shapes.items()}
  ret: Tensor = next(iter(run_onnx(inputs).values())).cast(dtypes.float32).contiguous()
  schedule = ret.lazydata.schedule()

  # filter schedule that don't depend on the inputs
  input_lb = [x.lazydata.base for x in inputs.values()]
  depends = set(input_lb)
  for si in schedule:
    if any(b in depends for b in si.inputs):
      depends.add(si.out)

  # run all kernels that don't depend on the inputs
  # NOTE: there's two extra kernels due to fusions that now happen since the weights aren't realized
  schedule, schedule_independent = partition(schedule, lambda si: si.out in depends)
  print(f"{len(schedule)} schedule items depend on the input, {len(schedule_independent)} don't")

  # confirm no loadops in the (non independent) schedule except for the ones that load the input buffers
  assert all(si.ast.op not in LoadOps or si.out in input_lb for si in schedule), "has loadops, can't compile to Thneed"
  return schedule, schedule_independent, inputs

def schedule_to_thneed(schedule, output_fn):
  from extra.thneed import Thneed

  # transform to CL.CACHE
  used_ops = 0
  cl_cache = []
  for si in schedule:
    prg = Device["GPU"].method_cache[si.ast]
    args = (si.out,) + si.inputs

    # pass these to thneed
    setattr(prg.clprg, 'op_estimate', prg.op_estimate)
    setattr(prg.clprg, 'prg', prg.prg)

    global_size = prg.global_size + [1]*(3-len(prg.global_size))
    local_size = prg.local_size + [1]*(3-len(prg.local_size))
    cl_cache.append((prg.clprg, [[int(g*l) for g,l in zip(global_size, local_size)], local_size, *[x.realized._buf for x in args]]))
    used_ops += prg.op_estimate

  from extra.thneed import Thneed
  input_rawbuffers = {k:inputs[k].lazydata.realized for k in inputs.keys()}
  t = Thneed(cl_cache, {k:v._buf for k,v in input_rawbuffers.items()})

  # save thneed (before run)
  t.save(output_fn)

  print(f"buffers to save: {len(t.buffers_to_save)}, inputs: {list(t.inputs.keys())}, outputs: {t.outputs}")
  runtime = t.run()
  print(f"network using {used_ops/1e9:.2f} GOPS with runtime {runtime*1e3:.2f} ms that's {used_ops/runtime*1e-9:.2f} GFLOPS")

def thneed_test_onnx(onnx_data, output_fn):
  import onnx
  import pyopencl as cl
  from tinygrad.runtime.ops_gpu import CL
  import numpy as np
  from extra.thneed import Thneed
  onnx_model = onnx.load(io.BytesIO(onnx_data))

  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  inputs = {k:Tensor.randn(*shp, requires_grad=False)*8 for k,shp in input_shapes.items()}
  new_np_inputs = {k:v.realize().numpy() for k,v in inputs.items()}

  if getenv("ORT"):
    # test with onnxruntime
    import onnxruntime as ort
    onnx_session = ort.InferenceSession(onnx_data)
    onnx_output = onnx_session.run([onnx_model.graph.output[0].name], {k:v.astype(np.float16) for k,v in new_np_inputs.items()})
    new_torch_out = onnx_output[0]
  else:
    # test with torch
    from test.models.test_onnx import run_onnx_torch
    new_torch_out = run_onnx_torch(onnx_model, new_np_inputs).numpy()

  if output_fn is None:
    # non thneed
    run_onnx = get_run_onnx(onnx_model)
    new_tinygrad_out = next(iter(run_onnx(inputs).values())).cast(dtypes.float32).numpy()
    np.testing.assert_allclose(new_torch_out, new_tinygrad_out, atol=1e-4, rtol=1e-2)
    print("classic self-test passed!")
  else:
    # load thneed and try that
    nt = Thneed()
    nt.load(output_fn)

    # inputs
    for k,v in nt.inputs.items():
      cl.enqueue_copy(CL.cl_queue[0], v, new_np_inputs[k], is_blocking=True)

    nt.run()
    new_thneed_out = np.empty((nt.outputs[0].size//4,), dtype=np.float32).reshape(new_torch_out.shape)
    cl.enqueue_copy(CL.cl_queue[0], new_thneed_out, nt.outputs[0], is_blocking=True)

    # compare torch to thneed
    np.testing.assert_allclose(new_torch_out, new_thneed_out, atol=1e-4, rtol=1e-2)
    print("thneed self-test passed!")

if __name__ == "__main__":
  onnx_fn = sys.argv[1] if len(sys.argv) > 1 else OPENPILOT_MODEL
  onnx_data = fetch(onnx_fn)

  # quick test for ONNX issues
  #thneed_test_onnx(onnx_data, None)
  #exit(0)

  # this is a hack due to supercombo being converted with f16 inputs but it uses f32 at runtime
  supercombo = 'supercombo'
  schedule, schedule_independent, inputs = get_schedule(onnx_data, supercombo_dtypes=supercombo)
  schedule, schedule_input = partition(schedule, lambda x: x.ast.op not in LoadOps)
  print(f"{len(schedule_input)} inputs")

  run_schedule(schedule_independent, disable_logging=True)
  run_schedule(schedule_input)
  with Context(DEBUG=2, BEAM=getenv("LATEBEAM")):
    schedule = fix_schedule_for_images(schedule)
    image_count = sum(isinstance(si.out.dtype, ImageDType) for si in schedule)
    print(f"**** running real kernels {image_count}/{len(schedule)} images ****")

    if GRAPH:
      for si in schedule_input: log_schedule_item(si)
      for si in schedule: log_schedule_item(si)

    GlobalCounters.reset()
    run_schedule(schedule[:])

  output_fn = sys.argv[2] if len(sys.argv) >= 3 else "/tmp/output.thneed"
  schedule_to_thneed(schedule, output_fn)

  FLOAT16 = getenv("FLOAT16", 0)
  if FLOAT16 == 0:
    try:
      thneed_test_onnx(onnx_data, output_fn)
    except ModuleNotFoundError as e:
      print(f"TEST NOT HAPPENING {e}")


