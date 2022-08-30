#!/usr/bin/env python3

import os
import sys
import numpy as np

# import torch
import time
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import onnxruntime as ort # pylint: disable=import-error

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def read(sz):
  dd = []
  gt = 0
  while gt < sz * 4:
    st = os.read(0, sz * 4 - gt)
    assert(len(st) > 0)
    dd.append(st)
    gt += len(st)
  return np.frombuffer(b''.join(dd), dtype=np.float32)

def write(d):
  os.write(1, d.tobytes())

# def to_numpy(tensor):
#    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def run_loop(m):
  ishapes = [[1]+ii.shape[1:] for ii in m.get_inputs()]
  keys = [x.name for x in m.get_inputs()]



  # x = torch.randn((1, 3, 224, 224))
  # x = x.to(device=device)
  # x = to_numpy(x)
  start_time = time.time()
  # run once to initialize CUDA provider
  if "CUDAExecutionProvider" in m.get_providers():
    # input_names = m.get_inputs()[0].name
    # output_names = None
    # io_binding = m.io_binding()
    # print("input_names: ",input_names, file=sys.stderr)
    # # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
    # io_binding.bind_cpu_input(input_names, zip(keys, [np.zeros(shp, dtype=np.float32) for shp in ishapes]))
    # io_binding.bind_output(output_names)
    # m.run_with_iobinding(io_binding)
    # # Y = io_binding.copy_outputs_to_cpu()[0]

    m.run(None, dict(zip(keys, [np.zeros(shp, dtype=np.float32) for shp in ishapes])))

  finish_time = time.time()
  # print("running the m.run spend ",finish_time-start_time, file=sys.stderr)

  print("ready to run onnx model", keys, ishapes, file=sys.stderr)
  last_time = 0
  while 1:
    inputs = []
    for shp in ishapes:
      ts = np.product(shp)
      inputs.append(read(ts).reshape(shp))
    ret = m.run(None, dict(zip(keys, inputs)))
    for r in ret:
      write(r)
    get_time = time.time()
    last_time=get_time


if __name__ == "__main__":
  print("Onnx available providers: ", ort.get_available_providers(), file=sys.stderr)
  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  if 'OpenVINOExecutionProvider' in ort.get_available_providers() and 'ONNXCPU' not in os.environ:
    provider = 'OpenVINOExecutionProvider'
  elif 'CUDAExecutionProvider' in ort.get_available_providers() and 'ONNXCPU' not in os.environ:
    options.intra_op_num_threads = 2
    provider = 'CUDAExecutionProvider'
  else:
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 8
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider = 'CPUExecutionProvider'

  print("Onnx selected provider: ", [provider], file=sys.stderr)
  
  start_time = time.time()
  ort_session = ort.InferenceSession(sys.argv[1], options, providers=[provider])
  finish_time = time.time()
  print("running the ort.InferenceSession spend ",finish_time-start_time, file=sys.stderr)

  print("Onnx using ", ort_session.get_providers(), file=sys.stderr)
  run_loop(ort_session)
