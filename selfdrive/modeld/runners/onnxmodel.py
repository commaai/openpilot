import onnx
import itertools
import os
import sys
import numpy as np
from typing import Any

from openpilot.selfdrive.modeld.runners.runmodel_pyx import RunModel

ORT_TYPES_TO_NP_TYPES = {'tensor(float16)': np.float16, 'tensor(float)': np.float32, 'tensor(uint8)': np.uint8}

def attributeproto_fp16_to_fp32(attr):
  float32_list = np.frombuffer(attr.raw_data, dtype=np.float16)
  attr.data_type = 1
  attr.raw_data = float32_list.astype(np.float32).tobytes()

def convert_fp16_to_fp32(path):
  model = onnx.load(path)
  for i in model.graph.initializer:
    if i.data_type == 10:
      attributeproto_fp16_to_fp32(i)
  for i in itertools.chain(model.graph.input, model.graph.output):
    if i.type.tensor_type.elem_type == 10:
      i.type.tensor_type.elem_type = 1
  for i in model.graph.node:
    for a in i.attribute:
      if hasattr(a, 't'):
        if a.t.data_type == 10:
          attributeproto_fp16_to_fp32(a.t)
  return model.SerializeToString()

def create_ort_session(path, fp16_to_fp32):
  os.environ["OMP_NUM_THREADS"] = "4"
  os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

  import onnxruntime as ort
  print("Onnx available providers: ", ort.get_available_providers(), file=sys.stderr)
  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

  provider: str | tuple[str, dict[Any, Any]]
  if 'OpenVINOExecutionProvider' in ort.get_available_providers() and 'ONNXCPU' not in os.environ:
    provider = 'OpenVINOExecutionProvider'
  elif 'CUDAExecutionProvider' in ort.get_available_providers() and 'ONNXCPU' not in os.environ:
    options.intra_op_num_threads = 2
    provider = ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'})
  else:
    options.intra_op_num_threads = 2
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider = 'CPUExecutionProvider'

  model_data = convert_fp16_to_fp32(path) if fp16_to_fp32 else path
  print("Onnx selected provider: ", [provider], file=sys.stderr)
  ort_session = ort.InferenceSession(model_data, options, providers=[provider])
  print("Onnx using ", ort_session.get_providers(), file=sys.stderr)
  return ort_session


class ONNXModel(RunModel):
  def __init__(self, path, output, runtime, use_tf8, cl_context):
    self.inputs = {}
    self.output = output
    self.use_tf8 = use_tf8

    self.session = create_ort_session(path, fp16_to_fp32=True)
    self.input_names = [x.name for x in self.session.get_inputs()]
    self.input_shapes = {x.name: [1, *x.shape[1:]] for x in self.session.get_inputs()}
    self.input_dtypes = {x.name: ORT_TYPES_TO_NP_TYPES[x.type] for x in self.session.get_inputs()}

    # run once to initialize CUDA provider
    if "CUDAExecutionProvider" in self.session.get_providers():
      self.session.run(None, {k: np.zeros(self.input_shapes[k], dtype=self.input_dtypes[k]) for k in self.input_names})
    print("ready to run onnx model", self.input_shapes, file=sys.stderr)

  def addInput(self, name, buffer):
    assert name in self.input_names
    self.inputs[name] = buffer

  def setInputBuffer(self, name, buffer):
    assert name in self.inputs
    self.inputs[name] = buffer

  def getCLBuffer(self, name):
    return None

  def execute(self):
    inputs = {k: (v.view(np.uint8) / 255. if self.use_tf8 and k == 'input_img' else v) for k,v in self.inputs.items()}
    inputs = {k: v.reshape(self.input_shapes[k]).astype(self.input_dtypes[k]) for k,v in inputs.items()}
    outputs = self.session.run(None, inputs)
    assert len(outputs) == 1, "Only single model outputs are supported"
    self.output[:] = outputs[0]
    return self.output
