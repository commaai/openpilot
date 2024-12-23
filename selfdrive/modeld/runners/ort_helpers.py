import onnx
import onnxruntime as ort
import numpy as np
import itertools

ORT_TYPES_TO_NP_TYPES = {'tensor(float16)': np.float16, 'tensor(float)': np.float32, 'tensor(uint8)': np.uint8}

def attributeproto_fp16_to_fp32(attr):
  float32_list = np.frombuffer(attr.raw_data, dtype=np.float16)
  attr.data_type = 1
  attr.raw_data = float32_list.astype(np.float32).tobytes()

def convert_fp16_to_fp32(model):
  for i in model.graph.initializer:
    if i.data_type == 10:
      attributeproto_fp16_to_fp32(i)
  for i in itertools.chain(model.graph.input, model.graph.output):
    if i.type.tensor_type.elem_type == 10:
      i.type.tensor_type.elem_type = 1
  for i in model.graph.node:
    if i.op_type == 'Cast' and i.attribute[0].i == 10:
      i.attribute[0].i = 1
    for a in i.attribute:
      if hasattr(a, 't'):
        if a.t.data_type == 10:
          attributeproto_fp16_to_fp32(a.t)
  return model.SerializeToString()


def make_onnx_cpu_runner(model_path):
  options = ort.SessionOptions()
  options.intra_op_num_threads = 4
  options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  model_data = convert_fp16_to_fp32(onnx.load(model_path))
  return ort.InferenceSession(model_data,  options, providers=['CPUExecutionProvider'])
