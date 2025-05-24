import time

onnx_path = "/tmp/my.onnx"
N = 2048
CNT = 400

"""
import torch
import torch.nn as nn
#dtype = torch.bfloat16
dtype = torch.float32
class MatMul(nn.Module):
  def __init__(self):
    super().__init__()
    self.a = nn.Linear(N, N, bias=False)
  def forward(self, x):
    x = x.to(dtype)
    for i in range(CNT): x = self.a(x).relu()
    return x.to(torch.float32)

torch_model = MatMul().to(dtype)
torch.onnx.export(torch_model, torch.randn(N, N), onnx_path)
"""

"""
import onnx
from tinygrad.tensor import Tensor
from extra.onnx import get_run_onnx
out = get_run_onnx(onnx.load(onnx_path))({"onnx::MatMul_0": Tensor.zeros(N, N)})
for x in out.values(): x.realize()
"""

from openvino.runtime import Core
core = Core()
devices = core.available_devices
for device in devices:
  device_name = core.get_property(device, "FULL_DEVICE_NAME")
  print(f"{device}: {device_name}")
model = core.read_model(onnx_path)
compiled_model = core.compile_model(model, device_name='GPU.0')
print(compiled_model)
ireq = compiled_model.create_infer_request()
for model_input in compiled_model.inputs:
  tensor = ireq.get_tensor(model_input)
  tensor.data[:] = 2
  print(tensor)
print("request")
ireq.infer()
ireq.infer()
print("did one")

REPS = 20
st = time.perf_counter()
for i in range(REPS): ireq.infer()
et = time.perf_counter() - st
print(f"{et*1000:.2f} ms {(CNT*N*N*N*REPS*2/et)*1e-9:.2f} GFLOPS")

