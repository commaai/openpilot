#!/usr/bin/env python3
import os
from openpilot.system.hardware import TICI
os.environ['DEV'] = 'QCOM' if TICI else 'CPU'
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import time
import pickle
import numpy as np
from pathlib import Path

from cereal import messaging
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.model import dmonitoringmodel_intrinsics
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext, MonitoringModelFrame
from openpilot.selfdrive.modeld.parse_model_outputs import sigmoid, safe_exp
from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address

PROCESS_NAME = "selfdrive.modeld.dmonitoringmodeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
MODEL_PKL_PATH = Path(__file__).parent / 'models/dmonitoring_model_tinygrad.pkl'
METADATA_PATH = Path(__file__).parent / 'models/dmonitoring_model_metadata.pkl'


class ModelState:
  inputs: dict[str, np.ndarray]
  output: np.ndarray

  def __init__(self, cl_ctx):
    with open(METADATA_PATH, 'rb') as f:
      model_metadata = pickle.load(f)
      self.input_shapes = model_metadata['input_shapes']
      self.output_slices = model_metadata['output_slices']

    self.frame = MonitoringModelFrame(cl_ctx)
    self.numpy_inputs = {
      'calib': np.zeros(self.input_shapes['calib'], dtype=np.float32),
    }

    self.tensor_inputs = {k: Tensor(v, device='NPY').realize() for k,v in self.numpy_inputs.items()}
    with open(MODEL_PKL_PATH, "rb") as f:
      self.model_run = pickle.load(f)

  def run(self, buf: VisionBuf, calib: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, float]:
    self.numpy_inputs['calib'][0,:] = calib

    t1 = time.perf_counter()

    input_img_cl = self.frame.prepare(buf, transform.flatten())
    if TICI:
      # The imgs tensors are backed by opencl memory, only need init once
      if 'input_img' not in self.tensor_inputs:
        self.tensor_inputs['input_img'] = qcom_tensor_from_opencl_address(input_img_cl.mem_address, self.input_shapes['input_img'], dtype=dtypes.uint8)
    else:
      self.tensor_inputs['input_img'] = Tensor(self.frame.buffer_from_cl(input_img_cl).reshape(self.input_shapes['input_img']), dtype=dtypes.uint8).realize()


    output = self.model_run(**self.tensor_inputs).numpy().flatten()

    t2 = time.perf_counter()
    return output, t2 - t1

def slice_outputs(model_outputs, output_slices):
  return  {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}

def parse_model_output(model_output):
  parsed = {}
  parsed['wheel_on_right'] = sigmoid(model_output['wheel_on_right'])
  for ds_suffix in ['lhd', 'rhd']:
    face_descs = model_output[f'face_descs_{ds_suffix}']
    parsed[f'face_descs_{ds_suffix}'] = face_descs[:, :-6]
    parsed[f'face_descs_{ds_suffix}_std'] = safe_exp(face_descs[:, -6:])
    for key in ['face_prob', 'left_eye_prob', 'right_eye_prob','left_blink_prob', 'right_blink_prob', 'sunglasses_prob', 'using_phone_prob']:
      parsed[f'{key}_{ds_suffix}'] = sigmoid(model_output[f'{key}_{ds_suffix}'])
  return parsed

def fill_driver_data(msg, model_output, ds_suffix):
  msg.faceOrientation = model_output[f'face_descs_{ds_suffix}'][0, :3].tolist()
  msg.faceOrientationStd = model_output[f'face_descs_{ds_suffix}_std'][0, :3].tolist()
  msg.facePosition = model_output[f'face_descs_{ds_suffix}'][0, 3:5].tolist()
  msg.facePositionStd = model_output[f'face_descs_{ds_suffix}_std'][0, 3:5].tolist()
  msg.faceProb = model_output[f'face_prob_{ds_suffix}'][0, 0].item()
  msg.leftEyeProb = model_output[f'left_eye_prob_{ds_suffix}'][0, 0].item()
  msg.rightEyeProb = model_output[f'right_eye_prob_{ds_suffix}'][0, 0].item()
  msg.leftBlinkProb = model_output[f'left_blink_prob_{ds_suffix}'][0, 0].item()
  msg.rightBlinkProb = model_output[f'right_blink_prob_{ds_suffix}'][0, 0].item()
  msg.sunglassesProb = model_output[f'sunglasses_prob_{ds_suffix}'][0, 0].item()
  msg.phoneProb = model_output[f'using_phone_prob_{ds_suffix}'][0, 0].item()

def get_driverstate_packet(model_output, frame_id: int, location_ts: int, exec_time: float, gpu_exec_time: float):
  msg = messaging.new_message('driverStateV2', valid=True)
  ds = msg.driverStateV2
  ds.frameId = frame_id
  ds.modelExecutionTime = exec_time
  ds.gpuExecutionTime = gpu_exec_time
  ds.rawPredictions = model_output['raw_pred']
  ds.wheelOnRightProb = model_output['wheel_on_right'][0, 0].item()
  fill_driver_data(ds.leftDriverData, model_output, 'lhd')
  fill_driver_data(ds.rightDriverData, model_output, 'rhd')
  return msg


def main():
  config_realtime_process(7, 5)

  cl_context = CLContext()
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, dmonitoringmodeld starting")

  cloudlog.warning("connecting to driver stream")
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True, cl_context)
  while not vipc_client.connect(False):
    time.sleep(0.1)
  assert vipc_client.is_connected()
  cloudlog.warning(f"connected with buffer size: {vipc_client.buffer_len}")

  sm = SubMaster(["liveCalibration"])
  pm = PubMaster(["driverStateV2"])

  calib = np.zeros(model.numpy_inputs['calib'].size, dtype=np.float32)
  model_transform = None

  while True:
    buf = vipc_client.recv()
    if buf is None:
      continue

    if model_transform is None:
      cam = _os_fisheye if buf.width == _os_fisheye.width else _ar_ox_fisheye
      model_transform = np.linalg.inv(np.dot(dmonitoringmodel_intrinsics, np.linalg.inv(cam.intrinsics))).astype(np.float32)

    sm.update(0)
    if sm.updated["liveCalibration"]:
      calib[:] = np.array(sm["liveCalibration"].rpyCalib)

    t1 = time.perf_counter()
    model_output, gpu_execution_time = model.run(buf, calib, model_transform)
    t2 = time.perf_counter()
    raw_pred = model_output.tobytes() if SEND_RAW_PRED else b''
    model_output = slice_outputs(model_output, model.output_slices)
    model_output = parse_model_output(model_output)
    model_output['raw_pred'] = raw_pred
    msg = get_driverstate_packet(model_output, vipc_client.frame_id, vipc_client.timestamp_sof, t2 - t1, gpu_execution_time)
    pm.send("driverStateV2", msg)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
