#!/usr/bin/env python3
import os
from openpilot.system.hardware import TICI
os.environ['DEV'] = 'QCOM' if TICI else 'CPU'
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import math
import time
import pickle
import ctypes
import numpy as np
from pathlib import Path

from cereal import messaging
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.model import dmonitoringmodel_intrinsics, DM_INPUT_SIZE
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext, MonitoringModelFrame
from openpilot.selfdrive.modeld.parse_model_outputs import sigmoid
from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address

MODEL_WIDTH, MODEL_HEIGHT = DM_INPUT_SIZE
CALIB_LEN = 3
FEATURE_LEN = 512
OUTPUT_SIZE = 84 + FEATURE_LEN

PROCESS_NAME = "selfdrive.modeld.dmonitoringmodeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
MODEL_PKL_PATH = Path(__file__).parent / 'models/dmonitoring_model_tinygrad.pkl'


class DriverStateResult(ctypes.Structure):
  _fields_ = [
    ("face_orientation", ctypes.c_float*3),
    ("face_position", ctypes.c_float*3),
    ("face_orientation_std", ctypes.c_float*3),
    ("face_position_std", ctypes.c_float*3),
    ("face_prob", ctypes.c_float),
    ("_unused_a", ctypes.c_float*8),
    ("left_eye_prob", ctypes.c_float),
    ("_unused_b", ctypes.c_float*8),
    ("right_eye_prob", ctypes.c_float),
    ("left_blink_prob", ctypes.c_float),
    ("right_blink_prob", ctypes.c_float),
    ("sunglasses_prob", ctypes.c_float),
    ("occluded_prob", ctypes.c_float),
    ("ready_prob", ctypes.c_float*4),
    ("not_ready_prob", ctypes.c_float*2)]


class DMonitoringModelResult(ctypes.Structure):
  _fields_ = [
    ("driver_state_lhd", DriverStateResult),
    ("driver_state_rhd", DriverStateResult),
    ("poor_vision_prob", ctypes.c_float),
    ("wheel_on_right_prob", ctypes.c_float),
    ("features", ctypes.c_float*FEATURE_LEN)]


class ModelState:
  inputs: dict[str, np.ndarray]
  output: np.ndarray

  def __init__(self, cl_ctx):
    assert ctypes.sizeof(DMonitoringModelResult) == OUTPUT_SIZE * ctypes.sizeof(ctypes.c_float)

    self.frame = MonitoringModelFrame(cl_ctx)
    self.numpy_inputs = {
      'calib': np.zeros((1, CALIB_LEN), dtype=np.float32),
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
        self.tensor_inputs['input_img'] = qcom_tensor_from_opencl_address(input_img_cl.mem_address, (1, MODEL_WIDTH*MODEL_HEIGHT), dtype=dtypes.uint8)
    else:
      self.tensor_inputs['input_img'] = Tensor(self.frame.buffer_from_cl(input_img_cl).reshape((1, MODEL_WIDTH*MODEL_HEIGHT)), dtype=dtypes.uint8).realize()


    output = self.model_run(**self.tensor_inputs).contiguous().realize().uop.base.buffer.numpy()

    t2 = time.perf_counter()
    return output, t2 - t1


def fill_driver_state(msg, ds_result: DriverStateResult):
  msg.faceOrientation = list(ds_result.face_orientation)
  msg.faceOrientationStd = [math.exp(x) for x in ds_result.face_orientation_std]
  msg.facePosition = list(ds_result.face_position[:2])
  msg.facePositionStd = [math.exp(x) for x in ds_result.face_position_std[:2]]
  msg.faceProb = float(sigmoid(ds_result.face_prob))
  msg.leftEyeProb = float(sigmoid(ds_result.left_eye_prob))
  msg.rightEyeProb = float(sigmoid(ds_result.right_eye_prob))
  msg.leftBlinkProb = float(sigmoid(ds_result.left_blink_prob))
  msg.rightBlinkProb = float(sigmoid(ds_result.right_blink_prob))
  msg.sunglassesProb = float(sigmoid(ds_result.sunglasses_prob))
  msg.occludedProb = float(sigmoid(ds_result.occluded_prob))
  msg.readyProb = [float(sigmoid(x)) for x in ds_result.ready_prob]
  msg.notReadyProb = [float(sigmoid(x)) for x in ds_result.not_ready_prob]


def get_driverstate_packet(model_output: np.ndarray, frame_id: int, location_ts: int, execution_time: float, gpu_execution_time: float):
  model_result = ctypes.cast(model_output.ctypes.data, ctypes.POINTER(DMonitoringModelResult)).contents
  msg = messaging.new_message('driverStateV2', valid=True)
  ds = msg.driverStateV2
  ds.frameId = frame_id
  ds.modelExecutionTime = execution_time
  ds.gpuExecutionTime = gpu_execution_time
  ds.poorVisionProb = float(sigmoid(model_result.poor_vision_prob))
  ds.wheelOnRightProb = float(sigmoid(model_result.wheel_on_right_prob))
  ds.rawPredictions = model_output.tobytes() if SEND_RAW_PRED else b''
  fill_driver_state(ds.leftDriverData, model_result.driver_state_lhd)
  fill_driver_state(ds.rightDriverData, model_result.driver_state_rhd)
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

  calib = np.zeros(CALIB_LEN, dtype=np.float32)
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

    pm.send("driverStateV2", get_driverstate_packet(model_output, vipc_client.frame_id, vipc_client.timestamp_sof, t2 - t1, gpu_execution_time))


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
