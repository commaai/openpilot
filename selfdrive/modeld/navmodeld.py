#!/usr/bin/env python3
import gc
import math
import time
import ctypes
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from cereal import messaging
from cereal.messaging import PubMaster, SubMaster
from cereal.visionipc import VisionIpcClient, VisionStreamType
from openpilot.system.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.realtime import set_realtime_priority
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime

NAV_INPUT_SIZE = 256*256
NAV_FEATURE_LEN = 256
NAV_DESIRE_LEN = 32
NAV_OUTPUT_SIZE = 2*2*ModelConstants.IDX_N + NAV_DESIRE_LEN + NAV_FEATURE_LEN
MODEL_PATHS = {
  ModelRunner.SNPE: Path(__file__).parent / 'models/navmodel_q.dlc',
  ModelRunner.ONNX: Path(__file__).parent / 'models/navmodel.onnx'}

class NavModelOutputXY(ctypes.Structure):
  _fields_ = [
    ("x", ctypes.c_float),
    ("y", ctypes.c_float)]

class NavModelOutputPlan(ctypes.Structure):
  _fields_ = [
    ("mean", NavModelOutputXY*ModelConstants.IDX_N),
    ("std", NavModelOutputXY*ModelConstants.IDX_N)]

class NavModelResult(ctypes.Structure):
  _fields_ = [
    ("plan", NavModelOutputPlan),
    ("desire_pred", ctypes.c_float*NAV_DESIRE_LEN),
    ("features", ctypes.c_float*NAV_FEATURE_LEN)]

class ModelState:
  inputs: Dict[str, np.ndarray]
  output: np.ndarray
  model: ModelRunner

  def __init__(self):
    assert ctypes.sizeof(NavModelResult) == NAV_OUTPUT_SIZE * ctypes.sizeof(ctypes.c_float)
    self.output = np.zeros(NAV_OUTPUT_SIZE, dtype=np.float32)
    self.inputs = {'input_img': np.zeros(NAV_INPUT_SIZE, dtype=np.uint8)}
    self.model = ModelRunner(MODEL_PATHS, self.output, Runtime.DSP, True, None)
    self.model.addInput("input_img", None)

  def run(self, buf:np.ndarray) -> Tuple[np.ndarray, float]:
    self.inputs['input_img'][:] = buf

    t1 = time.perf_counter()
    self.model.setInputBuffer("input_img", self.inputs['input_img'].view(np.float32))
    self.model.execute()
    t2 = time.perf_counter()
    return self.output, t2 - t1

def get_navmodel_packet(model_output: np.ndarray, valid: bool, frame_id: int, location_ts: int, execution_time: float, dsp_execution_time: float):
  model_result = ctypes.cast(model_output.ctypes.data, ctypes.POINTER(NavModelResult)).contents
  msg = messaging.new_message('navModel')
  msg.valid = valid
  msg.navModel.frameId = frame_id
  msg.navModel.locationMonoTime = location_ts
  msg.navModel.modelExecutionTime = execution_time
  msg.navModel.dspExecutionTime = dsp_execution_time
  msg.navModel.features = model_result.features[:]
  msg.navModel.desirePrediction = model_result.desire_pred[:]
  msg.navModel.position.x = [p.x for p in model_result.plan.mean]
  msg.navModel.position.y = [p.y for p in model_result.plan.mean]
  msg.navModel.position.xStd = [math.exp(p.x) for p in model_result.plan.std]
  msg.navModel.position.yStd = [math.exp(p.y) for p in model_result.plan.std]
  return msg


def main():
  gc.disable()
  set_realtime_priority(1)

  # there exists a race condition when two processes try to create a
  # SNPE model runner at the same time, wait for dmonitoringmodeld to finish
  cloudlog.warning("waiting for dmonitoringmodeld to initialize")
  if not Params().get_bool("DmModelInitialized", True):
    return

  model = ModelState()
  cloudlog.warning("models loaded, navmodeld starting")

  vipc_client = VisionIpcClient("navd", VisionStreamType.VISION_STREAM_MAP, True)
  while not vipc_client.connect(False):
    time.sleep(0.1)
  assert vipc_client.is_connected()
  cloudlog.warning(f"connected with buffer size: {vipc_client.buffer_len}")

  sm = SubMaster(["navInstruction"])
  pm = PubMaster(["navModel"])

  while True:
    buf = vipc_client.recv()
    if buf is None:
      continue

    sm.update(0)
    t1 = time.perf_counter()
    model_output, dsp_execution_time = model.run(buf.data[:buf.uv_offset])
    t2 = time.perf_counter()

    valid = vipc_client.valid and sm.valid["navInstruction"]
    pm.send("navModel", get_navmodel_packet(model_output, valid, vipc_client.frame_id, vipc_client.timestamp_sof, t2 - t1, dsp_execution_time))


if __name__ == "__main__":
  main()
