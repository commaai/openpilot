#!/usr/bin/env python3
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
from cereal.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.system.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext
from openpilot.selfdrive.modeld.models.driving_pyx import (
  PublishState, create_model_msg, create_pose_msg,
  FEATURE_LEN, HISTORY_BUFFER_LEN, DESIRE_LEN, TRAFFIC_CONVENTION_LEN, NAV_FEATURE_LEN, NAV_INSTRUCTION_LEN,
  OUTPUT_SIZE, NET_OUTPUT_SIZE, MODEL_FREQ)

MODEL_PATHS = {
  ModelRunner.THNEED: Path(__file__).parent / 'models/supercombo.thneed',
  ModelRunner.ONNX: Path(__file__).parent / 'models/supercombo.onnx'}

class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof

class ModelState:
  frame: ModelFrame
  wide_frame: ModelFrame
  inputs: Dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse
  model: ModelRunner

  def __init__(self, context: CLContext):
    self.frame = ModelFrame(context)
    self.wide_frame = ModelFrame(context)
    self.prev_desire = np.zeros(DESIRE_LEN, dtype=np.float32)
    self.output = np.zeros(NET_OUTPUT_SIZE, dtype=np.float32)
    self.inputs = {
      'desire': np.zeros(DESIRE_LEN * (HISTORY_BUFFER_LEN+1), dtype=np.float32),
      'traffic_convention': np.zeros(TRAFFIC_CONVENTION_LEN, dtype=np.float32),
      'nav_features': np.zeros(NAV_FEATURE_LEN, dtype=np.float32),
      'nav_instructions': np.zeros(NAV_INSTRUCTION_LEN, dtype=np.float32),
      'features_buffer': np.zeros(HISTORY_BUFFER_LEN * FEATURE_LEN, dtype=np.float32),
    }

    self.model = ModelRunner(MODEL_PATHS, self.output, Runtime.GPU, False, context)
    self.model.addInput("input_imgs", None)
    self.model.addInput("big_input_imgs", None)
    for k,v in self.inputs.items():
      self.model.addInput(k, v)

  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
                inputs: Dict[str, np.ndarray], prepare_only: bool) -> Optional[np.ndarray]:
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire'][0] = 0
    self.inputs['desire'][:-DESIRE_LEN] = self.inputs['desire'][DESIRE_LEN:]
    self.inputs['desire'][-DESIRE_LEN:] = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
    self.prev_desire[:] = inputs['desire']

    self.inputs['traffic_convention'][:] = inputs['traffic_convention']
    self.inputs['nav_features'][:] = inputs['nav_features']
    self.inputs['nav_instructions'][:] = inputs['nav_instructions']
    # self.inputs['driving_style'][:] = inputs['driving_style']

    # if getCLBuffer is not None, frame will be None
    self.model.setInputBuffer("input_imgs", self.frame.prepare(buf, transform.flatten(), self.model.getCLBuffer("input_imgs")))
    if wbuf is not None:
      self.model.setInputBuffer("big_input_imgs", self.wide_frame.prepare(wbuf, transform_wide.flatten(), self.model.getCLBuffer("big_input_imgs")))

    if prepare_only:
      return None

    self.model.execute()
    self.inputs['features_buffer'][:-FEATURE_LEN] = self.inputs['features_buffer'][FEATURE_LEN:]
    self.inputs['features_buffer'][-FEATURE_LEN:] = self.output[OUTPUT_SIZE:OUTPUT_SIZE+FEATURE_LEN]
    return self.output


def main():
  cloudlog.bind(daemon="selfdrive.modeld.modeld")
  setproctitle("selfdrive.modeld.modeld")
  config_realtime_process(7, 54)

  cl_context = CLContext()
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, modeld starting")

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster(["modelV2", "cameraOdometry"])
  sm = SubMaster(["lateralPlan", "roadCameraState", "liveCalibration", "driverMonitoringState", "navModel", "navInstruction"])

  state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0
  # last = 0.0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  driving_style = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
  nav_features = np.zeros(NAV_FEATURE_LEN, dtype=np.float32)
  nav_instructions = np.zeros(NAV_INSTRUCTION_LEN, dtype=np.float32)
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.error("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.error("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error("frames out of sync! main: {} ({:.5f}), extra: {} ({:.5f})".format(
          meta_main.frame_id, meta_main.timestamp_sof / 1e9,
          meta_extra.frame_id, meta_extra.timestamp_sof / 1e9))

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    # TODO: path planner timeout?
    sm.update(0)
    desire = sm["lateralPlan"].desire.raw
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    if sm.updated["liveCalibration"]:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      model_transform_main = get_warp_matrix(device_from_calib_euler, main_wide_camera, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, True, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < DESIRE_LEN:
      vec_desire[desire] = 1

    # Enable/disable nav features
    timestamp_llk = sm["navModel"].locationMonoTime
    nav_valid = sm.valid["navModel"] # and (nanos_since_boot() - timestamp_llk < 1e9)
    nav_enabled = nav_valid and params.get_bool("ExperimentalMode")

    if not nav_enabled:
      nav_features[:] = 0
      nav_instructions[:] = 0

    if nav_enabled and sm.updated["navModel"]:
      nav_features = np.array(sm["navModel"].features)

    if nav_enabled and sm.updated["navInstruction"]:
      nav_instructions[:] = 0
      for maneuver in sm["navInstruction"].allManeuvers:
        distance_idx = 25 + int(maneuver.distance / 20)
        direction_idx = 0
        if maneuver.modifier in ("left", "slight left", "sharp left"):
          direction_idx = 1
        if maneuver.modifier in ("right", "slight right", "sharp right"):
          direction_idx = 2
        if 0 <= distance_idx < 50:
          nav_instructions[distance_idx*3 + direction_idx] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    inputs:Dict[str, np.ndarray] = {
      'desire': vec_desire,
      'traffic_convention': traffic_convention,
      'driving_style': driving_style,
      'nav_features': nav_features,
      'nav_instructions': nav_instructions}

    mt1 = time.perf_counter()
    model_output = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      pm.send("modelV2", create_model_msg(model_output, state, meta_main.frame_id, meta_extra.frame_id, frame_id, frame_drop_ratio,
                                          meta_main.timestamp_eof, timestamp_llk, model_execution_time, nav_enabled, live_calib_seen))
      pm.send("cameraOdometry", create_pose_msg(model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen))

    # print("model process: %.2fms, from last %.2fms, vipc_frame_id %u, frame_id, %u, frame_drop %.3f" %
    #   ((mt2 - mt1)*1000, (mt1 - last)*1000, meta_extra.frame_id, frame_id, frame_drop_ratio))
    # last = mt1
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit()
