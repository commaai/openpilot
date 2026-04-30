#!/usr/bin/env python3
import os
from openpilot.selfdrive.modeld.tinygrad_helpers import MODELS_DIR, set_tinygrad_backend_from_compiled_flags
set_tinygrad_backend_from_compiled_flags()

USBGPU = "USBGPU" in os.environ
if USBGPU:
  os.environ['DEV'] = 'AMD'
  os.environ['AMD_IFACE'] = 'USB'
from tinygrad.tensor import Tensor
import time
import pickle
import numpy as np
from openpilot.system.hardware import ASIUS
import cereal.messaging as messaging
from cereal import car, log
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.compile_modeld import CompileConfig, make_input_queues
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.common.file_chunker import read_file_chunked
import ctypes as _c
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan


PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

VISION_METADATA_PATH = MODELS_DIR / 'driving_vision_metadata.pkl'
POLICY_METADATA_PATH = MODELS_DIR / 'driving_policy_metadata.pkl'

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3



def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    plan = model_output['plan'][0]
    desired_accel, should_stop = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                     plan[:,Plan.ACCELERATION][:,0],
                                                     ModelConstants.T_IDXS,
                                                     action_t=long_action_t)
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)

    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
    if v_ego > MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
    else:
      desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                  desiredAcceleration=float(desired_accel),
                                  shouldStop=bool(should_stop))

class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState:
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, cam_w: int, cam_h: int):
    with open(VISION_METADATA_PATH, 'rb') as f:
      vision_metadata = pickle.load(f)
      self.vision_input_shapes =  vision_metadata['input_shapes']
      self.vision_input_names = list(self.vision_input_shapes.keys())
      self.vision_output_slices = vision_metadata['output_slices']

    with open(POLICY_METADATA_PATH, 'rb') as f:
      policy_metadata = pickle.load(f)
      self.policy_input_shapes =  policy_metadata['input_shapes']
      self.policy_output_slices = policy_metadata['output_slices']

    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    self.input_queues, self.npy = make_input_queues(self.vision_input_shapes, self.policy_input_shapes, self.frame_skip)
    self.full_frames : dict[str, Tensor] = {}
    self._blob_cache : dict[int, Tensor] = {}
    self.parser = Parser()
    self.frame_buf_params = {k: get_nv12_info(cam_w, cam_h) for k in ('img', 'big_img')}
    if ASIUS:
      self._frame_upload_bufs = {k: Tensor.zeros(get_nv12_info(cam_w, cam_h)[3], dtype='uint8').contiguous().realize()
                                 for k in ('img', 'big_img')}
      import ctypes as _ctypes
      from tinygrad.device import Device
      from tinygrad.runtime.autogen import opencl as _cl
      self._cl, self._ctypes = _cl, _ctypes
      self._cl_dev = Device['CL']
    self.run_policy = pickle.loads(read_file_chunked(CompileConfig(cam_w, cam_h, prefix='driving_', prepare_only=False).pkl_path))
    if ASIUS:
      self._fp_lib = _c.CDLL('/data/openpilot/fast_parse.so')
      self._fp_lib.fast_parse_vision.argtypes = [_c.c_void_p]
      self._fp_lib.fast_parse_vision.restype = None
      self._fp_lib.fast_parse_policy.argtypes = [_c.c_void_p]
      self._fp_lib.fast_parse_policy.restype = None
    self.warp_enqueue = pickle.loads(read_file_chunked(CompileConfig(cam_w, cam_h, prefix='driving_', prepare_only=True).pkl_path))
    self.warp_enqueue(
      **self.input_queues,
      frame=Tensor.zeros(self.frame_buf_params['img'][3], dtype='uint8').contiguous().realize(),
      big_frame=Tensor.zeros(self.frame_buf_params['big_img'][3], dtype='uint8').contiguous().realize())

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def prepare_inputs(self, bufs, transforms, inputs):
    for key in bufs:
      yuv_size = self.frame_buf_params[key][3]
      arr = np.frombuffer(bufs[key].data, dtype=np.uint8)[:yuv_size]
      cl_buf = self._frame_upload_bufs[key].uop.base.buffer
      cl_buf.ensure_allocated()
      if cl_buf.nbytes == yuv_size:
        cl_buf.copyin(memoryview(arr))
      else:
        if not hasattr(self, '_padded_cache'): self._padded_cache = {}
        if key not in self._padded_cache or len(self._padded_cache[key]) != cl_buf.nbytes:
          self._padded_cache[key] = np.empty(cl_buf.nbytes, dtype=np.uint8)
        self._padded_cache[key][:len(arr)] = arr
        cl_buf.copyin(memoryview(self._padded_cache[key]))
      self.full_frames[key] = self._frame_upload_bufs[key]
    self._cl.clFinish(self._cl_dev.queue)
    self._cl_dev.pending_copyin.clear()
    inputs['desire_pulse'][0] = 0
    self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['tfm'][:,:] = transforms['img'][:,:]
    self.npy['big_tfm'][:,:] = transforms['big_img'][:,:]
    self._inputs_prepared = True

  def _build_parsed_dict(self, v, p):
    d = {}
    vs = self.vision_output_slices
    for name, sl, shape in [
      ('pose', vs['pose'], (1,6)), ('wide_from_device_euler', vs['wide_from_device_euler'], (1,3)),
      ('road_transform', vs['road_transform'], (1,6)),
      ('lane_lines', vs['lane_lines'], (1,4,33,2)), ('road_edges', vs['road_edges'], (1,2,33,2)),
      ('lead', vs['lead'], (1,3,6,4))]:
      nv = (sl.stop - sl.start) // 2
      d[name] = v[sl.start:sl.start+nv].reshape(shape)
      d[name+'_stds'] = v[sl.start+nv:sl.stop].reshape(shape)
    d['meta'] = v[np.newaxis, vs['meta']]
    d['desire_pred'] = v[vs['desire_pred'].start:vs['desire_pred'].stop].reshape(1,4,8)
    d['lane_lines_prob'] = v[np.newaxis, vs['lane_lines_prob']]
    d['lead_prob'] = v[np.newaxis, vs['lead_prob']]
    d['hidden_state'] = v[np.newaxis, vs['hidden_state']]
    ps = self.policy_output_slices
    psl = ps['plan']
    pnv = (psl.stop - psl.start) // 2
    d['plan'] = p[psl.start:psl.start+pnv].reshape(1,33,15)
    d['plan_stds'] = p[psl.start+pnv:psl.stop].reshape(1,33,15)
    d['desire_state'] = p[ps['desire_state'].start:ps['desire_state'].stop].reshape(1,8)
    return d

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
                inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    import gc
    gc.disable()
    try:
      return self._run_inner(bufs, transforms, inputs, prepare_only)
    finally:
      gc.enable()

  def _run_inner(self, bufs, transforms, inputs, prepare_only):
    _skip_copyin = getattr(self, '_inputs_prepared', False)
    if _skip_copyin: self._inputs_prepared = False
    for key in bufs:
      yuv_size = self.frame_buf_params[key][3]
      if ASIUS:
        if not _skip_copyin:
          arr = np.frombuffer(bufs[key].data, dtype=np.uint8)[:yuv_size]
          cl_buf = self._frame_upload_bufs[key].uop.base.buffer
          cl_buf.ensure_allocated()
          if cl_buf.nbytes == yuv_size:
            cl_buf.copyin(memoryview(arr))
          else:
            if not hasattr(self, '_padded_cache'): self._padded_cache = {}
            if key not in self._padded_cache or len(self._padded_cache[key]) != cl_buf.nbytes:
              self._padded_cache[key] = np.empty(cl_buf.nbytes, dtype=np.uint8)
            self._padded_cache[key][:len(arr)] = arr
            cl_buf.copyin(memoryview(self._padded_cache[key]))
        self.full_frames[key] = self._frame_upload_bufs[key]
      else:
        ptr = bufs[key].data.ctypes.data
        cache_key = (key, ptr)
        if cache_key not in self._blob_cache:
          self._blob_cache[cache_key] = Tensor.from_blob(ptr, (yuv_size,), dtype='uint8')
        self.full_frames[key] = self._blob_cache[cache_key]

    if not _skip_copyin:
      inputs['desire_pulse'][0] = 0
      self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
      self.prev_desire[:] = inputs['desire_pulse']
      self.npy['traffic_convention'][:] = inputs['traffic_convention']
      self.npy['tfm'][:,:] = transforms['img'][:,:]
      self.npy['big_tfm'][:,:] = transforms['big_img'][:,:]

    if prepare_only:
      self.warp_enqueue(**self.input_queues, frame=self.full_frames['img'], big_frame=self.full_frames['big_img'])
      if ASIUS:
        self._cl.clFlush(self._cl_dev.queue)
      return None

    vision_tensor, policy_tensor = self.run_policy(
      **self.input_queues, frame=self.full_frames['img'], big_frame=self.full_frames['big_img']
    )

    if ASIUS:
      _cl, dev, _ctypes = self._cl, self._cl_dev, self._ctypes
      vbuf = vision_tensor.uop.base.buffer
      pbuf = policy_tensor.uop.base.buffer
      vbuf.ensure_allocated()
      pbuf.ensure_allocated()
      if not hasattr(self, '_vision_np'):
        self._vision_np = np.empty(vbuf.nbytes // 4, dtype=np.float32)
        self._policy_np = np.empty(pbuf.nbytes // 4, dtype=np.float32)
        self._vision_ptr = _ctypes.c_void_p(self._vision_np.ctypes.data)
        self._policy_ptr = _ctypes.c_void_p(self._policy_np.ctypes.data)
      _cl.clEnqueueReadBuffer(dev.queue, vbuf._buf[0], False, 0, vbuf.nbytes, self._vision_ptr, 0, None, None)
      _cl.clEnqueueReadBuffer(dev.queue, pbuf._buf[0], False, 0, pbuf.nbytes, self._policy_ptr, 0, None, None)
      _cl.clFlush(dev.queue)
      _cl.clFinish(dev.queue)
      dev.pending_copyin.clear()
      return (self._vision_np, self._policy_np)
    else:
      vision_output = vision_tensor.numpy().flatten()
      policy_output = policy_tensor.numpy().flatten()

    vision_outputs_dict = self.parser.parse_vision_outputs(self.slice_outputs(vision_output, self.vision_output_slices))
    policy_outputs_dict = self.parser.parse_policy_outputs(self.slice_outputs(policy_output, self.policy_output_slices))
    combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}
    if SEND_RAW_PRED:
      combined_outputs_dict['raw_pred'] = np.concatenate([vision_output.copy(), policy_output.copy()])
    return combined_outputs_dict


def main(demo=False):
  cloudlog.warning("modeld init")

  if not USBGPU:
    # USB GPU currently saturates a core so can't do this yet,
    # also need to move the aux USB interrupts for good timings
    config_realtime_process(7, 54)

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  st = time.monotonic()
  cloudlog.warning("loading model")
  model = ModelState(vipc_client_main.width, vipc_client_main.height)
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()


  if demo or ASIUS:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

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

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    inputs:dict[str, np.ndarray] = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
    }

    if ASIUS:
      model.prepare_inputs(bufs, transforms, inputs)
    mt1 = time.perf_counter()
    model_output = model.run(bufs, transforms, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1
    if ASIUS and model_output is not None:
      vision_output, policy_output = model_output
      model._fp_lib.fast_parse_vision(vision_output.ctypes.data)
      model._fp_lib.fast_parse_policy(policy_output.ctypes.data)
      model_output = model._build_parsed_dict(vision_output, policy_output)
      if SEND_RAW_PRED:
        model_output['raw_pred'] = np.concatenate([vision_output.copy(), policy_output.copy()])

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')

      frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
      action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
      action = get_action_from_model(model_output, prev_action, lat_delay + frame_delay + action_delay, long_delay + frame_delay + action_delay, v_ego)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      modelv2_send.modelV2.meta.laneTurnDirection = DH.lane_turn_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction
      drivingdata_send.drivingModelData.meta.laneTurnDirection = DH.lane_turn_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
