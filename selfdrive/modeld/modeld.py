#!/usr/bin/env python3
import os
from openpilot.system.hardware import TICI
## TODO this is hack
if TICI:
  os.environ['QCOM'] = '1'
else:
  os.environ['GPU'] = '1'
import time
import pickle
import numpy as np
import cv2
import cereal.messaging as messaging
from cereal import car, log
from pathlib import Path
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system import sentry
from openpilot.selfdrive.car.card import convert_to_capnp
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext

from tinygrad.tensor import Tensor
import ctypes, array
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, to_mv, mv_address
Tensor.manual_seed(1337)
Tensor.no_grad = True

PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

MODEL_PATH = Path(__file__).parent / 'models/supercombo.onnx'
MODEL_PKL_PATH = Path(__file__).parent / 'models/supercombo_tinygrad.pkl'
METADATA_PATH = Path(__file__).parent / 'models/supercombo_metadata.pkl'

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2
IMG_INPUT_SHAPE = (1, 12, 128, 256)



def tensor_arange(end):
    return Tensor([float(i) for i in range(end)])

def tensor_round(tensor):
    return (tensor + 0.5).floor()

def warp_perspective_tinygrad(src, M_inv, dsize):
    h_dst, w_dst = dsize[1], dsize[0]
    h_src, w_src = src.shape[:2]

    x = tensor_arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst)
    y = tensor_arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst)
    ones = Tensor.ones_like(x)
    dst_coords = x.reshape((1,-1)).cat(y.reshape((1,-1))).cat(ones.reshape((1,-1)))


    src_coords = M_inv @ dst_coords
    src_coords = src_coords / src_coords[2:3, :]

    x_src = src_coords[0].reshape(h_dst, w_dst)
    y_src = src_coords[1].reshape(h_dst, w_dst)

    x_nearest = tensor_round(x_src).clip(0, w_src - 1).cast('int')
    y_nearest = tensor_round(y_src).clip(0, h_src - 1).cast('int')

    src_np = src
    if src_np.ndim == 3:
        dst = src[y_nearest, x_nearest, :]
    else:
        dst = src[y_nearest, x_nearest]

    return dst


def frame_prepare_tinygrad(input_frame, M_inv, M_inv_uv, W, H):
  y = warp_perspective_tinygrad(input_frame[:H*W].reshape((H,W)), M_inv, (MODEL_WIDTH, MODEL_HEIGHT)).numpy()
  u = warp_perspective_tinygrad(input_frame[H*W::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).numpy()
  v = warp_perspective_tinygrad(input_frame[H*W+1::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).numpy()
  yuv = np.concatenate([y.copy().flatten(), u.copy().flatten(), v.copy().flatten()]).reshape((1,MODEL_HEIGHT*3//2,MODEL_WIDTH))
  tensor = frames_to_tensor(yuv.copy())
  return Tensor(tensor)



def Tensor_from_cl(frame, cl_buffer):
  if TICI:
    cl_buf_desc_ptr = to_mv(cl_buffer.mem_address, 8).cast('Q')[0]
    rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20] # offset 0xA0 is a raw gpu pointer.
    return Tensor.from_blob(rawbuf_ptr, IMG_INPUT_SHAPE, dtype=dtypes.uint8)
  else:
    return Tensor(frame.buffer_from_cl(cl_buffer)).reshape(IMG_INPUT_SHAPE)
  

def frames_to_tensor(frames: np.ndarray) -> np.ndarray:
  H = (frames.shape[1]*2)//3
  W = frames.shape[2]
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

  in_img1[:, 0] = frames[:, 0:H:2, 0::2]
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))

  return in_img1


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
  inputs: dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, context: CLContext):
    self.frame = ModelFrame(context)
    self.wide_frame = ModelFrame(context)
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    self.inputs = {
      'desire': np.zeros((1, (ModelConstants.HISTORY_BUFFER_LEN+1), ModelConstants.DESIRE_LEN), dtype=np.float32),
      'traffic_convention': np.zeros((1, ModelConstants.TRAFFIC_CONVENTION_LEN), dtype=np.float32),
      'lateral_control_params': np.zeros((1, ModelConstants.LATERAL_CONTROL_PARAMS_LEN), dtype=np.float32),
      'prev_desired_curv': np.zeros((1,(ModelConstants.HISTORY_BUFFER_LEN+1), ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float32),
      'features_buffer': np.zeros((1, ModelConstants.HISTORY_BUFFER_LEN,  ModelConstants.FEATURE_LEN), dtype=np.float32),
    }
    self.tensor_inputs = {'input_imgs': Tensor.zeros(IMG_INPUT_SHAPE, dtype='uint8').contiguous().realize(),
                          'big_input_imgs': Tensor.zeros(IMG_INPUT_SHAPE, dtype='uint8').contiguous().realize(),}

    with open(METADATA_PATH, 'rb') as f:
      model_metadata = pickle.load(f)


    self.input_imgs = np.zeros(IMG_INPUT_SHAPE, dtype=np.uint8)
    self.big_input_imgs = np.zeros(IMG_INPUT_SHAPE, dtype=np.uint8)
    self.output_slices = model_metadata['output_slices']
    net_output_size = model_metadata['output_shapes']['outputs'][1]
    self.output = np.zeros(net_output_size, dtype=np.float32)
    self.parser = Parser()
    self.scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    with open(MODEL_PKL_PATH, "rb") as f:
      self.model_run = pickle.load(f)

  def slice_outputs(self, model_outputs: np.ndarray) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in self.output_slices.items()}
    if SEND_RAW_PRED:
      parsed_model_outputs['raw_pred'] = model_outputs.copy()
    return parsed_model_outputs


  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
                inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire'][0] = 0
    self.inputs['desire'][:-ModelConstants.DESIRE_LEN] = self.inputs['desire'][ModelConstants.DESIRE_LEN:]
    self.inputs['desire'][-ModelConstants.DESIRE_LEN:] = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
    self.prev_desire[:] = inputs['desire']

    self.inputs['traffic_convention'][:] = inputs['traffic_convention']
    self.inputs['lateral_control_params'][:] = inputs['lateral_control_params']
    for k, v in self.inputs.items():
      self.tensor_inputs[k] = Tensor(v)


    scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    self.tensor_inputs['M_inv'] = Tensor(transform)
    self.tensor_inputs['M_inv_uv'] = Tensor(scale_matrix @ transform @ np.linalg.inv(scale_matrix))
    self.tensor_inputs['M_inv_wide'] = Tensor(transform_wide)
    self.tensor_inputs['M_inv_uv_wide'] = Tensor(scale_matrix @ transform_wide @ np.linalg.inv(scale_matrix))

    input_frame = Tensor(self.frame.array_from_vision_buf(buf))
    wide_input_frame = Tensor(self.wide_frame.array_from_vision_buf(wbuf))


    # PURE TG
    self.tensor_inputs['input_imgs'][:,:6] = self.tensor_inputs['input_imgs'][:,6:]
    self.tensor_inputs['input_imgs'][:,6:] = frame_prepare_tinygrad(input_frame, self.tensor_inputs['M_inv'], self.tensor_inputs['M_inv_uv'], buf.width, buf.height)

    self.tensor_inputs['big_input_imgs'][:,:6] = self.tensor_inputs['big_input_imgs'][:,6:]
    self.tensor_inputs['big_input_imgs'][:,6:] = frame_prepare_tinygrad(wide_input_frame, self.tensor_inputs['M_inv_wide'], self.tensor_inputs['M_inv_uv_wide'], wbuf.width, wbuf.height)


  # END OF PURE TG





    if prepare_only:
      return None

    print(len(self.tensor_inputs))
    print(len(set(self.tensor_inputs.values())))
    other_tensor_inputs = {k: v for k, v in self.tensor_inputs.items() if k not in ['M_inv', 'M_inv_uv', 'M_inv_wide', 'M_inv_uv_wide']}
    self.output = self.model_run(**other_tensor_inputs)['outputs'].numpy().flatten()
    outputs = self.parser.parse_outputs(self.slice_outputs(self.output))

    self.inputs['features_buffer'][:-ModelConstants.FEATURE_LEN] = self.inputs['features_buffer'][ModelConstants.FEATURE_LEN:]
    self.inputs['features_buffer'][-ModelConstants.FEATURE_LEN:] = outputs['hidden_state'][0, :]
    self.inputs['prev_desired_curv'][:-ModelConstants.PREV_DESIRED_CURV_LEN] = self.inputs['prev_desired_curv'][ModelConstants.PREV_DESIRED_CURV_LEN:]
    self.inputs['prev_desired_curv'][-ModelConstants.PREV_DESIRED_CURV_LEN:] = outputs['desired_curvature'][0, :]
    return outputs


def main(demo=False):
  cloudlog.warning("modeld init")

  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  config_realtime_process(7, 54)

  cloudlog.warning("setting up CL context")
  cl_context = CLContext()
  cloudlog.warning("CL context ready; loading model")
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
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  #TODO dont read car params for demo

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  steer_delay =  .2

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
    lateral_control_params = np.array([sm["carState"].vEgo, steer_delay], dtype=np.float32)
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

    inputs:dict[str, np.ndarray] = {
      'desire': vec_desire,
      'traffic_convention': traffic_convention,
      'lateral_control_params': lateral_control_params,
      }

    mt1 = time.perf_counter()
    model_output = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      fill_model_msg(drivingdata_send, modelv2_send, model_output, publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

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
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise
