import time
import logging
import numpy as np
import cereal.messaging as messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from common.filter_simple import FirstOrderFilter
from common.realtime import set_core_affinity, set_realtime_priority
from common.transformations.model import medmodel_frame_from_calib_frame, sbigmodel_frame_from_calib_frame
from common.transformations.camera import view_frame_from_device_frame, tici_fcam_intrinsics, tici_ecam_intrinsics
from common.transformations.orientation import rot_from_euler
from system.hardware import PC

# NOTE: These are almost exactly the same as the numbers in modeld.cc, but to get perfect equivalence we might have to copy them exactly
calib_from_medmodel = np.linalg.inv(medmodel_frame_from_calib_frame[:, :3])
calib_from_sbigmodel = np.linalg.inv(sbigmodel_frame_from_calib_frame[:, :3])

def update_calibration(device_from_calib_euler:np.ndarray, wide_camera:bool, bigmodel_frame:bool) -> np.ndarray:
  cam_intrinsics = tici_ecam_intrinsics if wide_camera else tici_fcam_intrinsics
  calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
  device_from_calib = rot_from_euler(device_from_calib_euler)
  camera_from_calib = cam_intrinsics @ view_frame_from_device_frame @ device_from_calib
  warp_matrix = camera_from_calib @ calib_from_model

  # TODO: Is this needed?
  transform = np.zeros((3, 3), dtype=np.float32)
  for i in range(3*3):
    transform[i] = warp_matrix[i//3, i%3]

  return transform


def run_model(vipc_client_main:VisionIpcClient, vipc_client_extra:VisionIpcClient, main_wide_camera:bool, use_extra_client:bool):
  # messaging
  pm = messaging.PubMaster(["modelV2", "cameraOdometry"])
  sm = messaging.SubMaster(["lateralPlan", "roadCameraState", "liveCalibration", "driverMonitoringState"])

  # setup filter to track dropped frames
  # TODO: I don't think the python version of FirstOrderFilter matches the c++ version exactly
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  last = 0.0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  driving_style = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
  nav_features = np.zeros(NAV_FEATURE_LEN, dtype=np.float32)

  buf_main = None
  buf_extra = None
  meta_main = VisionIpcBufExtra()
  meta_extra = VisionIpcBufExtra()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv(meta_main)
      if buf_main is None: break

    if buf_main is None:
      logging.error("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv(meta_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000: break

      if buf_extra is None:
        logging.error("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        logging.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}), extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    # TODO: path planner timeout?
    sm.update(0)
    desire = sm["lateralPlan"].desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    if sm.updated["liveCalibration"]:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib)
      model_transform_main = update_calibration(device_from_calib_euler, main_wide_camera, False)
      model_transform_extra = update_calibration(device_from_calib_euler, True, True)
      live_calib_seen = True

    vec_desire = np.zeros(DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = meta_main.frame_id - last_vipc_frame_id - 1
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.reset(0)
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      logging.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    """
    mt1 = millis_since_boot()
    model_output = model_eval_frame(&model, buf_main, buf_extra, model_transform_main, model_transform_extra, vec_desire, is_rhd, driving_style, nav_features, prepare_only);
    mt2 = millis_since_boot()
    """
    model_execution_time = (mt2 - mt1) / 1000.0

    """
    if model_output:
      model_publish(pm, meta_main.frame_id, meta_extra.frame_id, frame_id, frame_drop_ratio, *model_output, meta_main.timestamp_eof, model_execution_time,
                    kj::ArrayPtr<const float>(model.output.data(), model.output.size()), live_calib_seen)
      posenet_publish(pm, meta_main.frame_id, vipc_dropped_frames, *model_output, meta_main.timestamp_eof, live_calib_seen)
    """

    # print("model process: %.2fms, from last %.2fms, vipc_frame_id %u, frame_id, %u, frame_drop %.3f\n" % (mt2 - mt1, mt1 - last, extra.frame_id, frame_id, frame_drop_ratio))
    last = mt1
    last_vipc_frame_id = meta_main.frame_id


if __name__ == '__main__':
  if not PC:
    set_realtime_priority(54)
    set_core_affinity([7])

  """
  # cl init
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

  # init the models
  ModelState model;
  model_init(&model, device_id, context);
  """
  logging.warning("models loaded, modeld starting")

  main_wide_camera = False
  use_extra_client = True  # set to false to use single camera

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
  logging.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  # TODO: Is it safe to use blocking=True here?
  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while not vipc_client_extra.connect(False):
    time.sleep(0.1)

  logging.warning(f"connected main cam with buffer size: {vipc_client_main.size()} ({vipc_client_main.width()} x {vipc_client_main.height()})")
  if use_extra_client:
    logging.warning(f"connected extra cam with buffer size: {vipc_client_extra.size()} ({vipc_client_extra.width()} x {vipc_client_extra.height()})")

  run_model(model, vipc_client_main, vipc_client_extra, main_wide_camera, use_extra_client)

  """
  model_free(&model);
  CL_CHECK(clReleaseContext(context));
  """
