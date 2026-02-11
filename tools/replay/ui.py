#!/usr/bin/env python3
import argparse
import os
import sys

import cv2
import numpy as np
import pyray as rl

import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.tools.replay.lib.ui_helpers import (
  UP,
  BLACK,
  GREEN,
  YELLOW,
  Calibration,
  get_blank_lid_overlay,
  init_plots,
  maybe_update_radar_points,
  plot_lead,
  plot_model,
)
from msgq.visionipc import VisionIpcClient, VisionStreamType

os.environ['BASEDIR'] = BASEDIR

ANGLE_SCALE = 5.0


def ui_thread(addr):
  cv2.setNumThreads(1)

  # Get monitor info before creating window
  rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)
  rl.init_window(1, 1, "")
  max_height = rl.get_monitor_height(0)
  rl.close_window()

  hor_mode = os.getenv("HORIZONTAL") is not None
  hor_mode = True if max_height < 960 + 300 else hor_mode

  if hor_mode:
    size = (640 + 384 + 640, 960)
    write_x = 5
    write_y = 680
  else:
    size = (640 + 384, 960 + 300)
    write_x = 645
    write_y = 970

  rl.set_trace_log_level(rl.TraceLogLevel.LOG_ERROR)
  rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)
  rl.init_window(size[0], size[1], "openpilot debug UI")
  rl.set_target_fps(60)

  # Load font
  font_path = os.path.join(BASEDIR, "selfdrive/assets/fonts/JetBrainsMono-Medium.ttf")
  font = rl.load_font_ex(font_path, 32, None, 0)

  # Create textures for camera and top-down view
  camera_image = rl.gen_image_color(640, 480, rl.BLACK)
  camera_texture = rl.load_texture_from_image(camera_image)
  rl.unload_image(camera_image)

  # lid_overlay array is (lidar_x, lidar_y) = (384, 960)
  # pygame treats first axis as width, so texture is 384 wide x 960 tall
  # For raylib, we need to transpose to get (height, width) = (960, 384) for the RGBA array
  top_down_image = rl.gen_image_color(UP.lidar_x, UP.lidar_y, rl.BLACK)
  top_down_texture = rl.load_texture_from_image(top_down_image)
  rl.unload_image(top_down_image)

  sm = messaging.SubMaster(
    [
      'carState',
      'longitudinalPlan',
      'carControl',
      'radarState',
      'liveCalibration',
      'controlsState',
      'selfdriveState',
      'liveTracks',
      'modelV2',
      'liveParameters',
      'roadCameraState',
    ],
    addr=addr,
  )

  img = np.zeros((480, 640, 3), dtype='uint8')
  imgff = None
  num_px = 0
  calibration = None

  lid_overlay_blank = get_blank_lid_overlay(UP)

  # plots
  name_to_arr_idx = {
    "gas": 0,
    "computer_gas": 1,
    "user_brake": 2,
    "computer_brake": 3,
    "v_ego": 4,
    "v_pid": 5,
    "angle_steers_des": 6,
    "angle_steers": 7,
    "angle_steers_k": 8,
    "steer_torque": 9,
    "v_override": 10,
    "v_cruise": 11,
    "a_ego": 12,
    "a_target": 13,
  }

  plot_arr = np.zeros((100, len(name_to_arr_idx.values())))

  plot_xlims = [(0, plot_arr.shape[0]), (0, plot_arr.shape[0]), (0, plot_arr.shape[0]), (0, plot_arr.shape[0])]
  plot_ylims = [(-0.1, 1.1), (-ANGLE_SCALE, ANGLE_SCALE), (0.0, 75.0), (-3.0, 2.0)]
  plot_names = [
    ["gas", "computer_gas", "user_brake", "computer_brake"],
    ["angle_steers", "angle_steers_des", "angle_steers_k", "steer_torque"],
    ["v_ego", "v_override", "v_pid", "v_cruise"],
    ["a_ego", "a_target"],
  ]
  plot_colors = [["b", "b", "g", "r", "y"], ["b", "g", "y", "r"], ["b", "g", "r", "y"], ["b", "r"]]
  plot_styles = [["-", "-", "-", "-", "-"], ["-", "-", "-", "-"], ["-", "-", "-", "-"], ["-", "-"]]

  draw_plots = init_plots(plot_arr, name_to_arr_idx, plot_xlims, plot_ylims, plot_names, plot_colors, plot_styles)

  # Palette for converting lid_overlay grayscale indices to RGBA colors
  palette = np.zeros((256, 4), dtype=np.uint8)
  palette[:, 3] = 255  # alpha
  palette[1] = [255, 0, 0, 255]  # RED
  palette[2] = [0, 255, 0, 255]  # GREEN
  palette[3] = [0, 0, 255, 255]  # BLUE
  palette[4] = [255, 255, 0, 255]  # YELLOW
  palette[110] = [110, 110, 110, 255]  # car_color (gray)
  palette[255] = [255, 255, 255, 255]  # WHITE

  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while not rl.window_should_close():
    # ***** frame *****
    if not vipc_client.is_connected():
      vipc_client.connect(False)

    rl.begin_drawing()
    rl.clear_background(rl.Color(64, 64, 64, 255))

    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any():
      rl.draw_text_ex(font, "waiting for frames", rl.Vector2(200, 200), 30, 0, rl.WHITE)
      rl.end_drawing()
      continue

    lid_overlay = lid_overlay_blank.copy()
    top_down = top_down_texture, lid_overlay

    sm.update(0)

    camera = DEVICE_CAMERAS[("tici", str(sm['roadCameraState'].sensor))]

    # Use received buffer dimensions (full HEVC can have stride != buffer_len/rows due to VENUS padding)
    h, w, stride = yuv_img_raw.height, yuv_img_raw.width, yuv_img_raw.stride
    nv12_size = h * 3 // 2 * stride
    imgff = np.frombuffer(yuv_img_raw.data, dtype=np.uint8, count=nv12_size).reshape((h * 3 // 2, stride))
    num_px = w * h
    rgb = cv2.cvtColor(imgff[: h * 3 // 2, : w], cv2.COLOR_YUV2RGB_NV12)

    qcam = "QCAM" in os.environ
    bb_scale = (528 if qcam else camera.fcam.width) / 640.0
    calib_scale = camera.fcam.width / 640.0
    zoom_matrix = np.asarray([[bb_scale, 0.0, 0.0], [0.0, bb_scale, 0.0], [0.0, 0.0, 1.0]])
    cv2.warpAffine(rgb, zoom_matrix[:2], (img.shape[1], img.shape[0]), dst=img, flags=cv2.WARP_INVERSE_MAP)

    intrinsic_matrix = camera.fcam.intrinsics

    w = sm['controlsState'].lateralControlState.which()
    if w == 'lqrStateDEPRECATED':
      angle_steers_k = sm['controlsState'].lateralControlState.lqrStateDEPRECATED.steeringAngleDeg
    elif w == 'indiState':
      angle_steers_k = sm['controlsState'].lateralControlState.indiState.steeringAngleDeg
    else:
      angle_steers_k = np.inf

    plot_arr[:-1] = plot_arr[1:]
    plot_arr[-1, name_to_arr_idx['angle_steers']] = sm['carState'].steeringAngleDeg
    plot_arr[-1, name_to_arr_idx['angle_steers_des']] = sm['carControl'].actuators.steeringAngleDeg
    plot_arr[-1, name_to_arr_idx['angle_steers_k']] = angle_steers_k
    plot_arr[-1, name_to_arr_idx['gas']] = sm['carState'].gasDEPRECATED
    # TODO gas is deprecated
    plot_arr[-1, name_to_arr_idx['computer_gas']] = np.clip(sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
    plot_arr[-1, name_to_arr_idx['user_brake']] = sm['carState'].brake
    plot_arr[-1, name_to_arr_idx['steer_torque']] = sm['carControl'].actuators.torque * ANGLE_SCALE
    # TODO brake is deprecated
    plot_arr[-1, name_to_arr_idx['computer_brake']] = np.clip(-sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
    plot_arr[-1, name_to_arr_idx['v_ego']] = sm['carState'].vEgo
    plot_arr[-1, name_to_arr_idx['v_cruise']] = sm['carState'].cruiseState.speed
    plot_arr[-1, name_to_arr_idx['a_ego']] = sm['carState'].aEgo

    if len(sm['longitudinalPlan'].accels):
      plot_arr[-1, name_to_arr_idx['a_target']] = sm['longitudinalPlan'].accels[0]

    if sm.recv_frame['modelV2']:
      plot_model(sm['modelV2'], img, calibration, top_down)

    if sm.recv_frame['radarState']:
      plot_lead(sm['radarState'], top_down)

    # draw all radar points
    maybe_update_radar_points(sm['liveTracks'].points, top_down[1])

    if sm.updated['liveCalibration'] and num_px:
      rpyCalib = np.asarray(sm['liveCalibration'].rpyCalib)
      calibration = Calibration(num_px, rpyCalib, intrinsic_matrix, calib_scale)

    # *** blits ***
    # Update camera texture from numpy array
    img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    rl.update_texture(camera_texture, rl.ffi.cast("void *", img_rgba.ctypes.data))
    rl.draw_texture(camera_texture, 0, 0, rl.WHITE)

    # display alerts
    rl.draw_text_ex(font, sm['selfdriveState'].alertText1, rl.Vector2(180, 150), 30, 0, rl.RED)
    rl.draw_text_ex(font, sm['selfdriveState'].alertText2, rl.Vector2(180, 190), 20, 0, rl.RED)

    # draw plots (texture is reused internally)
    plot_texture = draw_plots(plot_arr)
    if hor_mode:
      rl.draw_texture(plot_texture, 640 + 384, 0, rl.WHITE)
    else:
      rl.draw_texture(plot_texture, 0, 600, rl.WHITE)

    # Convert lid_overlay to RGBA and update top_down texture
    # lid_overlay is (384, 960), need to transpose to (960, 384) for row-major RGBA buffer
    lid_rgba = palette[lid_overlay.T]
    rl.update_texture(top_down_texture, rl.ffi.cast("void *", np.ascontiguousarray(lid_rgba).ctypes.data))
    rl.draw_texture(top_down_texture, 640, 0, rl.WHITE)

    SPACING = 25
    lines = [
      ("ENABLED", GREEN if sm['selfdriveState'].enabled else BLACK),
      ("SPEED: " + str(round(sm['carState'].vEgo, 1)) + " m/s", YELLOW),
      ("LONG CONTROL STATE: " + str(sm['controlsState'].longControlState), YELLOW),
      ("LONG MPC SOURCE: " + str(sm['longitudinalPlan'].longitudinalPlanSource), YELLOW),
      None,
      ("ANGLE OFFSET (AVG): " + str(round(sm['liveParameters'].angleOffsetAverageDeg, 2)) + " deg", YELLOW),
      ("ANGLE OFFSET (INSTANT): " + str(round(sm['liveParameters'].angleOffsetDeg, 2)) + " deg", YELLOW),
      ("STIFFNESS: " + str(round(sm['liveParameters'].stiffnessFactor * 100.0, 2)) + " %", YELLOW),
      ("STEER RATIO: " + str(round(sm['liveParameters'].steerRatio, 2)), YELLOW),
    ]

    for i, line in enumerate(lines):
      if line is not None:
        color = rl.Color(line[1][0], line[1][1], line[1][2], 255)
        rl.draw_text_ex(font, line[0], rl.Vector2(write_x, write_y + i * SPACING), 20, 0, color)

    rl.end_drawing()

  rl.unload_texture(camera_texture)
  rl.unload_texture(top_down_texture)
  rl.unload_font(font)
  rl.close_window()


def get_arg_parser():
  parser = argparse.ArgumentParser(description="Show replay data in a UI.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("ip_address", nargs="?", default="127.0.0.1", help="The ip address on which to receive zmq messages.")

  parser.add_argument("--frame-address", default=None, help="The frame address (fully qualified ZMQ endpoint for frames) on which to receive zmq messages.")
  return parser


if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])

  if args.ip_address != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.reset_context()

  ui_thread(args.ip_address)
