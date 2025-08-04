#!/usr/bin/env python3
import argparse
import os
import sys

import cv2
import numpy as np
import pygame

import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.tools.replay.lib.ui_helpers import (UP,
                                         BLACK, GREEN,
                                         YELLOW, Calibration,
                                         get_blank_lid_overlay, init_plots,
                                         maybe_update_radar_points, plot_lead,
                                         plot_model,
                                         pygame_modules_have_loaded)
from msgq.visionipc import VisionIpcClient, VisionStreamType

os.environ['BASEDIR'] = BASEDIR

ANGLE_SCALE = 5.0

def ui_thread(addr):
  cv2.setNumThreads(1)
  pygame.init()
  pygame.font.init()
  assert pygame_modules_have_loaded()

  disp_info = pygame.display.Info()
  max_height = disp_info.current_h

  hor_mode = os.getenv("HORIZONTAL") is not None
  hor_mode = True if max_height < 960+300 else hor_mode

  if hor_mode:
    size = (640+384+640, 960)
    write_x = 5
    write_y = 680
  else:
    size = (640+384, 960+300)
    write_x = 645
    write_y = 970

  pygame.display.set_caption("openpilot debug UI")
  screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

  alert1_font = pygame.font.SysFont("arial", 30)
  alert2_font = pygame.font.SysFont("arial", 20)
  info_font = pygame.font.SysFont("arial", 15)

  camera_surface = pygame.surface.Surface((640, 480), 0, 24).convert()
  top_down_surface = pygame.surface.Surface((UP.lidar_x, UP.lidar_y), 0, 8)

  sm = messaging.SubMaster(['carState', 'longitudinalPlan', 'carControl', 'radarState', 'liveCalibration', 'controlsState',
                            'selfdriveState', 'liveTracks', 'modelV2', 'liveParameters', 'roadCameraState'], addr=addr)

  img = np.zeros((480, 640, 3), dtype='uint8')
  imgff = None
  num_px = 0
  calibration = None

  lid_overlay_blank = get_blank_lid_overlay(UP)

  # plots
  name_to_arr_idx = { "gas": 0,
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
                      "a_target": 13}

  plot_arr = np.zeros((100, len(name_to_arr_idx.values())))

  plot_xlims = [(0, plot_arr.shape[0]), (0, plot_arr.shape[0]), (0, plot_arr.shape[0]), (0, plot_arr.shape[0])]
  plot_ylims = [(-0.1, 1.1), (-ANGLE_SCALE, ANGLE_SCALE), (0., 75.), (-3.0, 2.0)]
  plot_names = [["gas", "computer_gas", "user_brake", "computer_brake"],
                ["angle_steers", "angle_steers_des", "angle_steers_k", "steer_torque"],
                ["v_ego", "v_override", "v_pid", "v_cruise"],
                ["a_ego", "a_target"]]
  plot_colors = [["b", "b", "g", "r", "y"],
                 ["b", "g", "y", "r"],
                 ["b", "g", "r", "y"],
                 ["b", "r"]]
  plot_styles = [["-", "-", "-", "-", "-"],
                 ["-", "-", "-", "-"],
                 ["-", "-", "-", "-"],
                 ["-", "-"]]

  draw_plots = init_plots(plot_arr, name_to_arr_idx, plot_xlims, plot_ylims, plot_names, plot_colors, plot_styles)

  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()

    screen.fill((64, 64, 64))
    lid_overlay = lid_overlay_blank.copy()
    top_down = top_down_surface, lid_overlay

    # ***** frame *****
    if not vipc_client.is_connected():
      vipc_client.connect(True)

    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any():
      continue

    sm.update(0)

    camera = DEVICE_CAMERAS[("tici", str(sm['roadCameraState'].sensor))]

    imgff = np.frombuffer(yuv_img_raw.data, dtype=np.uint8).reshape((len(yuv_img_raw.data) // vipc_client.stride, vipc_client.stride))
    num_px = vipc_client.width * vipc_client.height
    rgb = cv2.cvtColor(imgff[:vipc_client.height * 3 // 2, :vipc_client.width], cv2.COLOR_YUV2RGB_NV12)

    qcam = "QCAM" in os.environ
    bb_scale = (528 if qcam else camera.fcam.width) / 640.
    calib_scale = camera.fcam.width / 640.
    zoom_matrix = np.asarray([
        [bb_scale, 0., 0.],
        [0., bb_scale, 0.],
        [0., 0., 1.]])
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
    plot_arr[-1, name_to_arr_idx['computer_gas']] = np.clip(sm['carControl'].actuators.accel/4.0, 0.0, 1.0)
    plot_arr[-1, name_to_arr_idx['user_brake']] = sm['carState'].brake
    plot_arr[-1, name_to_arr_idx['steer_torque']] = sm['carControl'].actuators.torque * ANGLE_SCALE
    # TODO brake is deprecated
    plot_arr[-1, name_to_arr_idx['computer_brake']] = np.clip(-sm['carControl'].actuators.accel/4.0, 0.0, 1.0)
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
    pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
    screen.blit(camera_surface, (0, 0))

    # display alerts
    alert_line1 = alert1_font.render(sm['selfdriveState'].alertText1, True, (255, 0, 0))
    alert_line2 = alert2_font.render(sm['selfdriveState'].alertText2, True, (255, 0, 0))
    screen.blit(alert_line1, (180, 150))
    screen.blit(alert_line2, (180, 190))

    if hor_mode:
      screen.blit(draw_plots(plot_arr), (640+384, 0))
    else:
      screen.blit(draw_plots(plot_arr), (0, 600))

    pygame.surfarray.blit_array(*top_down)
    screen.blit(top_down[0], (640, 0))

    SPACING = 25

    lines = [
      info_font.render("ENABLED", True, GREEN if sm['selfdriveState'].enabled else BLACK),
      info_font.render("SPEED: " + str(round(sm['carState'].vEgo, 1)) + " m/s", True, YELLOW),
      info_font.render("LONG CONTROL STATE: " + str(sm['controlsState'].longControlState), True, YELLOW),
      info_font.render("LONG MPC SOURCE: " + str(sm['longitudinalPlan'].longitudinalPlanSource), True, YELLOW),
      None,
      info_font.render("ANGLE OFFSET (AVG): " + str(round(sm['liveParameters'].angleOffsetAverageDeg, 2)) + " deg", True, YELLOW),
      info_font.render("ANGLE OFFSET (INSTANT): " + str(round(sm['liveParameters'].angleOffsetDeg, 2)) + " deg", True, YELLOW),
      info_font.render("STIFFNESS: " + str(round(sm['liveParameters'].stiffnessFactor * 100., 2)) + " %", True, YELLOW),
      info_font.render("STEER RATIO: " + str(round(sm['liveParameters'].steerRatio, 2)), True, YELLOW)
    ]

    for i, line in enumerate(lines):
      if line is not None:
        screen.blit(line, (write_x, write_y + i * SPACING))

    # this takes time...vsync or something
    pygame.display.flip()

def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Show replay data in a UI.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("ip_address", nargs="?", default="127.0.0.1",
                      help="The ip address on which to receive zmq messages.")

  parser.add_argument("--frame-address", default=None,
                      help="The frame address (fully qualified ZMQ endpoint for frames) on which to receive zmq messages.")
  return parser

if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])

  if args.ip_address != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.reset_context()

  ui_thread(args.ip_address)
