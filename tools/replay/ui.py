#!/usr/bin/env python3
import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

import cv2  # pylint: disable=import-error
import numpy as np
import pygame  # pylint: disable=import-error

import cereal.messaging as messaging
from common.numpy_fast import clip
from common.basedir import BASEDIR
from selfdrive.config import UIParams as UP
from tools.replay.lib.ui_helpers import (_BB_TO_FULL_FRAME, _FULL_FRAME_SIZE,
                                         _INTRINSICS, BLACK, GREEN,
                                         YELLOW, Calibration,
                                         get_blank_lid_overlay, init_plots,
                                         maybe_update_radar_points, plot_lead,
                                         plot_model,
                                         pygame_modules_have_loaded)

os.environ['BASEDIR'] = BASEDIR

ANGLE_SCALE = 5.0

def ui_thread(addr, frame_address):
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

  frame = messaging.sub_sock('roadCameraState', addr=addr, conflate=True)
  sm = messaging.SubMaster(['carState', 'longitudinalPlan', 'carControl', 'radarState', 'liveCalibration', 'controlsState',
                            'liveTracks', 'modelV2', 'liveParameters', 'lateralPlan', 'roadCameraState'], addr=addr)

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

  draw_plots = init_plots(plot_arr, name_to_arr_idx, plot_xlims, plot_ylims, plot_names, plot_colors, plot_styles, bigplots=True)

  while 1:
    list(pygame.event.get())

    screen.fill((64, 64, 64))
    lid_overlay = lid_overlay_blank.copy()
    top_down = top_down_surface, lid_overlay

    # ***** frame *****
    fpkt = messaging.recv_one(frame)
    rgb_img_raw = fpkt.roadCameraState.image

    num_px = len(rgb_img_raw) // 3
    if rgb_img_raw and num_px in _FULL_FRAME_SIZE.keys():
      FULL_FRAME_SIZE = _FULL_FRAME_SIZE[num_px]

      imgff_shape = (FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3)

      if imgff is None or imgff.shape != imgff_shape:
        imgff = np.zeros(imgff_shape, dtype=np.uint8)

      imgff = np.frombuffer(rgb_img_raw, dtype=np.uint8).reshape((FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3))
      imgff = imgff[:, :, ::-1]  # Convert BGR to RGB
      zoom_matrix = _BB_TO_FULL_FRAME[num_px]
      cv2.warpAffine(imgff, zoom_matrix[:2], (img.shape[1], img.shape[0]), dst=img, flags=cv2.WARP_INVERSE_MAP)

      intrinsic_matrix = _INTRINSICS[num_px]
    else:
      img.fill(0)
      intrinsic_matrix = np.eye(3)

    sm.update()

    w = sm['controlsState'].lateralControlState.which()
    if w == 'lqrState':
      angle_steers_k = sm['controlsState'].lateralControlState.lqrState.steeringAngleDeg
    elif w == 'indiState':
      angle_steers_k = sm['controlsState'].lateralControlState.indiState.steeringAngleDeg
    else:
      angle_steers_k = np.inf

    plot_arr[:-1] = plot_arr[1:]
    plot_arr[-1, name_to_arr_idx['angle_steers']] = sm['carState'].steeringAngleDeg
    plot_arr[-1, name_to_arr_idx['angle_steers_des']] = sm['carControl'].actuators.steeringAngleDeg
    plot_arr[-1, name_to_arr_idx['angle_steers_k']] = angle_steers_k
    plot_arr[-1, name_to_arr_idx['gas']] = sm['carState'].gas
    # TODO gas is deprecated
    plot_arr[-1, name_to_arr_idx['computer_gas']] = clip(sm['carControl'].actuators.accel/4.0, 0.0, 1.0)
    plot_arr[-1, name_to_arr_idx['user_brake']] = sm['carState'].brake
    plot_arr[-1, name_to_arr_idx['steer_torque']] = sm['carControl'].actuators.steer * ANGLE_SCALE
    # TODO brake is deprecated
    plot_arr[-1, name_to_arr_idx['computer_brake']] = clip(-sm['carControl'].actuators.accel/4.0, 0.0, 1.0)
    plot_arr[-1, name_to_arr_idx['v_ego']] = sm['carState'].vEgo
    plot_arr[-1, name_to_arr_idx['v_pid']] = sm['controlsState'].vPid
    plot_arr[-1, name_to_arr_idx['v_cruise']] = sm['carState'].cruiseState.speed
    plot_arr[-1, name_to_arr_idx['a_ego']] = sm['carState'].aEgo

    if len(sm['longitudinalPlan'].accels):
      plot_arr[-1, name_to_arr_idx['a_target']] = sm['longitudinalPlan'].accels[0]

    if sm.rcv_frame['modelV2']:
      plot_model(sm['modelV2'], img, calibration, top_down)

    if sm.rcv_frame['radarState']:
      plot_lead(sm['radarState'], top_down)

    # draw all radar points
    maybe_update_radar_points(sm['liveTracks'], top_down[1])

    if sm.updated['liveCalibration'] and num_px:
      rpyCalib = np.asarray(sm['liveCalibration'].rpyCalib)
      calibration = Calibration(num_px, rpyCalib, intrinsic_matrix)

    # *** blits ***
    pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
    screen.blit(camera_surface, (0, 0))

    # display alerts
    alert_line1 = alert1_font.render(sm['controlsState'].alertText1, True, (255, 0, 0))
    alert_line2 = alert2_font.render(sm['controlsState'].alertText2, True, (255, 0, 0))
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
      info_font.render("ENABLED", True, GREEN if sm['controlsState'].enabled else BLACK),
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
    messaging.context = messaging.Context()

  ui_thread(args.ip_address, args.frame_address)
