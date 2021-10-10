#!/usr/bin/env python3
import os

SCALE = float(os.getenv("SCALE", "1"))

import argparse
import pygame  # pylint: disable=import-error
import numpy as np
import cv2  # pylint: disable=import-error
import sys
from cereal.visionipc.visionipc_pyx import VisionIpcClient, VisionStreamType # pylint: disable=no-name-in-module, import-error

def pygame_modules_have_loaded():
  return pygame.display.get_init() and pygame.font.get_init()

def ui_thread():
  pygame.init()
  pygame.font.init()
  assert pygame_modules_have_loaded()

  pygame.display.set_caption("comma one debug UI")
  screen = None
  camera_surface = None
  width = 0
  height = 0

  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_RGB_BACK, True)
  while 1:
    list(pygame.event.get())
    # ***** frame *****
    if not vipc_client.is_connected():
      vipc_client.connect(True)

    rgb_img_raw = vipc_client.recv()
    if rgb_img_raw is None or not rgb_img_raw.any():
      continue

    if width != vipc_client.width or height != vipc_client.height:
      width = vipc_client.width
      height = vipc_client.height
      size = (int(width * SCALE), int(height * SCALE)) 
      screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
      camera_surface = pygame.surface.Surface(size, 0, 24).convert()
  
    img = np.frombuffer(rgb_img_raw, dtype=np.uint8).reshape((height, width, 3))
    img = img[:, :, ::-1]  # Convert BGR to RGB

    if (SCALE != 1.0):
      img = cv2.resize(
        img, (int(SCALE * width), int(SCALE * height)), interpolation=cv2.INTER_CUBIC)
    # *** blits ***
    pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
    screen.blit(camera_surface, (0, 0))

    # this takes time...vsync or something
    pygame.display.flip()


def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Show replay data in a UI.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  return parser


if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])
  ui_thread()
