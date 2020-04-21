#!/usr/bin/env python
import os

from common.basedir import BASEDIR
os.environ['BASEDIR'] = BASEDIR
SCALE = 1

import argparse
import zmq
import pygame
import numpy as np
import cv2
import sys
import traceback
from collections import namedtuple
from cereal import car
from common.params import Params
from tools.lib.lazy_property import lazy_property
from cereal.messaging import sub_sock, recv_one_or_none, recv_one
from cereal.services import service_list
import cereal.messaging as messaging

_BB_OFFSET = 0, 0
_BB_TO_FULL_FRAME = np.asarray([[1., 0., _BB_OFFSET[0]], [0., 1., _BB_OFFSET[1]],
                                [0., 0., 1.]])
_FULL_FRAME_TO_BB = np.linalg.inv(_BB_TO_FULL_FRAME)
_FULL_FRAME_SIZE = 1164, 874



def pygame_modules_have_loaded():
  return pygame.display.get_init() and pygame.font.get_init()


def ui_thread(addr, frame_address):
  context = zmq.Context.instance()

  pygame.init()
  pygame.font.init()
  assert pygame_modules_have_loaded()

  size = (_FULL_FRAME_SIZE[0] * SCALE, _FULL_FRAME_SIZE[1] * SCALE)
  pygame.display.set_caption("comma one debug UI")
  screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

  camera_surface = pygame.surface.Surface((_FULL_FRAME_SIZE[0] * SCALE, _FULL_FRAME_SIZE[1] * SCALE), 0, 24).convert()

  frame = messaging.sub_sock('frame', conflate=True)

  img = np.zeros((_FULL_FRAME_SIZE[1], _FULL_FRAME_SIZE[0], 3), dtype='uint8')
  imgff = np.zeros((_FULL_FRAME_SIZE[1], _FULL_FRAME_SIZE[0], 3), dtype=np.uint8)

  while 1:
    list(pygame.event.get())
    screen.fill((64, 64, 64))

    # ***** frame *****
    fpkt = messaging.recv_one(frame)
    yuv_img = fpkt.frame.image

    if fpkt.frame.transform:
      yuv_transform = np.array(fpkt.frame.transform).reshape(3, 3)
    else:
      # assume frame is flipped
      yuv_transform = np.array([[-1.0, 0.0, _FULL_FRAME_SIZE[0] - 1],
                                [0.0, -1.0, _FULL_FRAME_SIZE[1] - 1], [0.0, 0.0, 1.0]])

    if yuv_img and len(yuv_img) == _FULL_FRAME_SIZE[0] * _FULL_FRAME_SIZE[1] * 3 // 2:
      yuv_np = np.frombuffer(
        yuv_img, dtype=np.uint8).reshape(_FULL_FRAME_SIZE[1] * 3 // 2, -1)
      cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420, dst=imgff)
      cv2.warpAffine(
        imgff,
        np.dot(yuv_transform, _BB_TO_FULL_FRAME)[:2], (img.shape[1], img.shape[0]),
        dst=img,
        flags=cv2.WARP_INVERSE_MAP)
    else:
      # actually RGB
      img = np.frombuffer(yuv_img, dtype=np.uint8).reshape((_FULL_FRAME_SIZE[1], _FULL_FRAME_SIZE[0], 3))
      img = img[:, :, ::-1] # Convert BGR to RGB

    height, width = img.shape[:2]
    img_resized = cv2.resize(
      img, (SCALE * width, SCALE * height), interpolation=cv2.INTER_CUBIC)
    # *** blits ***
    pygame.surfarray.blit_array(camera_surface, img_resized.swapaxes(0, 1))
    screen.blit(camera_surface, (0, 0))

    # this takes time...vsync or something
    pygame.display.flip()


def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Show replay data in a UI.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
    "ip_address",
    nargs="?",
    default="127.0.0.1",
    help="The ip address on which to receive zmq messages.")

  parser.add_argument(
    "--frame-address",
    default=None,
    help="The ip address on which to receive zmq messages.")
  return parser


if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])
  ui_thread(args.ip_address, args.frame_address)
