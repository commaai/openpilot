#!/usr/bin/env python3
import os
import argparse
import pygame  # pylint: disable=import-error
import numpy as np
import cv2  # pylint: disable=import-error

from cereal import log
import cereal.messaging as messaging

from helpers import draw_pose

if __name__ == "__main__":

  os.environ["ZMQ"] = "1"

  parser = argparse.ArgumentParser(description='Sniff a communcation socket')
  parser.add_argument('--addr', default='192.168.5.11')
  args = parser.parse_args()

  messaging.context = messaging.Context()

  poller = messaging.Poller()

  m = 'driverMonitoring'
  messaging.sub_sock(m, poller, addr=args.addr)

  pygame.init()
  pygame.display.set_caption('livedm')
  screen = pygame.display.set_mode((320, 640), pygame.DOUBLEBUF)
  camera_surface = pygame.surface.Surface((160, 320), 0, 24).convert()

  while 1:
    polld = poller.poll(1000)
    for sock in polld:
      msg = sock.receive()
      evt = log.Event.from_bytes(msg)

      faceProb = np.array(evt.driverMonitoring.faceProb)
      faceOrientation = np.array(evt.driverMonitoring.faceOrientation)
      facePosition = np.array(evt.driverMonitoring.facePosition)

      print(faceProb)
      # print(faceOrientation)
      # print(facePosition)
      faceOrientation[1] *= -1
      facePosition[0] *= -1

      img = np.zeros((320, 160, 3))
      if faceProb > 0.4:
        cv2.putText(img, 'you', (int(facePosition[0]*160+40), int(facePosition[1]*320+110)), cv2.FONT_ITALIC, 0.5, (255, 255, 0))
        cv2.rectangle(img, (int(facePosition[0]*160+40), int(facePosition[1]*320+120)),\
                    (int(facePosition[0]*160+120), int(facePosition[1]*320+200)), (255, 255, 0), 1)

        not_blink = evt.driverMonitoring.leftBlinkProb + evt.driverMonitoring.rightBlinkProb < 1

        if evt.driverMonitoring.leftEyeProb > 0.6:
          cv2.line(img, (int(facePosition[0]*160+95), int(facePosition[1]*320+140)),\
                      (int(facePosition[0]*160+105), int(facePosition[1]*320+140)), (255, 255, 0), 2)
          if not_blink:
            cv2.line(img, (int(facePosition[0]*160+99), int(facePosition[1]*320+143)),\
                      (int(facePosition[0]*160+101), int(facePosition[1]*320+143)), (255, 255, 0), 2)

        if evt.driverMonitoring.rightEyeProb > 0.6:
          cv2.line(img, (int(facePosition[0]*160+55), int(facePosition[1]*320+140)),\
                      (int(facePosition[0]*160+65), int(facePosition[1]*320+140)), (255, 255, 0), 2)
          if not_blink:
            cv2.line(img, (int(facePosition[0]*160+59), int(facePosition[1]*320+143)),\
                      (int(facePosition[0]*160+61), int(facePosition[1]*320+143)), (255, 255, 0), 2)

      else:
        cv2.putText(img, 'you not found', (int(facePosition[0]*160+40), int(facePosition[1]*320+110)), cv2.FONT_ITALIC, 0.5, (64, 64, 64))
      draw_pose(img, faceOrientation, facePosition,
              W = 160, H = 320, xyoffset = (0, 0), faceprob=faceProb)

      pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
      camera_surface_2x = pygame.transform.scale2x(camera_surface)
      screen.blit(camera_surface_2x, (0, 0))
      pygame.display.flip()
