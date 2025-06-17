#!/usr/bin/env python3
import argparse

import numpy as np
import matplotlib.pyplot as plt

from openpilot.tools.lib.logreader import LogReader

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--width', default=2160, type=int)
  parser.add_argument('--height', default=1080, type=int)
  parser.add_argument('--route', default='rlog', type=str)
  args = parser.parse_args()

  w = args.width
  h = args.height
  route = args.route

  fingers = [[-1, -1]] * 5
  touch_points = []
  current_slot = 0

  lr = list(LogReader(route))
  for msg in lr:
    if msg.which() == 'touch':
      for event in msg.touch:
        if event.type == 3 and event.code == 47:
          current_slot = event.value
        elif event.type == 3 and event.code == 57 and event.value == -1:
          fingers[current_slot] = [-1, -1]
        elif event.type == 3 and event.code == 53:
          fingers[current_slot][1] = event.value
          if fingers[current_slot][0] != -1:
            touch_points.append(fingers[current_slot].copy())
        elif event.type == 3 and event.code == 54:
          fingers[current_slot][0] = w - event.value
          if fingers[current_slot][1] != -1:
            touch_points.append(fingers[current_slot].copy())

  if not touch_points:
    print(f'No touch events found for {route}')
    quit()

  unique_points, counts = np.unique(touch_points, axis=0, return_counts=True)

  plt.figure(figsize=(10, 3))
  plt.scatter(unique_points[:, 0], unique_points[:, 1], c=counts, s=counts * 20, edgecolors='red')
  plt.colorbar()
  plt.title(f'Touches for {route}')
  plt.xlim(0, w)
  plt.ylim(0, h)
  plt.grid(True)
  plt.show()
