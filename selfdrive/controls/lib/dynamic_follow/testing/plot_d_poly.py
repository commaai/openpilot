# import matplotlib
# matplotlib.use('Qt5Agg')
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import time

os.chdir(os.getcwd())

data = 'lane_speed3'

# good ones 2: 134000, 107823 + 900, 98708, 98708+9000, 117734
# good ones 3: 167000 + int(4800), 167000 + 4800 + 5400, 110512+5200, 237541+2500, 58000+750+2000
start = 58000+750+2000

with open(data, 'r') as f:
  data = f.read().split('\n')[:-1][start:start+10000]

data_parsed = []
for idx, line in enumerate(data):
  if 'nan' in line:
    continue
  line = line.replace('builder', 'reader').replace('<capnp list reader ', '').replace('>', '')
  line = line.replace('array(', '').replace('), ', ', ')
  try:
    line = ast.literal_eval(line)
  except:
    continue
  if len(line['d_poly']) == 0:
    continue
  # if abs(line['v_ego'] * 2.2369 - 57) < 0.2235 and len(line['live_tracks']) > 1:

  if len([trk for trk in line['live_tracks'] if trk['vRel'] + line['v_ego'] > line['v_ego'] * 0.05]) > 5 and line['v_ego'] * 2.2369 > 10 and abs(line['steer_angle']) > .001:
    print(line['v_ego'] * 2.2369)
    print(idx)
    print()
  data_parsed.append(line)
data = data_parsed

# dPoly = [line['d_poly'] for line in data]
max_dist = 225
preprocessed = []

for idx, line in enumerate(data):
  preprocessed.append({})
  preprocessed[-1]['title'] = 'v_ego: {} mph, idx: {}'.format(line['v_ego'] * 2.2369, idx)

  dPoly = line['d_poly']

  x = np.linspace(0, max_dist, 100)
  y = np.polyval(dPoly, x)
  preprocessed[-1]['x'] = x
  preprocessed[-1]['y'] = y

  preprocessed[-1]['v_ego'] = line['v_ego']

  preprocessed[-1]['live_tracks'] = line['live_tracks']

  # for track in line['live_tracks']:
  #   plt.plot(track['dRel'], track['yRel'], 'bo')

  # plt.plot(x, y, label='desired path')
  # plt.legend()
  # plt.xlabel('longitudinal position (m)')
  # plt.ylabel('lateral position (m)')
  # plt.xlim(0, max_dist)
  ylim = [max(min(min(y), -15), -20), min(max(max(y), 15), 20)]
  preprocessed[-1]['ylim'] = ylim
  preprocessed[-1]['d_poly'] = dPoly
  # plt.ylim(*ylim)
  # plt.pause(0.01)

preprocessed = preprocessed[::6]
plt.clf()
plt.pause(0.01)
input('press any key to continue')
for line in preprocessed:
  t = time.time()
  plt.clf()
  plt.title(line['title'])

  for idx, track in enumerate(line['live_tracks']):
    if track['vRel'] + line['v_ego'] > line['v_ego'] * 0.1:
      if idx == 0:
        plt.plot(track['dRel'], track['yRel'], 'bo', label='radar tracks')
      else:
        plt.plot(track['dRel'], track['yRel'], 'bo')

  # plt.plot(line['x'], line['y'], label='desired path')
  plt.plot(line['x'], line['y'] + 3.7 / 2, 'r--', label='lane line')
  plt.plot(line['x'], line['y'] - 3.7 / 2, 'r--')

  plt.plot(line['x'], line['y'] + 3.7 / 2 + 3.7, 'g--', label='lane line')
  plt.plot(line['x'], line['y'] - 3.7 / 2 - 3.7, 'g--')

  plt.show()

  plt.legend()
  plt.xlabel('longitudinal position (m)')
  plt.ylabel('lateral position (m)')
  plt.xlim(0, max_dist)
  plt.ylim(*line['ylim'])
  plt.pause(0.001)
  print(time.time() - t)
  print(line['d_poly'])
  input()
  # time.sleep(0.01)
  # print(dPoly)
  # input()
