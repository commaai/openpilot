import os
import ast
import matplotlib.pyplot as plt
import numpy as np

os.chdir('C:/Git/openpilot-repos/op-smiskol/selfdrive/controls/lib/dynamic_follow/testing')

with open('lane_speed_log', 'r') as f:
  _data = f.read().split('\n')

live_tracks_find = "'live_tracks': <capnp list reader "
data = []
for line in _data:
  # ugh, WHY did I not correctly code the live tracks data when gathering?? now i have to do all this to fix it in post lol
  # never fix in post
  if 'dtype' in line or 'nan' in line:
    continue
  try:
    line = line.replace('array(', '').replace('])', ']')
    lt_start = line.index(", 'live_tracks':")
    lt_end = line.index(')]>')
    lt = line[lt_start:lt_end+3]
    line = line.replace(lt, '')
  except:
    continue
  line = ast.literal_eval(line)
  lt = lt[37:-2].split('),')
  lt_new = []
  for trk in lt:
    trk = trk.strip(' ').replace('(', '').replace(')', '')
    trk = trk.split(',')
    trk_new = {}
    for val in trk:
      if any([i in val for i in ['timeStamp', 'trackId', 'stationary', 'oncoming', 'currentTime', 'status', 'aRel']]):
        continue
      val = val.strip()
      val_type = val[:4]
      trk_new[val_type] = ast.literal_eval(val[6:].strip())
    lt_new.append(trk_new)
  line['live_tracks'] = lt_new
  data.append(line)

data = data[4715:10857]

one_time_oncoming_idx = 3656
data = data[1000+500+1900:]

# for line in data:
#   # print('---LINE---')
#   # print('v_ego: {}'.format(line['v_ego']))
#   for trk in line['live_tracks']:
#     if trk['vRel'] + line['v_ego'] < -1:
#       print('Track:')
#       print('trk_speed: {}'.format(trk['vRel'] + line['v_ego']))

for idx, line in enumerate(data):
  if idx % 4 == 0:
    continue
  plt.clf()
  max_dist = 180
  x = np.linspace(0, max_dist)
  y = np.polyval(line['d_poly'], x)
  dRels = [trk['dRel'] for trk in line['live_tracks']]
  vRels = [trk['vRel'] for trk in line['live_tracks']]
  yRels = [trk['yRel'] for trk in line['live_tracks']]
  plt.title('v_ego: {}, idx: {}'.format(round(line['v_ego'], 3), idx))
  plt.scatter(dRels, yRels)
  plt.plot(x, y - 3.7/2, 'b')
  plt.plot(x, y + 3.7/2, 'b')
  plt.plot(x, y + 3.7*1.5, 'r--')
  plt.plot(x, y - 3.7*1.5, 'r--')
  plt.xlim(0, max_dist)
  plt.ylim(-10, 10)
  plt.show()
  plt.pause(0.01)

v_ego = [line['v_ego'] for line in data]
plt.plot(v_ego, label='v_ego')
plt.legend()
plt.show()

