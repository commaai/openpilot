import matplotlib
matplotlib.use('Qt5Agg')
import os
os.environ["FILEREADER_CACHE"] = "1"
from collections import deque
from tools.lib.logreader import MultiLogIterator
from tools.lib.route import Route
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

lr = list(MultiLogIterator(Route("54827bf84c38b14f|2023-02-22--15-16-51").log_paths(), sort_by_time=True))

maxlen = 100  # 0.5s
cooldown_frames = 300  # wait this many frames until we can register another injection event
steering_presses = deque(maxlen=maxlen)
curvatures = deque(maxlen=maxlen)
lat_actives = deque(maxlen=maxlen)
injecting = deque(maxlen=maxlen)

CS = None
CC = None

all_1s_curv_rates = []
all_1s_curv_rates_speeds = []

start_time = None

for msg in tqdm(lr):
  if msg.which() == 'can':
    if start_time is None:
      start_time = msg.logMonoTime

  elif msg.which() == 'carState':
    CS = msg.carState
    steering_presses.append(bool(CS.steeringPressed))

  elif msg.which() == 'carControl':
    CC = msg.carControl
    lat_actives.append(CC.latActive)
    injecting.append(abs(CC.actuators.curvature) > 0.015)

  elif msg.which() == 'controlsState':
    CoS = msg.controlsState
    curvatures.append(CoS.curvature)

    if {maxlen} == {len(lat_actives), len(steering_presses), len(injecting)} and \
        all(lat_actives) and all(injecting) and steering_presses.count(True) < 10 and \
        None not in [CS, CC]:
      # If initial injection event started at too high of a lateral accel, skip
      if abs(curvatures[0] * CS.vEgo ** 2) > 1.5:
        continue
      # if abs(curv_rate * CS.vEgo ** 2) < 0.1
      curv_rate = (curvatures[0] - curvatures[-1]) * (1 / maxlen * 100)
      all_1s_curv_rates.append(curv_rate)
      all_1s_curv_rates_speeds.append(CS.vEgo)
      print('injection! curvature rate limit over 1s: {}'.format(curv_rate))
      print('speed: {} m/s, estimated lat jerk: {} m/s/s'.format(CS.vEgo, round(curv_rate * CS.vEgo ** 2, 4)))
      print('current v: {} s'.format(round((msg.logMonoTime - start_time) * 1e-9, 3)))
      print('')
      lat_actives.clear()
      steering_presses.clear()
      injecting.clear()

all_1s_curv_rates = np.array(all_1s_curv_rates)
all_1s_curv_rates_speeds = np.array(all_1s_curv_rates_speeds)


plt.figure(0)
plt.clf()
plt.title('Curvature rates for injection test events')
all_1s_curv_rates = np.array(all_1s_curv_rates)
all_1s_curv_rates_speeds = np.array(all_1s_curv_rates_speeds)
plt.scatter(all_1s_curv_rates_speeds, abs(all_1s_curv_rates))
plt.xlabel('speed m/s')
plt.ylabel('Curvature rate over 1 second')
plt.show()

plt.figure(1)
plt.clf()
plt.title('Lateral jerks for injection test events')
all_1s_curv_rates = np.array(all_1s_curv_rates)
all_1s_curv_rates_speeds = np.array(all_1s_curv_rates_speeds)
plt.scatter(all_1s_curv_rates_speeds, abs(all_1s_curv_rates) * all_1s_curv_rates_speeds ** 2)
plt.xlabel('speed m/s')
plt.ylabel('Lateral jerk over 1 second (m/s^3)')
plt.show()

plt.figure(2)
plt.clf()
sns.displot(np.abs(all_1s_curv_rates), bins=20)
sns.displot(np.abs(all_1s_curv_rates * all_1s_curv_rates_speeds ** 2), bins=20)


print('Samples: {} injection events'.format(len(all_1s_curv_rates)))
print('Mean curvature rate (1s): {} 1/m'.format(np.mean(np.abs(all_1s_curv_rates))))
print('Std curvature rate (1s): {} 1/m'.format(np.std(np.abs(all_1s_curv_rates))))
