import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tools.lib.logreader import LogReader
from selfdrive.controls.radard import RADAR_TO_CAMERA
from openpilot.common.realtime import DT_MDL
from openpilot.common.simple_kalman import KF1D, get_kalman_gain

plt.ion()

# Camry w/ accurate lead from radar
# lr = LogReader('https://connect.comma.ai/3f447b402cbe27b6/0000008d--b6a2350b41/351/448', sort_by_time=True)
lr = LogReader('https://connect.comma.ai/3f447b402cbe27b6/0000008d--b6a2350b41/572/950', sort_by_time=True)
lr = sorted(lr, key=lambda m: m.logMonoTime)

CS = None

radar_data = []  # from radard/radar
model_data = []  # straight from model
pred_data = []  # kf predictions derived from model

# deque (noisy)
dRel_deque = deque(maxlen=round(1.0 / DT_MDL))


# kf (inaccurate?)
class KF:
  MIN_STD = 0.5
  MAX_STD = 15

  def __init__(self):
    self.x = 0.0  # state estimate  # TODO: initialize properly
    self.P = 0.0  # variance of the state estimate
    self.Q = 0.1  # process variance per step

  def update(self, x, x_std):
    P_pred = self.P + self.Q

    x_std = np.clip(x_std, self.MIN_STD, self.MAX_STD)
    R = x_std ** 2
    K = P_pred / (P_pred + R)

    self.x = self.x + K * (x - self.x)
    self.P = (1.0 - K) * P_pred
    return self.x


kf = KF()

for msg in lr:
  if msg.which() == 'carState':
    CS = msg.carState

  elif msg.which() == 'radarState':
    RS = msg.radarState

    if RS.leadOne.status:
      radar_data.append((msg.logMonoTime, RS.leadOne.dRel, RS.leadOne.vLead, RS.leadOne.aLeadK))

  elif msg.which() == 'modelV2':
    MD = msg.modelV2

    if CS is None:
      continue

    if not len(MD.leadsV3):
      continue

    lead = MD.leadsV3[0]
    if lead.prob > 0.5:
      dRel = lead.x[0] - RADAR_TO_CAMERA
      model_data.append((msg.logMonoTime, dRel, lead.v[0], lead.a[0], lead.xStd[0]))

      # simple kf prediction for vlead
      if len(dRel_deque) == dRel_deque.maxlen:
        # vLead = CS.vEgo + (dRel - dRel_deque[0]) / (DT_MDL * len(dRel_deque))

        kf_dRel = kf.update(dRel, lead.xStd[0])
        print(dRel, kf_dRel)
        # kf_dRel = kf.x[0][0]
        # kf_vLead = CS.vEgo + kf.x[1][0]

        pred_data.append((msg.logMonoTime, kf_dRel, lead.v[0], lead.a[0]))

      dRel_deque.append(dRel)
    else:
      dRel_deque.clear()

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot([d[0] for d in radar_data], [d[1] for d in radar_data], label='radar dRel')
ax[0].plot([d[0] for d in model_data], [d[1] for d in model_data], label='model dRel')
ax[0].plot([d[0] for d in pred_data], [d[1] for d in pred_data], label='predicted dRel')
ax[0].set_ylabel('dRel (m)')
ax[0].legend()

ax[1].plot([d[0] for d in radar_data], [d[2] for d in radar_data], label='radar vLead')
ax[1].plot([d[0] for d in model_data], [d[2] for d in model_data], label='model vLead')
ax[1].plot([d[0] for d in pred_data], [d[2] for d in pred_data], label='predicted vLead')
ax[1].set_ylabel('vLead (m/s)')
ax[1].legend()

ax[2].plot([d[0] for d in radar_data], [d[3] for d in radar_data], label='radar aLeadK')
ax[2].plot([d[0] for d in model_data], [d[3] for d in model_data], label='model aLead')
# ax[2].plot([d[0] for d in pred_data], [d[3] for d in pred_data], label='predicted aLead')
ax[2].set_ylabel('aLead (m/s²)')
ax[2].legend()

# prob
# ax[3].plot([d[0] for d in model_data], [d[4] for d in model_data], label='model lead prob')
# ax[3].set_ylabel('lead prob')
# ax[3].legend()

# xStd
ax[3].plot([d[0] for d in model_data], [d[4] for d in model_data], label='model lead xStd')
ax[3].set_ylabel('lead xStd (m)')
ax[3].legend()

# TODO: print some stats about how close model and kf are from radar ground truths
