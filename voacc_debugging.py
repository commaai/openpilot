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

# tesla with no radar
# lr = LogReader('https://connect.comma.ai/0f79c454f812791a/000000a3--e1e54de187/1037/1067', sort_by_time=True)
lr = sorted(lr, key=lambda m: m.logMonoTime)

CS = None

radar_data = []  # from radard/radar
model_data = []  # straight from model
pred_data = []  # kf predictions derived from model
kf_data = []  # internal kf state to visualize

# deque (noisy)
dRel_deque = deque(maxlen=round(1.0 / DT_MDL))


# kf (inaccurate?)
class KF:
  MIN_STD = 1
  MAX_STD = 15
  SIGMA_A = 2.5  # m/s^2, how much can lead relative velocity realistically change (not per step)
  DT = DT_MDL

  def __init__(self, x):
    self.x = x  # distance state estimate (m)
    self.vRel = 0.0  # relative velocity state estimate (m/s)
    self.P = np.diag([10.0 ** 2, 5.0 ** 2])  # variance of the state estimate. x uncertainty: 10m, vRel uncertainty: 5 m/s
    self.K = 0.0  # Kalman gain

    # TODO: understand this fully
    var_a = self.SIGMA_A ** 2  # acceleration variance
    self.Q = var_a * np.array([
      [self.DT ** 4 / 4.0, self.DT ** 3 / 2.0],  # affects x (m^2) and coupling
      [self.DT ** 3 / 2.0, self.DT ** 2],  # affects vRel ((m/s)^2)
    ],)

  def update(self, x, x_std, a_ego):
    # predict state
    # self.x = self.x + self.vRel * self.DT
    self.x = self.x + self.vRel * self.DT - 0.5 * a_ego * self.DT * self.DT
    self.vRel = self.vRel - a_ego * self.DT

    # predict covariance
    A = np.array([[1.0, self.DT],
                  [0.0, 1.0]])

    self.P = A @ self.P @ A.T + self.Q

    # update
    H = np.array([[1.0, 0.0]])

    x_std = np.clip(x_std, self.MIN_STD, self.MAX_STD)
    R = x_std ** 2

    x_pred = (H @ np.array([[self.x], [self.vRel]]))[0, 0]

    S = (H @ self.P @ H.T)[0, 0] + R

    K = (self.P @ H.T) / S

    y = x - x_pred  # how far meas is form prediction
    self.x = self.x + K[0, 0] * y
    self.vRel = self.vRel + K[1, 0] * y

    # update covariance
    I = np.eye(2)
    self.P = (I - K @ H) @ self.P

    return self.x, self.vRel

  # def update(self, x, x_std):
  #   P_pred = self.P + self.Q
  #
  #   x_std = np.clip(x_std, self.MIN_STD, self.MAX_STD)
  #   R = x_std ** 2
  #   self.K = P_pred / (P_pred + R)
  #
  #   self.x = self.x + self.K * (x - self.x)
  #   self.P = (1.0 - self.K) * P_pred
  #
  #   # velocity estimate
  #   x_pred = self.x + self.vRel * self.DT
  #   vRel_pred = self.vRel
  #   return self.x


kf = None

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
      if kf is None:
        kf = KF(dRel)

      model_data.append((msg.logMonoTime, dRel, lead.v[0], lead.a[0], lead.xStd[0]))

      # simple kf prediction for vlead
      if len(dRel_deque) == dRel_deque.maxlen:
        # vLead = CS.vEgo + (dRel - dRel_deque[0]) / (DT_MDL * len(dRel_deque))

        kf_dRel, kf_vRel = kf.update(dRel, lead.xStd[0], CS.aEgo)

        kf_data.append((msg.logMonoTime, kf.P, kf.K))
        print(dRel, kf_dRel)
        # kf_dRel = kf.x[0][0]
        # kf_vLead = CS.vEgo + kf.x[1][0]
        kf_vLead = CS.vEgo + kf_vRel

        pred_data.append((msg.logMonoTime, kf_dRel, kf_vLead, lead.a[0]))

      dRel_deque.append(dRel)
    else:
      dRel_deque.clear()
      kf = None

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

# # kf internal stats
# ax[4].plot([d[0] for d in kf_data], [d[1] for d in kf_data], label='kf P')
# ax[4].plot([d[0] for d in kf_data], [d[2] for d in kf_data], label='kf K')
# ax[4].set_ylabel('kf stats')
# ax[4].legend()

# TODO: print some stats about how close model and kf are from radar ground truths
