from collections import deque
import matplotlib.pyplot as plt
from tools.lib.logreader import LogReader
from selfdrive.controls.radard import RADAR_TO_CAMERA
from openpilot.common.realtime import DT_MDL

plt.ion()

# Camry w/ accurate lead from radar
# lr = LogReader('https://connect.comma.ai/3f447b402cbe27b6/0000008d--b6a2350b41/351/448', sort_by_time=True)
lr = LogReader('https://connect.comma.ai/3f447b402cbe27b6/0000008d--b6a2350b41/572/950', sort_by_time=True)
lr = sorted(lr, key=lambda m: m.logMonoTime)

CS = None

radar_data = []  # from radard/radar
model_data = []  # straight from model
pred_data = []  # kf predictions derived from model

dRel_deque = deque(maxlen=round(1.0 / DT_MDL))

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
      model_data.append((msg.logMonoTime, dRel, lead.v[0], lead.a[0]))

      # simple kf prediction for vlead
      if len(dRel_deque) == dRel_deque.maxlen:
        vLead = CS.vEgo + (dRel - dRel_deque[0]) / (DT_MDL * len(dRel_deque))
        pred_data.append((msg.logMonoTime, dRel, vLead, lead.a[0]))

      dRel_deque.append(dRel)
    else:
      dRel_deque.clear()

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot([d[0] for d in radar_data], [d[1] for d in radar_data], label='radar dRel',
           # marker='o', markersize=2, markeredgecolor='orange', markerfacecolor='orange'
           )
ax[0].plot([d[0] for d in model_data], [d[1] for d in model_data], label='model dRel',
           # marker='o', markersize=2, markeredgecolor='blue', markerfacecolor='blue'
           )
ax[0].set_ylabel('dRel (m)')
ax[0].legend()

ax[1].plot([d[0] for d in radar_data], [d[2] for d in radar_data], label='radar vLead',
           # marker='o', markersize=2, markeredgecolor='orange', markerfacecolor='orange'
           )
ax[1].plot([d[0] for d in model_data], [d[2] for d in model_data], label='model vLead',
           # marker='o', markersize=2, markeredgecolor='blue', markerfacecolor='blue'
           )
ax[1].plot([d[0] for d in pred_data], [d[2] for d in pred_data], label='predicted vLead',
           # marker='o', markersize=2, markeredgecolor='green', markerfacecolor='green'
           )
ax[1].set_ylabel('vLead (m/s)')
ax[1].legend()

ax[2].plot([d[0] for d in radar_data], [d[3] for d in radar_data], label='radar aLeadK',
           # marker='o', markersize=2, markeredgecolor='orange', markerfacecolor='orange'
           )
ax[2].plot([d[0] for d in model_data], [d[3] for d in model_data], label='model aLead',
           # marker='o', markersize=2, markeredgecolor='blue', markerfacecolor='blue'
           )
# ax[2].plot([d[0] for d in pred_data], [d[3] for d in pred_data], label='predicted aLead',
#          # marker='o', markersize=2, markeredgecolor='green', markerfacecolor='green'
#          )
ax[2].set_ylabel('aLead (m/s²)')
ax[2].legend()

# plt.plot([d[0] for d in radar_data], [d[1] for d in radar_data], label='radar dRel',
#          # marker='o', markersize=2, markeredgecolor='orange', markerfacecolor='orange'
#          )
# plt.plot([d[0] for d in model_data], [d[1] for d in model_data], label='model dRel',
#          # marker='o', markersize=2, markeredgecolor='blue', markerfacecolor='blue'
#          )
# plt.legend()
# plt.show()
