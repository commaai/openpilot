from tools.lib.logreader import LogReader
import matplotlib.pyplot as plt
from selfdrive.controls.radard import RADAR_TO_CAMERA

plt.ion()

# Camry w/ accurate lead from radar
# lr = LogReader('https://connect.comma.ai/3f447b402cbe27b6/0000008d--b6a2350b41/351/448', sort_by_time=True)
lr = LogReader('https://connect.comma.ai/3f447b402cbe27b6/0000008d--b6a2350b41/572/950', sort_by_time=True)
lr = sorted(lr, key=lambda m: m.logMonoTime)

radar_data = []
model_data = []

for msg in lr:
  if msg.which() == 'radarState':
    RS = msg.radarState

    if RS.leadOne.status:
      radar_data.append((msg.logMonoTime, RS.leadOne.dRel, RS.leadOne.vLead, RS.leadOne.aLeadK))

  elif msg.which() == 'modelV2':
    MD = msg.modelV2

    if len(MD.leadsV3):
      lead = MD.leadsV3[0]
      if lead.prob > 0.5:
        model_data.append((msg.logMonoTime, lead.x[0] - RADAR_TO_CAMERA, lead.v[0], lead.a[0]))

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
ax[1].set_ylabel('vLead (m/s)')
ax[1].legend()

ax[2].plot([d[0] for d in radar_data], [d[3] for d in radar_data], label='radar aLeadK',
           # marker='o', markersize=2, markeredgecolor='orange', markerfacecolor='orange'
           )
ax[2].plot([d[0] for d in model_data], [d[3] for d in model_data], label='model aLead',
           # marker='o', markersize=2, markeredgecolor='blue', markerfacecolor='blue'
           )
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
