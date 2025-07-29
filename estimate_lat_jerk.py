import math
import matplotlib.pyplot as plt
from openpilot.common.filter_simple import JerkEstimator1, JerkEstimator2, JerkEstimator3
from tools.lib.logreader import LogReader
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY

plt.ion()

lr = LogReader("https://connect.comma.ai/8011d605be1cbb77/000001c6--130f915e07/130/138")
calibrator = PoseCalibrator()

sm = {}

j1 = JerkEstimator1(0.01)
j2 = JerkEstimator2(0.01)
j3 = JerkEstimator3(0.01)

accels = []
jerks1, jerks2, jerks3 = [], [], []

for msg in lr:
  if msg.which() == 'livePose':
    sm['livePose'] = msg.livePose
  elif msg.which() == 'liveParameters':
    sm['liveParameters'] = msg.liveParameters
  elif msg.which() == 'carState':
    if len(sm) < 2:
      continue

    CS = msg.carState
    device_pose = Pose.from_live_pose(sm['livePose'])
    calibrated_pose = calibrator.build_calibrated_pose(device_pose)

    yaw_rate = calibrated_pose.angular_velocity.yaw
    roll = sm['liveParameters'].roll
    roll_compensated_lateral_accel = (CS.vEgo * yaw_rate) - (math.sin(roll) * ACCELERATION_DUE_TO_GRAVITY)

    accels.append(roll_compensated_lateral_accel)
    jerks1.append(j1.update(roll_compensated_lateral_accel))
    jerks2.append(j2.update(roll_compensated_lateral_accel))
    jerks3.append(j3.update(roll_compensated_lateral_accel))

    print(roll_compensated_lateral_accel)


fig, axs = plt.subplots(2, sharex=True)

axs[0].plot(accels, label='Lateral Accel')
axs[0].set_ylabel('Lateral Acceleration (m/s²)')
axs[0].legend()

axs[1].plot(jerks1, label='Jerk Estimator 1')
axs[1].plot(jerks2, label='Jerk Estimator 2')
axs[1].plot(jerks3, label='Jerk Estimator 3')
axs[1].set_ylabel('Lateral Jerk (m/s³)')
axs[1].legend()

