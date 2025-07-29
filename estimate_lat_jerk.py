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

j1 = JerkEstimator1(1/20)
j2 = JerkEstimator2(1/100)
j3 = JerkEstimator3(1/20)
# j4 = JerkEstimator4(1/20)
j5 = JerkEstimator1(1/100)

accels = []
kf_accels = []
jerks1, jerks2, jerks3, jerks4, jerks5 = [], [], [], [], []
lp_updated = False

for msg in lr:
  if msg.which() == 'livePose':
    sm['livePose'] = msg.livePose
    lp_updated = True
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

    _j2 = j2.update(roll_compensated_lateral_accel)
    _j5 = j5.update(roll_compensated_lateral_accel)
    if lp_updated:
      _j1 = j1.update(roll_compensated_lateral_accel)
      _j3 = j3.update(roll_compensated_lateral_accel)
      # _j4 = j4.update(roll_compensated_lateral_accel)
      lp_updated = False
    else:
      _j1 = j1.x
      _j2 = j2.x
      _j3 = j3.x
      # _j4 = j4.x
      _j5 = j5.x

    jerks1.append(_j1)
    jerks2.append(_j2)
    jerks3.append(_j3)
    # jerks4.append(_j4)
    jerks5.append(_j5)
    accels.append(roll_compensated_lateral_accel)

    print(roll_compensated_lateral_accel)


fig, axs = plt.subplots(2, sharex=True)

axs[0].plot(accels, label='Lateral Accel')
axs[0].set_ylabel('Lateral Acceleration (m/s²)')
axs[0].legend()

axs[1].plot(jerks1, label='Low pass filter at 20 Hz (1)')
axs[1].plot(jerks2, label='Kalman filter (2)')
axs[1].plot(jerks3, label='Windowed (3)')
# axs[1].plot(jerks4, label='Jerk Estimator 4')
# axs[1].plot(jerks5, label='Low pass filter at 100 Hz (5)')
axs[1].set_ylabel('Lateral Jerk (m/s³)')
axs[1].legend()

