#!/usr/bin/env python3
import numpy as np
import math
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt


from selfdrive.car.honda.interface import CarInterface
from selfdrive.car.honda.values import CAR
from selfdrive.controls.lib.vehicle_model import VehicleModel, create_dyn_state_matrices
from selfdrive.locationd.kalman.models.car_kf import CarKalman, ObservationKind, States

T_SIM = 5 * 60  # s
DT = 0.01


CP = CarInterface.get_params(CAR.CIVIC)
VM = VehicleModel(CP)

x, y = 0, 0  # m, m
psi = math.radians(0)  # rad

# The state is x = [v, r]^T
# with v lateral speed [m/s], and r rotational speed [rad/s]
state = np.array([[0.0], [0.0]])


ts = np.arange(0, T_SIM, DT)
speeds = 10 * np.sin(2 * np.pi * ts / 200.) + 25

angle_offsets = math.radians(1.0) * np.ones_like(ts)
angle_offsets[ts > 60] = 0
steering_angles = np.radians(5 * np.cos(2 * np.pi * ts / 100.))

xs = []
ys = []
psis = []
yaw_rates = []
speed_ys = []


kf_states = []
kf_ps = []

kf = CarKalman()

for i, t in tqdm(list(enumerate(ts))):
  u = speeds[i]
  sa = steering_angles[i]
  ao = angle_offsets[i]

  A, B = create_dyn_state_matrices(u, VM)

  state += DT * (A.dot(state) + B.dot(sa + ao))

  x += u * math.cos(psi) * DT
  y += (float(state[0]) * math.sin(psi) + u * math.sin(psi)) * DT
  psi += float(state[1]) * DT

  kf.predict_and_observe(t, ObservationKind.CAL_DEVICE_FRAME_YAW_RATE, [float(state[1])])
  kf.predict_and_observe(t, ObservationKind.CAL_DEVICE_FRAME_XY_SPEED, [[u, float(state[0])]])
  kf.predict_and_observe(t, ObservationKind.STEER_ANGLE, [sa])
  kf.predict_and_observe(t, ObservationKind.ANGLE_OFFSET_FAST, [0])
  kf.predict(t)

  speed_ys.append(float(state[0]))
  yaw_rates.append(float(state[1]))
  kf_states.append(kf.x.copy())
  kf_ps.append(kf.P.copy())

  xs.append(x)
  ys.append(y)
  psis.append(psi)


xs = np.asarray(xs)
ys = np.asarray(ys)
psis = np.asarray(psis)
speed_ys = np.asarray(speed_ys)
kf_states = np.asarray(kf_states)
kf_ps = np.asarray(kf_ps)


palette = sns.color_palette()

def plot_with_bands(ts, state, label, ax, idx=1, converter=None):
  mean = kf_states[:, state].flatten()
  stds = np.sqrt(kf_ps[:, state, state].flatten())

  if converter is not None:
    mean = converter(mean)
    stds = converter(stds)

  sns.lineplot(ts, mean, label=label, ax=ax)
  ax.fill_between(ts, mean - stds, mean + stds, alpha=.2, color=palette[idx])


print(kf.x)

sns.set_context("paper")
f, axes = plt.subplots(6, 1)

sns.lineplot(ts, np.degrees(steering_angles), label='Steering Angle [deg]', ax=axes[0])
plot_with_bands(ts, States.STEER_ANGLE, 'Steering Angle kf [deg]', axes[0], converter=np.degrees)

sns.lineplot(ts, np.degrees(yaw_rates), label='Yaw Rate [deg]', ax=axes[1])
plot_with_bands(ts, States.YAW_RATE, 'Yaw Rate kf [deg]', axes[1], converter=np.degrees)

sns.lineplot(ts, np.ones_like(ts) * VM.sR, label='Steer ratio [-]', ax=axes[2])
plot_with_bands(ts, States.STEER_RATIO, 'Steer ratio kf [-]', axes[2])
axes[2].set_ylim([10, 20])


sns.lineplot(ts, np.ones_like(ts), label='Tire stiffness[-]', ax=axes[3])
plot_with_bands(ts, States.STIFFNESS, 'Tire stiffness kf [-]', axes[3])
axes[3].set_ylim([0.8, 1.2])


sns.lineplot(ts, np.degrees(angle_offsets), label='Angle offset [deg]', ax=axes[4])
plot_with_bands(ts, States.ANGLE_OFFSET, 'Angle offset kf deg', axes[4], converter=np.degrees)
plot_with_bands(ts, States.ANGLE_OFFSET_FAST, 'Fast Angle offset kf deg', axes[4], converter=np.degrees, idx=2)

axes[4].set_ylim([-2, 2])

sns.lineplot(ts, speeds, ax=axes[5])

plt.show()
