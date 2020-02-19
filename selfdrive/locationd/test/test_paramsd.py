#!/usr/bin/env python3
import numpy as np
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.signal


from selfdrive.car.honda.interface import CarInterface
from selfdrive.car.honda.values import CAR
from selfdrive.controls.lib.vehicle_model import VehicleModel, create_dyn_state_matrices

T_SIM = 2 * 60  # s
DT = 0.01


CP = CarInterface.get_params(CAR.CIVIC)
VM = VehicleModel(CP)

x, y = 0, 0  # m, m
psi = math.radians(0)  # rad

# The state is x = [v, r]^T
# with v lateral speed [m/s], and r rotational speed [rad/s]
state = np.array([[0.0], [0.0]])

angle_offset = 0

ts = np.arange(0, T_SIM, DT)
speeds = 10 * np.sin(2 * np.pi * ts / 1000.) + 25
steering_angles = np.radians(5 * np.sin(2 * np.pi * ts / 100.) + angle_offset)

xs = []
ys = []
psis = []

for i, t in tqdm(list(enumerate(ts))):
  u = speeds[i]
  sa = steering_angles[i]

  A, B = create_dyn_state_matrices(u, VM)
  A_d, B_d, _, _, _ = scipy.signal.cont2discrete((A, B, 0, 0), DT)

  state = A_d.dot(state) + B_d.dot(sa)

  x += u * math.cos(psi) * DT
  y += (float(state[0]) * math.sin(psi) + u * math.sin(psi)) * DT
  psi += float(state[1]) * DT

  xs.append(x)
  ys.append(y)
  psis.append(psi)


xs = np.asarray(xs)
ys = np.asarray(ys)
psis = np.asarray(psis)
print(psis)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(-ys, xs)

plt.subplot(3, 1, 2)
plt.plot(np.degrees(psis), label='Psi [deg]')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(np.degrees(steering_angles), label='Steering Angle [deg]')
plt.legend()
plt.show()
