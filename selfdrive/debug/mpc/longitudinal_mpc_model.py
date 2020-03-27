#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt

from selfdrive.controls.lib.longitudinal_mpc_model import libmpc_py

libmpc = libmpc_py.libmpc

dt = 1
speeds = [6.109375, 5.9765625, 6.6367188, 7.6875, 8.7578125, 9.4375, 10.21875, 11.070312, 11.679688, 12.21875]
accelerations = [0.15405273, 0.39575195, 0.36669922, 0.29248047, 0.27856445, 0.27832031, 0.29736328, 0.22705078, 0.16003418, 0.15185547]
ts = [t * dt for t in range(len(speeds))]

# TODO: Get from actual model packet
x = 0.0
positions = []
for v in speeds:
  positions.append(x)
  x += v * dt


# Polyfit trajectories
x_poly = list(map(float, np.polyfit(ts, positions, 3)))
v_poly = list(map(float, np.polyfit(ts, speeds, 3)))
a_poly = list(map(float, np.polyfit(ts, accelerations, 3)))

x_poly = libmpc_py.ffi.new("double[4]", x_poly)
v_poly = libmpc_py.ffi.new("double[4]", v_poly)
a_poly = libmpc_py.ffi.new("double[4]", a_poly)

cur_state = libmpc_py.ffi.new("state_t *")
cur_state[0].x_ego = 0
cur_state[0].v_ego = 10
cur_state[0].a_ego = 0

libmpc.init(1.0, 1.0, 1.0, 1.0, 1.0)

mpc_solution = libmpc_py.ffi.new("log_t *")
libmpc.init_with_simulation(cur_state[0].v_ego)

libmpc.run_mpc(cur_state, mpc_solution, x_poly, v_poly, a_poly)

# Converge to solution
for _ in range(10):
  libmpc.run_mpc(cur_state, mpc_solution, x_poly, v_poly, a_poly)


ts_sol = list(mpc_solution[0].t)
x_sol = list(mpc_solution[0].x_ego)
v_sol = list(mpc_solution[0].v_ego)
a_sol = list(mpc_solution[0].a_ego)


plt.figure()
plt.subplot(3, 1, 1)
plt.plot(ts, positions, 'k--')
plt.plot(ts_sol, x_sol)
plt.ylabel('Position [m]')
plt.xlabel('Time [s]')

plt.subplot(3, 1, 2)
plt.plot(ts, speeds, 'k--')
plt.plot(ts_sol, v_sol)
plt.xlabel('Time [s]')
plt.ylabel('Speed [m/s]')

plt.subplot(3, 1, 3)
plt.plot(ts, accelerations, 'k--')
plt.plot(ts_sol, a_sol)

plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')

plt.show()
