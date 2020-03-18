#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
import math

# plot liongitudinal MPC trajectory by defining boundary conditions:
# ego and lead vehicles state. Use this script to tune MPC costs

def RW(v_ego, v_l):
    TR = 1.8
    G = 9.81
    return (v_ego * TR - (v_l - v_ego) * TR + v_ego*v_ego/(2*G) - v_l*v_l / (2*G))


def NORM_RW_ERROR(v_ego, v_l, p):
    return (RW(v_ego, v_l) + 4.0 - p)
    return (RW(v_ego, v_l) + 4.0 - p) / (np.sqrt(v_ego + 0.5) + 0.1)


v_ego = 20.0
a_ego = 0

x_lead = 10.0
v_lead = 20.0
a_lead = -3.0
a_lead_tau = 0.

# v_ego = 7.02661012716
# a_ego = -1.26143024772

# x_lead = 29.625 + 20
# v_lead = 0.725235462189 + 1
# a_lead = -1.00025629997

# a_lead_tau = 2.90729817665

min_a_lead_tau = (a_lead**2 * math.pi) / (2 * (v_lead + 0.01)**2)
min_a_lead_tau = 0.0

print(a_lead_tau, min_a_lead_tau)
a_lead_tau = max(a_lead_tau, min_a_lead_tau)

ffi, libmpc = libmpc_py.get_libmpc(1)
libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
libmpc.init_with_simulation(v_ego, x_lead, v_lead, a_lead, a_lead_tau)

cur_state = ffi.new("state_t *")
cur_state[0].x_ego = 0.0
cur_state[0].v_ego = v_ego
cur_state[0].a_ego = a_ego
cur_state[0].x_l = x_lead
cur_state[0].v_l = v_lead

mpc_solution = ffi.new("log_t *")

for _ in range(10):
    print(libmpc.run_mpc(cur_state, mpc_solution, a_lead_tau, a_lead))


for i in range(21):
  print("t: %.2f\t x_e: %.2f\t v_e: %.2f\t a_e: %.2f\t" % (mpc_solution[0].t[i], mpc_solution[0].x_ego[i], mpc_solution[0].v_ego[i], mpc_solution[0].a_ego[i]))
  print("x_l: %.2f\t v_l: %.2f\t \t" % (mpc_solution[0].x_l[i], mpc_solution[0].v_l[i]))

t = np.hstack([np.arange(0., 1.0, 0.2), np.arange(1.0, 10.1, 0.6)])

print(map(float, mpc_solution[0].x_ego)[-1])
print(map(float, mpc_solution[0].x_l)[-1] - map(float, mpc_solution[0].x_ego)[-1])

plt.figure(figsize=(8, 8))

plt.subplot(4, 1, 1)
x_l = np.array(map(float, mpc_solution[0].x_l))
plt.plot(t, map(float, mpc_solution[0].x_ego))
plt.plot(t, x_l)
plt.legend(['ego', 'lead'])
plt.title('x')
plt.grid()

plt.subplot(4, 1, 2)
v_ego = np.array(map(float, mpc_solution[0].v_ego))
v_l = np.array(map(float, mpc_solution[0].v_l))
plt.plot(t, v_ego)
plt.plot(t, v_l)
plt.legend(['ego', 'lead'])
plt.ylim([-1, max(max(v_ego), max(v_l))])
plt.title('v')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t, map(float, mpc_solution[0].a_ego))
plt.plot(t, map(float, mpc_solution[0].a_l))
plt.legend(['ego', 'lead'])
plt.title('a')
plt.grid()


plt.subplot(4, 1, 4)
d_l = np.array(map(float, mpc_solution[0].x_l)) - np.array(map(float, mpc_solution[0].x_ego))
desired = 4.0 + RW(v_ego, v_l)

plt.plot(t, d_l)
plt.plot(t, desired, '--')
plt.ylim(-1, max(max(desired), max(d_l)))
plt.legend(['relative distance', 'desired distance'])
plt.grid()

plt.show()

# c1 = np.exp(0.3 * NORM_RW_ERROR(v_ego, v_l, d_l))
# c2 = np.exp(4.5 - d_l)
# print(c1)
# print(c2)

# plt.figure()
# plt.plot(t, c1, label="NORM_RW_ERROR")
# plt.plot(t, c2, label="penalty function")
# plt.legend()

# ## OLD MPC
# a_lead_tau = 1.5
# a_lead_tau = max(a_lead_tau, -a_lead / (v_lead + 0.01))

# ffi, libmpc = libmpc_py.get_libmpc(1)
# libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
# libmpc.init_with_simulation(v_ego, x_lead, v_lead, a_lead, a_lead_tau)

# cur_state = ffi.new("state_t *")
# cur_state[0].x_ego = 0.0
# cur_state[0].v_ego = v_ego
# cur_state[0].a_ego = a_ego
# cur_state[0].x_lead = x_lead
# cur_state[0].v_lead = v_lead
# cur_state[0].a_lead = a_lead

# mpc_solution = ffi.new("log_t *")

# for _ in range(10):
#     print libmpc.run_mpc(cur_state, mpc_solution, a_lead_tau)

# t = np.hstack([np.arange(0., 1.0, 0.2), np.arange(1.0, 10.1, 0.6)])

# print(map(float, mpc_solution[0].x_ego)[-1])
# print(map(float, mpc_solution[0].x_lead)[-1] - map(float, mpc_solution[0].x_ego)[-1])
# plt.subplot(4, 2, 2)
# plt.plot(t, map(float, mpc_solution[0].x_ego))
# plt.plot(t, map(float, mpc_solution[0].x_lead))
# plt.legend(['ego', 'lead'])
# plt.title('x')

# plt.subplot(4, 2, 4)
# plt.plot(t, map(float, mpc_solution[0].v_ego))
# plt.plot(t, map(float, mpc_solution[0].v_lead))
# plt.legend(['ego', 'lead'])
# plt.title('v')

# plt.subplot(4, 2, 6)
# plt.plot(t, map(float, mpc_solution[0].a_ego))
# plt.plot(t, map(float, mpc_solution[0].a_lead))
# plt.legend(['ego', 'lead'])
# plt.title('a')


# plt.subplot(4, 2, 8)
# plt.plot(t, np.array(map(float, mpc_solution[0].x_lead)) - np.array(map(float, mpc_solution[0].x_ego)))

# plt.show()
