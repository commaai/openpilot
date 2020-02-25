#! /usr/bin/env python
import matplotlib.pyplot as plt
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT
import math

libmpc = libmpc_py.libmpc
libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, 1.)

cur_state = libmpc_py.ffi.new("state_t *")
cur_state[0].x = 0.0
cur_state[0].y = 0.0
cur_state[0].psi = 0.0
cur_state[0].delta = 0.0

mpc_solution = libmpc_py.ffi.new("log_t *")
xx = []
yy = []
deltas = []
psis = []
times = []

curvature_factor = 0.3
v_ref = 1.0 * 20.12  # 45 mph

LANE_WIDTH = 3.7
p = [0.0, 0.0, 0.0, 0.0]
p_l = p[:]
p_l[3] += LANE_WIDTH / 2.0

p_r = p[:]
p_r[3] -= LANE_WIDTH / 2.0


l_poly = libmpc_py.ffi.new("double[4]", p_l)
r_poly = libmpc_py.ffi.new("double[4]", p_r)
p_poly = libmpc_py.ffi.new("double[4]", p)

l_prob = 1.0
r_prob = 1.0
p_prob = 1.0

for i in range(1):
  cur_state[0].delta = math.radians(510. / 13.)
  libmpc.run_mpc(cur_state, mpc_solution, l_poly, r_poly, p_poly, l_prob, r_prob,
                 curvature_factor, v_ref, LANE_WIDTH)

timesi = []
ct = 0
for i in range(21):
  timesi.append(ct)
  if i <= 4:
    ct += 0.05
  else:
    ct += 0.15


xi = list(mpc_solution[0].x)
yi = list(mpc_solution[0].y)
psii = list(mpc_solution[0].psi)
deltai = list(mpc_solution[0].delta)
print("COST: ", mpc_solution[0].cost)


plt.figure(0)
plt.subplot(3, 1, 1)
plt.plot(timesi, psii)
plt.ylabel('psi')
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(timesi, deltai)
plt.ylabel('delta')
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(timesi, yi)
plt.ylabel('y')
plt.grid(True)
plt.show()


####  UNCOMMENT TO CHECK ITERATIVE SOLUTION
####
####for i in range(100):
####  libmpc.run_mpc(cur_state, mpc_solution, l_poly, r_poly, p_poly, l_prob, r_prob,
####                 curvature_factor, v_ref, LANE_WIDTH)
####  print "x", list(mpc_solution[0].x)
####  print "y", list(mpc_solution[0].y)
####  print "delta", list(mpc_solution[0].delta)
####  print "psi", list(mpc_solution[0].psi)
####  # cur_state[0].x = mpc_solution[0].x[1]
####  # cur_state[0].y = mpc_solution[0].y[1]
####  # cur_state[0].psi = mpc_solution[0].psi[1]
####  cur_state[0].delta = radians(200 / 13.)#mpc_solution[0].delta[1]
####
####  xx.append(cur_state[0].x)
####  yy.append(cur_state[0].y)
####  psis.append(cur_state[0].psi)
####  deltas.append(cur_state[0].delta)
####  times.append(i * 0.05)
####
####
####def f(x):
####  return p_poly[0] * x**3 + p_poly[1] * x**2 + p_poly[2] * x + p_poly[3]
####
####
##### planned = map(f, xx)
##### plt.figure(1)
##### plt.plot(yy, xx, 'r-')
##### plt.plot(planned, xx, 'b--', linewidth=0.5)
##### plt.axes().set_aspect('equal', 'datalim')
##### plt.gca().invert_xaxis()
####
##### planned = map(f, map(float, list(mpc_solution[0].x)[1:]))
##### plt.figure(1)
##### plt.plot(map(float, list(mpc_solution[0].y)[1:]), map(float, list(mpc_solution[0].x)[1:]), 'r-')
##### plt.plot(planned, map(float, list(mpc_solution[0].x)[1:]), 'b--', linewidth=0.5)
##### plt.axes().set_aspect('equal', 'datalim')
##### plt.gca().invert_xaxis()
####
####plt.figure(2)
####plt.subplot(2, 1, 1)
####plt.plot(times, psis)
####plt.ylabel('psi')
####plt.subplot(2, 1, 2)
####plt.plot(times, deltas)
####plt.ylabel('delta')
####
####
####plt.show()
