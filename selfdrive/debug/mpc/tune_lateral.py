#! /usr/bin/env python
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from selfdrive.car.honda.interface import CarInterface
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from selfdrive.controls.lib.vehicle_model import VehicleModel

# plot lateral MPC trajectory by defining boundary conditions:
# lane lines, p_poly and vehicle states. Use this script to tune MPC costs

libmpc = libmpc_py.libmpc

mpc_solution = libmpc_py.ffi.new("log_t *")

points_l = np.array([1.1049711, 1.1053879, 1.1073375, 1.1096942, 1.1124474, 1.1154714, 1.1192677, 1.1245866, 1.1321017, 1.1396152, 1.146443, 1.1555313, 1.1662073, 1.1774249, 1.1888939, 1.2009926, 1.2149779, 1.2300836, 1.2450289, 1.2617753, 1.2785473, 1.2974714, 1.3151019, 1.3331807, 1.3545501, 1.3763691, 1.3983455, 1.4215056, 1.4446729, 1.4691089, 1.4927692, 1.5175346, 1.5429921, 1.568854, 1.5968665, 1.6268958, 1.657122, 1.6853137, 1.7152609, 1.7477539, 1.7793678, 1.8098511, 1.8428392, 1.8746407, 1.9089606, 1.9426043, 1.9775689, 2.0136933, 2.0520134, 2.0891454])

points_r = np.array([-2.4442139, -2.4449506, -2.4448867, -2.44377, -2.4422617, -2.4393811, -2.4374201, -2.4334245, -2.4286852, -2.4238286, -2.4177458, -2.4094386, -2.3994849, -2.3904033, -2.380136, -2.3699453, -2.3594661, -2.3474073, -2.3342307, -2.3194637, -2.3046403, -2.2881098, -2.2706163, -2.2530098, -2.235604, -2.2160542, -2.1967411, -2.1758952, -2.1544619, -2.1325269, -2.1091819, -2.0850561, -2.0621953, -2.0364127, -2.0119917, -1.9851667, -1.9590458, -1.9306552, -1.9024918, -1.8745357, -1.8432863, -1.8131843, -1.7822732, -1.7507075, -1.7180918, -1.6845931, -1.650871, -1.6157099, -1.5787286, -1.5418037])


points_c = (points_l + points_r) / 2.0

def compute_path_pinv():
  deg = 3
  x = np.arange(50.0)
  X = np.vstack(tuple(x**n for n in range(deg, -1, -1))).T
  pinv = np.linalg.pinv(X)
  return pinv


def model_polyfit(points):
  path_pinv = compute_path_pinv()
  return np.dot(path_pinv, map(float, points))


xx = []
yy = []
deltas = []
psis = []
times = []

CP = CarInterface.get_params("HONDA CIVIC 2016 TOURING")
VM = VehicleModel(CP)

v_ref = 32.00  # 45 mph
curvature_factor = VM.curvature_factor(v_ref)
print(curvature_factor)

LANE_WIDTH = 3.9
p_l = map(float, model_polyfit(points_l))
p_r = map(float, model_polyfit(points_r))
p_p = map(float, model_polyfit(points_c))

l_poly = libmpc_py.ffi.new("double[4]", p_l)
r_poly = libmpc_py.ffi.new("double[4]", p_r)
p_poly = libmpc_py.ffi.new("double[4]", p_p)
l_prob = 1.0
r_prob = 1.0
p_prob = 1.0  # This is always 1


mpc_x_points = np.linspace(0., 2.5*v_ref, num=50)
points_poly_l = np.polyval(p_l, mpc_x_points)
points_poly_r = np.polyval(p_r, mpc_x_points)
points_poly_p = np.polyval(p_p, mpc_x_points)
print(points_poly_l)

lanes_x = np.linspace(0, 49)

cur_state = libmpc_py.ffi.new("state_t *")
cur_state[0].x = 0.0
cur_state[0].y = 0.5
cur_state[0].psi = 0.0
cur_state[0].delta = 0.0

xs = []
ys = []
deltas = []
titles = [
  'Steer rate cost',
  'Heading cost',
  'Lane cost',
  'Path cost',
]

# Steer rate cost
sol_x = OrderedDict()
sol_y = OrderedDict()
delta = OrderedDict()
for cost in np.logspace(-1, 1.0, 5):
  libmpc.init(1.0, 3.0, 1.0, cost)
  for _ in range(10):
    libmpc.run_mpc(cur_state, mpc_solution, l_poly, r_poly, p_poly, l_prob, r_prob,
                  curvature_factor, v_ref, LANE_WIDTH)
  sol_x[cost] = map(float, list(mpc_solution[0].x))
  sol_y[cost] = map(float, list(mpc_solution[0].y))
  delta[cost] = map(float, list(mpc_solution[0].delta))
xs.append(sol_x)
ys.append(sol_y)
deltas.append(delta)

# Heading cost
sol_x = OrderedDict()
sol_y = OrderedDict()
delta = OrderedDict()
for cost in np.logspace(-1, 1.0, 5):
  libmpc.init(1.0, 3.0, cost, 1.0)
  for _ in range(10):
    libmpc.run_mpc(cur_state, mpc_solution, l_poly, r_poly, p_poly, l_prob, r_prob,
                  curvature_factor, v_ref, LANE_WIDTH)
  sol_x[cost] = map(float, list(mpc_solution[0].x))
  sol_y[cost] = map(float, list(mpc_solution[0].y))
  delta[cost] = map(float, list(mpc_solution[0].delta))
xs.append(sol_x)
ys.append(sol_y)
deltas.append(delta)

# Lane cost
sol_x = OrderedDict()
sol_y = OrderedDict()
delta = OrderedDict()
for cost in np.logspace(-1, 2.0, 5):
  libmpc.init(1.0, cost, 1.0, 1.0)
  for _ in range(10):
    libmpc.run_mpc(cur_state, mpc_solution, l_poly, r_poly, p_poly, l_prob, r_prob,
                  curvature_factor, v_ref, LANE_WIDTH)
  sol_x[cost] = map(float, list(mpc_solution[0].x))
  sol_y[cost] = map(float, list(mpc_solution[0].y))
  delta[cost] = map(float, list(mpc_solution[0].delta))
xs.append(sol_x)
ys.append(sol_y)
deltas.append(delta)


# Path cost
sol_x = OrderedDict()
sol_y = OrderedDict()
delta = OrderedDict()
for cost in np.logspace(-1, 1.0, 5):
  libmpc.init(cost, 3.0, 1.0, 1.0)
  for _ in range(10):
    libmpc.run_mpc(cur_state, mpc_solution, l_poly, r_poly, p_poly, l_prob, r_prob,
                  curvature_factor, v_ref, LANE_WIDTH)
  sol_x[cost] = map(float, list(mpc_solution[0].x))
  sol_y[cost] = map(float, list(mpc_solution[0].y))
  delta[cost] = map(float, list(mpc_solution[0].delta))
xs.append(sol_x)
ys.append(sol_y)
deltas.append(delta)



plt.figure()

for i in range(len(xs)):
  ax = plt.subplot(2, 2, i + 1)
  sol_x = xs[i]
  sol_y = ys[i]
  for cost in sol_x.keys():
    plt.plot(sol_x[cost], sol_y[cost])

  plt.plot(lanes_x, points_r, '.b')
  plt.plot(lanes_x, points_l, '.b')
  plt.plot(lanes_x, (points_l + points_r) / 2.0, '--g')
  plt.plot(mpc_x_points, points_poly_l, 'b')
  plt.plot(mpc_x_points, points_poly_r, 'b')
  plt.plot(mpc_x_points, (points_poly_l + points_poly_r) / 2.0, 'g')
  plt.legend(map(lambda x: str(round(x, 2)), sol_x.keys()) + ['right', 'left', 'center'], loc=3)
  plt.title(titles[i])
  plt.grid(True)
  # ax.set_aspect('equal', 'datalim')


plt.figure()
for i in range(len(xs)):
  plt.subplot(2, 2, i + 1)
  sol_x = xs[i]
  delta = deltas[i]

  for cost in sol_x.keys():
    plt.plot(delta[cost])
  plt.title(titles[i])
  plt.legend(map(lambda x: str(round(x, 2)), sol_x.keys()), loc=3)
  plt.grid(True)

plt.show()
