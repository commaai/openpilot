import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import minimize

a = -9.81
dt = 0.1

r = 2.0

v_ls = []
x_ls = []
v_egos = []

for vv_ego in np.arange(35, 40, 1):
    for vv_l in np.arange(35, 40, 1):
        for xx_l in np.arange(0, 100, 1.0):
            x_l = xx_l
            v_l = vv_l
            v_ego = vv_ego
            x_ego = 0.0

            ttc = None
            for t in np.arange(0, 100, dt):
                x_l += v_l * dt
                v_l += a * dt
                v_l = max(v_l, 0.0)

                x_ego += v_ego * dt
                if t > r:
                    v_ego += a * dt
                    v_ego = max(v_ego, 0.0)

                if x_ego >= x_l:
                    ttc = t
                    break

            if ttc is None:
                if xx_l < 0.1:
                    break

                v_ls.append(vv_l)
                x_ls.append(xx_l)
                v_egos.append(vv_ego)
                break


def eval_f(x, v_ego, v_l):
    est = x[0] * v_l + x[1] * v_l**2 \
            + x[2] * v_ego + x[3] * v_ego**2
    return est

def f(x):
    r = 0.0
    for v_ego, v_l, x_l in zip(v_egos, v_ls, x_ls):
        est = eval_f(x, v_ego, v_l)
        r += (x_l - est)**2

    return r

x0 = [0.5, 0.5, 0.5, 0.5]
res = minimize(f, x0, method='Nelder-Mead')
print(res)
print(res.x)

g = 9.81
t_r = 1.8

estimated = [4.0 + eval_f(res.x, v_ego, v_l) for (v_ego, v_l) in zip(v_egos, v_ls)]
new_formula = [4.0 + v_ego * t_r - (v_l - v_ego) * t_r + v_ego**2/(2*g) - v_l**2 / (2*g)  for (v_ego, v_l) in zip(v_egos, v_ls)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.scatter(v_egos, v_ls, x_ls, s=1)
# surf = ax.scatter(v_egos, v_ls, estimated, s=1)
surf = ax.scatter(v_egos, v_ls, new_formula, s=1)

ax.set_xlabel('v ego')
ax.set_ylabel('v lead')
ax.set_zlabel('min distance')
plt.show()
