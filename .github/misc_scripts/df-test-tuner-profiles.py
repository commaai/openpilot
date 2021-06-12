import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV

x_vel_relaxed = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]
y_dist_relaxed = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.521, 1.544, 1.568, 1.588, 1.599, 1.613, 1.634]
plt.plot(np.array(x_vel_relaxed) * CV.MS_TO_MPH, y_dist_relaxed, label='relaxed')


# x_vel_traffic = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]
# y_dist_traffic = [1.3781, 1.3791, 1.3457, 1.3134, 1.3145, 1.318, 1.3485, 1.257, 1.144, 0.979, 0.9461, 0.9156]
# plt.plot(np.array(x_vel_traffic) * CV.MS_TO_MPH, y_dist_traffic, label='traffic')


x_vel_roadtrip = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]
y_dist_roadtrip = [1.3978, 1.4132, 1.4318, 1.4536, 1.4862, 1.5321, 1.6058, 1.6589, 1.7798, 1.8748, 1.8953, 1.9127, 1.9276]
plt.plot(np.array(x_vel_roadtrip) * CV.MS_TO_MPH, y_dist_roadtrip, label='roadtrip')

x_vel_stock = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]
y_dist_stock = [1.8 for _ in range(len(x_vel_stock))]
plt.plot(np.array(x_vel_stock) * CV.MS_TO_MPH, y_dist_stock, label='stock')

y_dist_roadtrip_new = []
for (x_vel, y_dist), y_stock in zip(zip(x_vel_roadtrip, y_dist_roadtrip), y_dist_stock):
  print(x_vel)
  x = [0, 55, 90]
  y_roadtrip_mod = [0.625, 0.8, 0.2]
  roadtrip_mod = np.interp(x_vel, x, y_roadtrip_mod)
  stock_mod = 1 - roadtrip_mod
  y_dist_roadtrip_new.append((y_dist * roadtrip_mod) + (y_stock * stock_mod))

plt.plot(np.array(x_vel_roadtrip) * CV.MS_TO_MPH, y_dist_roadtrip_new, label='new roadtrip')


# y_dist_roadtrip_new = []
# for x_vel, y_dist in zip(x_vel_roadtrip, y_dist_roadtrip):
#   x = [16, 22, 30, 45, 60, 90]
#   y = [1., 1.015, 1.046666, 1.075, 1.045, 1.08]
#   y_dist *= ((np.interp(x_vel, x, y) - 1) / 2) + 1
#   y_dist_roadtrip_new.append(y_dist)
#
# plt.plot(np.array(x_vel_roadtrip), np.round(y_dist_roadtrip_new, 3), label='new roadtrip')



# plt.plot([min(x), max(x)], [0, 0], 'r--')
# plt.plot([0, 0], [min(y), max(y)], 'r--')

plt.xlabel('mph')
plt.ylabel('sec')

# poly = np.polyfit(x, y, 6)
# x = np.linspace(min(x), max(x), 100)
# y = np.polyval(poly, x)
# plt.plot(x, y, label='poly fit')

# to_round = True
# if to_round:
#   x = np.round(x, 4)
#   y = np.round(y, 5)
#   print('x = {}'.format(x.tolist()))
#   print('y = {}'.format(y.tolist()))

plt.legend()
plt.show()
