import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV

x = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]  # velocities
x_traffic = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]

relaxed_old = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.521, 1.544, 1.568, 1.588, 1.599, 1.613, 1.634]
relaxed_new = y_dist = [1.4503, 1.4546, 1.4614, 1.4705, 1.4874, 1.5132, 1.557, 1.5868, 1.6207, 1.6488, 1.6535, 1.6595, 1.668]
roadtrip = [1.6428, 1.646, 1.6514, 1.6591, 1.6744, 1.6992, 1.7422, 1.7739, 1.8335, 1.8687, 1.8755, 1.8833, 1.8961]

traffic = [1.3781, 1.3791, 1.3457, 1.3134, 1.3145, 1.318, 1.3485, 1.257, 1.144, 0.979, 0.9461, 0.9156]

# plt.plot(x, roadtrip, 'o-', label='roadtrip')
plt.plot(x, relaxed_old, 'o-', label='relaxed old')
plt.plot(x, relaxed_new, 'o-', label='relaxed new')

relaxed = np.array(relaxed_old) * 0.6 + np.array(relaxed_new) * 0.4

plt.plot(x, relaxed, 'o-', label='relaxed new new')
# relaxed = np.array(roadtrip) - 0.1678*1.05
# relaxed[8] = 1.6476
# # plt.plot(x, relaxed, 'o-', label='relaxed new')
#
# relaxed = [i * np.interp(_x, [13.51, 41], [1, 0.9740044537214158]) for i, _x in zip(relaxed, x)]
# # plt.plot(x, relaxed, 'o-', label='relaxed new new')
# relaxed = np.array(relaxed_old) * 0.2 + np.array(relaxed) * 0.8
# plt.plot(x, relaxed, 'o-', label='relaxed new new new')
# # plt.plot(x_traffic, traffic, 'o-', label='traffic')




# plt.plot([min(x), max(x)], [0, 0], 'r--')
# plt.plot([0, 0], [min(y), max(y)], 'r--')

# plt.xlabel('mph')
# plt.ylabel('feet')

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
