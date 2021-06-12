import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


x = [-26, -15.6464, -9.8422, -6.0, -4.0, -2.68, -2.3, -1.8, -1.26, -0.61, 0, 0.61, 1.26, 2.1, 2.68, 4.4704]  # relative velocity values
y = [.76, .504, 0.34, 0.29, 0.25, 0.22, 0.19, 0.13, 0.053, 0.017, 0, -0.015, -0.042, -0.13, -0.19, -.315]  # modification values
TR = 1.6

old_y = []
new_y = []
for _x, _y in zip(x, y):
  old_y.append(_y)
  _y = _y + 1
  new_y.append(_y)
  # assert np.isclose(TR + old_y[-1], TR * new_y[-1])

new_TR = 1.2
plt.plot(x, np.array(old_y) + new_TR, label='old_y')
plt.plot(x, ((np.array(new_y) - 1) / new_TR + 1) * new_TR, label='new_y')
plt.legend()
print(np.round(new_y, 4).tolist())
