import numpy as np
import matplotlib.pyplot as plt

MPH_TO_MS = 0.44704
MS_TO_MPH = 1. / MPH_TO_MS

x = [0.0, 1.4082, 2.80311, 4.22661, 5.38271, 6.16561, 7.24781, 8.28308, 10.24465, 12.96402, 15.42303, 18.11903, 20.11703, 24.46614, 29.05805, 32.71015, 35.76326]
y = [0.35587, 0.46747, 0.41816, 0.33261, 0.27844, 0.2718, 0.28396, 0.29537, 0.30647, 0.31161, 0.3168, 0.3272, 0.34, 0.3824, 0.44, 0.4968, 0.56]

speeds = [6.16561, 36.5 * MPH_TO_MS, 35.76326]
mods = [1, 0.93, .89]
y_new = [i_y * np.interp(i_x, speeds, mods) for i_x, i_y in zip(x, y)]

plt.plot(np.array(x) * MS_TO_MPH, y, label='current gas')
plt.plot(np.array(x) * MS_TO_MPH, y_new, label='new gas')

plt.legend()
plt.show()

print(np.round(y_new, 5).tolist())
