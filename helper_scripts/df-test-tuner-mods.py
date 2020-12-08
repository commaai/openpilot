import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV

x = [-60, -44.80314961, -35.09104331, -25.04585719, -17.59238547, -11.06657122, -6.83182713, -4.97584109, -3.36547065, -1.76896922, -0.71492484, 0.0, 1.25, 3.06057623, 4.24570508, 6.11041518, 10.0]
y = [.76, 0.62323, 0.49488, 0.40656, 0.32227, 0.23914, 0.12269, 0.10483, 0.08074, 0.04886, 0.0072, 0.0, -0.05648, -0.0792, -0.15675, -0.23289, -0.315]  # modification values
plt.plot(x, y, 'o-', label='relative velocity mod')




plt.plot([min(x), max(x)], [0, 0], 'r--')
plt.plot([0, 0], [min(y), max(y)], 'r--')

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
