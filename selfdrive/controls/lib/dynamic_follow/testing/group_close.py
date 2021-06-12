import time
import numpy as np


def cluster(data, maxgap):
  data.sort()
  groups = [[data[0]]]
  for x in data[1:]:
    if abs(x - groups[-1][-1]) <= maxgap:
      groups[-1].append(x)
    else:
      groups.append([x])
  return groups


_n = [1, 2, 3, 4, 4.2, 5, 6, 6.24, -5.4, -5.425, 25, 26, 32, 32.458] #, 16, 18]

t = time.time()
for _ in range(100):
  cluster(_n, 0.5)
print(time.time() - t)

p1 = [25.32, -.198]
p2 = [25.36, 0.04036]

t = time.time()
for _ in range(100):
  for _ in range(14*14):
    np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
print(time.time() - t)
