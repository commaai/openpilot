import time
import numpy as np

from common.realtime import sec_since_boot

N = 1000

times = []
for i in range(1000):
    t1 = sec_since_boot()
    time.sleep(0.01)
    t2 = sec_since_boot()
    dt = t2 - t1
    times.append(dt)


print("Mean", np.mean(times))
print("Max", np.max(times))
print("Min", np.min(times))
print("Variance", np.var(times))
print("STD", np.sqrt(np.var(times)))
