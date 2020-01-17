#!/usr/bin/env python3
import time

from common.realtime import sec_since_boot, monotonic_time


if __name__ == "__main__":
    N = 100000

    t = time.time()
    for _ in range(N):
        monotonic_time()
    dt = time.time() - t

    print("Monotonic", dt)

    t = time.time()
    for _ in range(N):
        sec_since_boot()
    dt = time.time() - t

    print("Boot", dt)
