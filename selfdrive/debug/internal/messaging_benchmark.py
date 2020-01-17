#!/usr/bin/env python3
import time
import cereal.messaging as messaging


def init_message_bench(N=100000):
    t = time.time()
    for _ in range(N):
        dat = messaging.new_message()
        dat.init('controlsState')

    dt = time.time() - t
    print("Init message %d its, %f s" % (N, dt))


if __name__ == "__main__":
    init_message_bench()
