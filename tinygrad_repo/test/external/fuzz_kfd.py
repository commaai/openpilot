#!/usr/bin/env python3
import random
from tqdm import trange
from typing import List
from tinygrad import Device
from tinygrad.runtime.ops_amd import AMDDevice, HWQueue

if __name__ == "__main__":
  dev: List[AMDDevice] = [Device[f"KFD:{i}"] for i in range(6)]
  print(f"got {len(dev)} devices")

  buffers = [(rd:=random.choice(dev), rd.allocator.alloc(random.randint(1, 10000))) for i in range(100)]

  for _ in trange(100000):
    d1, b1 = random.choice(buffers)
    d2, b2 = random.choice(buffers)
    d1._gpu_map(b2)
    q = HWQueue()
    q.signal(sig:=AMDDevice._alloc_signal(10))
    qc = HWQueue()
    qc.wait(sig)
    qc.copy(b1.va_addr, b2.va_addr, min(b1.size, b2.size))
    d1.completion_signal.value = 1
    qc.signal(d1.completion_signal)
    qc.submit(d1)
    q.wait(d1.completion_signal)
    q.submit(d1)
    AMDDevice._wait_on(d1.completion_signal.event_id)
