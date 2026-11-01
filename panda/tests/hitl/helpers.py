import time
import random


def get_random_can_messages(n):
  m = []
  for _ in range(n):
    bus = random.randrange(3)
    addr = random.randrange(1 << 29)
    dat = bytes([random.getrandbits(8) for _ in range(random.randrange(1, 9))])
    m.append([addr, dat, bus])
  return m


def clear_can_buffers(panda, speed: int | None = None):
  if speed is not None:
    for bus in range(3):
      panda.set_can_speed_kbps(bus, speed)

  # clear tx buffers
  for i in range(4):
    panda.can_clear(i)

  # clear rx buffers
  panda.can_clear(0xFFFF)
  r = [1]
  st = time.monotonic()
  while len(r) > 0:
    r = panda.can_recv()
    time.sleep(0.05)
    if (time.monotonic() - st) > 10:
      raise Exception("Unable to clear can buffers for panda ", panda.get_serial())
