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


def time_many_sends(p, bus, p_recv=None, msg_count=100, two_pandas=False, msg_len=8):
  if p_recv is None:
    p_recv = p
  if p == p_recv and two_pandas:
    raise ValueError("Cannot have two pandas that are the same panda")

  msg_id = random.randint(0x100, 0x200)
  to_send = [(msg_id, b"\xaa" * msg_len, bus)] * msg_count

  start_time = time.monotonic()
  p.can_send_many(to_send)
  r = []
  r_echo = []
  r_len_expected = msg_count if two_pandas else msg_count * 2
  r_echo_len_exected = msg_count if two_pandas else 0

  while len(r) < r_len_expected and (time.monotonic() - start_time) < 5:
    r.extend(p_recv.can_recv())
  end_time = time.monotonic()
  if two_pandas:
    while len(r_echo) < r_echo_len_exected and (time.monotonic() - start_time) < 10:
      r_echo.extend(p.can_recv())

  sent_echo = [x for x in r if x[2] == 0x80 | bus and x[0] == msg_id]
  sent_echo.extend([x for x in r_echo if x[2] == 0x80 | bus and x[0] == msg_id])
  resp = [x for x in r if x[2] == bus and x[0] == msg_id]

  leftovers = [x for x in r if (x[2] != 0x80 | bus and x[2] != bus) or x[0] != msg_id]
  assert len(leftovers) == 0

  assert len(resp) == msg_count
  assert len(sent_echo) == msg_count

  end_time = (end_time - start_time) * 1000.0
  comp_kbps = (1 + 11 + 1 + 1 + 1 + 4 + (msg_len * 8) + 15 + 1 + 1 + 1 + 7) * msg_count / end_time

  return comp_kbps


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
