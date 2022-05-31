import time

from cereal import messaging
from cereal.messaging import drain_sock
from selfdrive.car.car_helpers import get_one_can

RADAR_START_ADDR = 0x500

can_sock = messaging.sub_sock('can', timeout=20)
get_one_can(can_sock)

# msg_seen = False
t = time.monotonic()
seen_t = None
i = 0

while 1:
  # # === get_one_can ===
  # a = get_one_can(can_sock)
  # for can in a.can:
  #   if can.address == RADAR_START_ADDR and seen_t is None:
  #     seen_t = t.monotonic()
  #     print('Message seen at {} s, {} its'.format(seen_t - t, i))

  # === drain_sock ===
  a = drain_sock(can_sock)
  for msg in a:
    for can in msg.can:
      if can.address == RADAR_START_ADDR and seen_t is None:
        seen_t = t.monotonic()
        print('Message seen at {} s, {} its'.format(seen_t - t, i))

  i += 1
