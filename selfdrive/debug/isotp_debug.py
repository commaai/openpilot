from tools.lib.logreader import LogReader
from tools.lib.route import Route
from cereal import messaging
from tqdm import tqdm
import time
from panda.python import uds

# from selfdrive.car.fw_versions import REQUESTS, CHRYSLER_VERSION_RESPONSE, CHRYSLER_VERSION_REQUEST
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery

# lr = list(LogReader(Route("b4069565beef1091|2024-01-08--08-58-07").log_paths()[0]))  # rav4 hybrid WITH radar fw
# lr = list(LogReader(Route("b4069565beef1091|2024-01-07--09-30-22").log_paths()[0]))  # rav4 hybrid without radar fw  # TODO: this is!
# lr = list(LogReader(Route("e906fc18a593603a|2024-01-26--03-05-06").log_paths()[0]))  # pilot 2023 no fw
# lr = list(LogReader(Route("fc4d8f4e5f8b353a|2024-01-22--07-19-26").log_paths()[0]))  # civic 23 mostly missing radar
lr = list(LogReader(Route("5130484aa8069bad|2024-01-26--21-05-14").log_paths()[0]))  # civic 22 marco dingo missing radar


all_msgs = list(filter(lambda x: x.which() in ["can", "sendcan"], sorted(lr, key=lambda x: x.logMonoTime)))

"[[1848, None, 1], [1864, None, 1], [1807, None, 1], [1848, None, 0], [1896, None, 1], [1900, None, 1], [2024, None, 0], [2024, None, 1], [1996, None, 1], [1896, None, 0], [2025, None, 1], [1900, None, 0], [2025, None, 0]]"

PUBLISH = False
can_send = True
if PUBLISH:
  pm = messaging.PubMaster(["can"])
  for msg in tqdm(all_msgs):
    if msg.which() == 'can':
      # for can in msg.can:
      #   if can
      if not can_send:
        if msg.logMonoTime > (162854120012 - 0.04e-9):
          can_send = True

      if can_send:
        # print('sending', [m.address for m in msg.can])
        pm.send('can', msg.as_builder())
        time.sleep(0.01)

  print('Done publishing')

start_mono_time = None
prev_mono_time = 0

can_finger = {}

counter = 0
for msg in lr:
  if msg.which() == 'can':
    if start_mono_time is None:
      start_mono_time = msg.logMonoTime

  if msg.which() in ("can", 'sendcan'):

    for can in getattr(msg, msg.which()):
      # if can.address in (0x747 - 0x280, 0x747 + 0x8):  # Chrysler
      # if can.address in (2024, 2016-0x280):  # Chrysler
      # print(can.address)
      # if can.address in [0x700 + i - 0x280 for i in range(256)]:
      # if can.address in (0x747+0x8, 0x747-0x280):
      # if can.address in (0x7d0 + 8,):
      # if b'KND' in can.dat:
      #   print(can.address, can.dat)
      # continue
      # print(can.address)
      # if can.address in (2000, 2000 + 8) or can.dat == bytes([0x30, 0x08, 0x0a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa]):#,0x7df):  # and can.src in (1, 129):
      # if can.src < 128:
      #
      #   can_finger[can.address] = len(can.dat)
      #   print(can_finger)
      # addr = 0x7d0
      # addrs = [0x7d0, 0x7d1, 0x7c4, 0x7e1, 0x7e0, 0x7d4, 0x730, 0x7b1, 0x7b3, 0x7b7]
      # addrs = [0x742, 0x744, 0x747, 0x753, 0x75A, 0x7e0, 0x7e1, 0x761, ]

      # if len(can.dat) > 2 and (can.dat[1] == 0x3e or can.dat[1] == (0x3e + 0x40) or can.dat[2] == 0x3e or can.dat[1] == (0x3e + 0x40)):
      #   print('tester present addr!', can.address, can.src, can.dat.hex(), can.dat)
      # continue

      addrs = [0x18dab0f1]  # [0x7a1, 0x7d2]  # [0x7a1, 0x700]
      # addrs = addrs + [addr + 8 for addr in addrs]# + [addr - 0x280 for addr in addrs]
      addrs = addrs + [uds.get_rx_addr_for_tx_addr(addr) for addr in addrs]# + [addr - 0x280 for addr in addrs]
      if can.address in addrs:  # or b'991' in can.dat:  # or (len(can.dat) > 2 and can.dat[1] == 0x3e+0x40):
        # if b'5A' in can.dat:# or (len(can.dat) > 2 and can.dat[1] == 0x3e+0x40):
        # if b'\x00' in can.dat:
        # if can.dat == 0x30080aaaaaaaaaaa:  # and can.src in (1, 129):
        # if can.address in (2012,) or b'4CNDC' in can.dat:
        # if can.address in [0x700 + i for i in range(256)]:
        # if can.address in [0x18da0000 + (i << 8) + 0xf1 for i in range(256)]:
        # if can.src in [129, 1]:
        if msg.logMonoTime != prev_mono_time:
          print('')
          prev_mono_time = msg.logMonoTime
        print(f"{msg.logMonoTime} rxaddr={can.address}, bus={can.src}, {round((msg.logMonoTime - start_mono_time) * 1e-6, 2)} ms, 0x{can.dat.hex()}, {can.dat}, {len(can.dat)=}")
        # print(f"rxaddr={can.address}, txaddr={hex(can.address + - (8 if can.src < 120 else 0))}, bus={can.src}, {round((msg.logMonoTime - start_mono_time) * 1e-6, 2)} ms, 0x{can.dat.hex()}, {can.dat}")
        # print('rxaddr:', can.address, hex(can.address + - 8), can.src, msg.logMonoTime, can.dat)

        # expected_response = [CHRYSLER_VERSION_RESPONSE][counter]
        # response_valid = can.dat[:len(expected_response)] == expected_response
        # print(response_valid, can.dat[:len(expected_response)], expected_response)
        # if response_valid:
        #   if counter + 1 < len(CHRYSLER_VERSION_REQUEST):
        #     print('SENDING: {}'.format([CHRYSLER_VERSION_REQUEST][counter + 1]))
        #     # msg.send(CHRYSLER_VERSION_REQUEST[counter + 1])
        #     counter += 1
        #   else:
        #     results = can.dat[len(expected_response):]

