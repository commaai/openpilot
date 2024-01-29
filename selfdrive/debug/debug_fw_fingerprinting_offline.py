from tools.lib.logreader import LogReader
from cereal import messaging
from tqdm import tqdm
import argparse
import time
from panda.python import uds


lr = list(LogReader("5130484aa8069bad|2024-01-26--21-05-14/0/r"))  # civic 22 marco dingo missing radar


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
      addrs = [0x18dab0f1]
      addrs = addrs + [uds.get_rx_addr_for_tx_addr(addr) for addr in addrs]
      if can.address in addrs:
        if msg.logMonoTime != prev_mono_time:
          print()
          prev_mono_time = msg.logMonoTime
        print(f"{msg.logMonoTime} rxaddr={can.address}, bus={can.src}, {round((msg.logMonoTime - start_mono_time) * 1e-6, 2)} ms, 0x{can.dat.hex()}, {can.dat}, {len(can.dat)=}")


if __name__ == "__main__":
  # argparse:
  parser = argparse.ArgumentParser(description='View ISO-TP communication between various ECUs given an address')
  parser.add_argument('route', help='Route name')
  parser.add_argument('address', help='tx address (0x7e0 for engine)')
