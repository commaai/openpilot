#!/usr/bin/env python3

import csv
import time
from panda import Panda

def can_logger():
  p = Panda()

  try:
    outputfile = open('output.csv', 'w')
    csvwriter = csv.writer(outputfile)
    # Write Header
    csvwriter.writerow(['Bus', 'MessageID', 'Message', 'MessageLength', 'Time'])
    print("Writing csv file output.csv. Press Ctrl-C to exit...\n")

    bus0_msg_cnt = 0
    bus1_msg_cnt = 0
    bus2_msg_cnt = 0

    start_time = time.time()
    while True:
      can_recv = p.can_recv()

      for address, dat, src in can_recv:
        csvwriter.writerow(
          [str(src), str(hex(address)), f"0x{dat.hex()}", len(dat), str(time.time() - start_time)])

        if src == 0:
          bus0_msg_cnt += 1
        elif src == 1:
          bus1_msg_cnt += 1
        elif src == 2:
          bus2_msg_cnt += 1

        print(f"Message Counts... Bus 0: {bus0_msg_cnt} Bus 1: {bus1_msg_cnt} Bus 2: {bus2_msg_cnt}", end='\r')

  except KeyboardInterrupt:
    print(f"\nNow exiting. Final message Counts... Bus 0: {bus0_msg_cnt} Bus 1: {bus1_msg_cnt} Bus 2: {bus2_msg_cnt}")
    outputfile.close()

if __name__ == "__main__":
  can_logger()
