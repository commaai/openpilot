#!/usr/bin/env python
from __future__ import print_function
import binascii
import csv
import time
import sys
from panda import Panda

def can_logger():

  try:
    print("Trying to connect to Panda over USB...")
    p = Panda()

  except AssertionError:
    print("USB connection failed. Trying WiFi...")

    try:
      p = Panda("WIFI")
    except:
      print("WiFi connection timed out. Please make sure your Panda is connected and try again.")
      sys.exit(0)

  try:
    outputfile = open('output.csv', 'wb')
    csvwriter = csv.writer(outputfile)
    #Write Header
    csvwriter.writerow(['Time', 'MessageID', 'Bus', 'Message'])
    #csvwriter.writerow(['Bus', 'MessageID', 'Message', 'MessageLength'])
    print("Writing csv file output.csv. Press Ctrl-C to exit...\n")

    bus0_msg_cnt = 0
    bus1_msg_cnt = 0
    bus2_msg_cnt = 0
    startTime = 0.0
    cycle_count = 0
    p.can_clear(0)
    p.can_clear(1)
    p.can_clear(2)
    while True:
      can_recv = p.can_recv() 
      cycle_count += 1
      for address, _, dat, src  in can_recv:
        if startTime == 0.0:
          startTime = time.time()
        csvwriter.writerow([str(time.time()-startTime), str(address), str(src), binascii.hexlify(dat)])

        if src == 0:
          bus0_msg_cnt += 1
        elif src == 1:
          bus1_msg_cnt += 1
        elif src == 2:
          bus2_msg_cnt += 1

        if (bus0_msg_cnt + bus1_msg_cnt + bus2_msg_cnt) % 1000 == 0: print("Message Counts... Bus 0: " + str(bus0_msg_cnt) + " Bus 1: " + str(bus1_msg_cnt) + " Bus 2: " + str(bus2_msg_cnt), end='\r')

  except KeyboardInterrupt:
    print("\nNow exiting. Final message Counts... Bus 0: " + str(bus0_msg_cnt) + " Bus 1: " + str(bus1_msg_cnt) + " Bus 2: " + str(bus2_msg_cnt))
    outputfile.close()

if __name__ == "__main__":
  can_logger()
