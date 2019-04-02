#!/usr/bin/env python2

# This trims CAN message CSV files to just the messages relevant for Panda testing.
# Usage:
# cat input.csv | ./trim_csv.py > output.csv
import fileinput

addr_to_keep = [544, 0x1f4, 0x292]  # For Chrysler, update to the addresses that matter for you.

for line in fileinput.input():
  line = line.strip()
  cols = line.split(',')
  if len(cols) != 4:
    continue  # malformed, such as at the end or every 60s.
  (_, addr, bus, _) = cols
  if (addr == 'addr'):
    continue
  if (int(bus) == 128):  # Keep all messages sent by OpenPilot.
    print line
  elif (int(addr) in addr_to_keep):
    print line
