#!/usr/bin/env python3

# Given an interesting CSV file of CAN messages and a list of background CAN
# messages, print which bits in the interesting file have never appeared
# in the background files.

# Expects the CSV file to be in one of the following formats:

# can_logger.py
# Bus,MessageID,Message,MessageLength
# 0,0x292,0x040000001068,6

# The old can_logger.py format is also supported:
# Bus,MessageID,Message
# 0,344,c000c00000000000

# Cabana "Save Log" format
# time,addr,bus,data
# 240.47911496100002,53,0,0acc0ade0074bf9e


import csv
import sys

class Message():
  """Details about a specific message ID."""

  def __init__(self, message_id):
    self.message_id = message_id
    self.data = {}  # keyed by hex string encoded message data
    self.ones = [0] * 64   # bit set if 1 is seen
    self.zeros = [0] * 64  # bit set if 0 has been seen

  def printBitDiff(self, other):
    """Prints bits that are set or cleared compared to other background."""
    for i in range(len(self.ones)):
      new_ones = ((~other.ones[i]) & 0xff) & self.ones[i]
      if new_ones:
        print('id %s new one  at byte %d bitmask %d' % (
            self.message_id, i, new_ones))
      new_zeros = ((~other.zeros[i]) & 0xff) & self.zeros[i]
      if new_zeros:
        print('id %s new zero at byte %d bitmask %d' % (
            self.message_id, i, new_zeros))


class Info():
  """A collection of Messages."""

  def __init__(self):
    self.messages = {}  # keyed by MessageID

  def load(self, filename):
    """Given a CSV file, adds information about message IDs and their values."""
    with open(filename) as inp:
      reader = csv.reader(inp)
      header = next(reader, None)
      if header[0] == 'time':
        self.cabana(reader)
      else:
        self.logger(reader)

  def cabana(self, reader):
    for row in reader:
      bus = row[2]
      message_id = hex(int(row[1]))[2:]
      message_id = f'{bus}:{message_id}'
      data = row[3]
      self.store(message_id, data)

  def logger(self, reader):
    for row in reader:
      bus = row[0]
      if row[1].startswith('0x'):
        message_id = row[1][2:]  # remove leading '0x'
      else:
        message_id = hex(int(row[1]))[2:]  # old message IDs are in decimal
      message_id = f'{bus}:{message_id}'
      if row[1].startswith('0x'):
        data = row[2][2:]  # remove leading '0x'
      else:
        data = row[2]
      self.store(message_id, data)

  def store(self, message_id, data):
      if message_id not in self.messages:
        self.messages[message_id] = Message(message_id)
      message = self.messages[message_id]
      if data not in self.messages[message_id].data:
        message.data[data] = True
      bts = bytearray.fromhex(data)
      for i in range(len(bts)):
        message.ones[i] = message.ones[i] | int(bts[i])
        # Inverts the data and masks it to a byte to get the zeros as ones.
        message.zeros[i] = message.zeros[i] | ((~int(bts[i])) & 0xff)


def PrintUnique(interesting_file, background_files):
  background = Info()
  for background_file in background_files:
    background.load(background_file)
  interesting = Info()
  interesting.load(interesting_file)
  for message_id in sorted(interesting.messages):
    if message_id not in background.messages:
      print('New message_id: %s' % message_id)
    else:
      interesting.messages[message_id].printBitDiff(
          background.messages[message_id])


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print('Usage:\n%s interesting.csv background*.csv' % sys.argv[0])
    sys.exit(0)
  PrintUnique(sys.argv[1], sys.argv[2:])
