#!/usr/bin/env python3
import csv
import sys

class Message():
  """Details about a specific message ID."""

  def __init__(self, message_id):
    self.message_id = message_id
    self.ones = [0] * 8   # bit set if 1 is always seen
    self.zeros = [0] * 8  # bit set if 0 is always seen

  def printBitDiff(self, other):
    """Prints bits that transition from always zero to always 1 and vice versa."""
    for i in range(len(self.ones)):
      zero_to_one = other.zeros[i] & self.ones[i]
      if zero_to_one:
        print('id %s 0 -> 1 at byte %d bitmask %d' % (self.message_id, i, zero_to_one))
      one_to_zero = other.ones[i] & self.zeros[i]
      if one_to_zero:
        print('id %s 1 -> 0 at byte %d bitmask %d' % (self.message_id, i, one_to_zero))


class Info():
  """A collection of Messages."""

  def __init__(self):
    self.messages = {}  # keyed by MessageID

  def load(self, filename, start, end):
    """Given a CSV file, adds information about message IDs and their values."""
    with open(filename, 'rb') as inp:
      reader = csv.reader(inp)
      next(reader, None)  # skip the CSV header
      for row in reader:
        if not len(row):
          continue
        time = float(row[0])
        bus = int(row[2])
        if time < start or bus > 127:
          continue
        elif time > end:
          break
        if row[1].startswith('0x'):
          message_id = row[1][2:]  # remove leading '0x'
        else:
          message_id = hex(int(row[1]))[2:]  # old message IDs are in decimal
        message_id = '%s:%s' % (bus, message_id)
        if row[3].startswith('0x'):
          data = row[3][2:]  # remove leading '0x'
        else:
          data = row[3]
        new_message = False
        if message_id not in self.messages:
          self.messages[message_id] = Message(message_id)
          new_message = True
        message = self.messages[message_id]
        bts = bytearray.fromhex(data)
        for i in range(len(bts)):
          ones = int(bts[i])
          message.ones[i] = ones if new_message else message.ones[i] & ones
          # Inverts the data and masks it to a byte to get the zeros as ones.
          zeros = (~int(bts[i])) & 0xff
          message.zeros[i] = zeros if new_message else message.zeros[i] & zeros

def PrintUnique(log_file, low_range, high_range):
  # find messages with bits that are always low
  start, end = list(map(float, low_range.split('-')))
  low = Info()
  low.load(log_file, start, end)
  # find messages with bits that are always high
  start, end = list(map(float, high_range.split('-')))
  high = Info()
  high.load(log_file, start, end)
  # print messages that go from low to high
  found = False
  for message_id in sorted(high.messages):
    if message_id in low.messages:
      high.messages[message_id].printBitDiff(low.messages[message_id])
      found = True
  if not found:
    print('No messages that transition from always low to always high found!')

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print('Usage:\n%s log.csv <low-start>-<low-end> <high-start>-<high-end>' % sys.argv[0])
    sys.exit(0)
  PrintUnique(sys.argv[1], sys.argv[2], sys.argv[3])
