#!/usr/bin/env python3
import re
import os
import struct
import sys
import numbers
from collections import namedtuple, defaultdict

def int_or_float(s):
  # return number, trying to maintain int format
  if s.isdigit():
    return int(s, 10)
  else:
    return float(s)


DBCSignal = namedtuple(
  "DBCSignal", ["name", "start_bit", "size", "is_little_endian", "is_signed",
                "factor", "offset", "tmin", "tmax", "units"])


class dbc():
  def __init__(self, fn):
    self.name, _ = os.path.splitext(os.path.basename(fn))
    with open(fn, encoding="ascii") as f:
      self.txt = f.readlines()
    self._warned_addresses = set()

    # regexps from https://github.com/ebroecker/canmatrix/blob/master/canmatrix/importdbc.py
    bo_regexp = re.compile(r"^BO\_ (\w+) (\w+) *: (\w+) (\w+)")
    sg_regexp = re.compile(r"^SG\_ (\w+) : (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*)")
    sgm_regexp = re.compile(r"^SG\_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*)")
    val_regexp = re.compile(r"VAL\_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+\".+?\"[^;]*)")

    # A dictionary which maps message ids to tuples ((name, size), signals).
    #   name is the ASCII name of the message.
    #   size is the size of the message in bytes.
    #   signals is a list signals contained in the message.
    # signals is a list of DBCSignal in order of increasing start_bit.
    self.msgs = {}

    # A dictionary which maps message ids to a list of tuples (signal name, definition value pairs)
    self.def_vals = defaultdict(list)

    # lookup to bit reverse each byte
    self.bits_index = [(i & ~0b111) + ((-i - 1) & 0b111) for i in range(64)]

    for l in self.txt:
      l = l.strip()

      if l.startswith("BO_ "):
        # new group
        dat = bo_regexp.match(l)

        if dat is None:
          print("bad BO {0}".format(l))

        name = dat.group(2)
        size = int(dat.group(3))
        ids = int(dat.group(1), 0)  # could be hex
        if ids in self.msgs:
          sys.exit("Duplicate address detected %d %s" % (ids, self.name))

        self.msgs[ids] = ((name, size), [])

      if l.startswith("SG_ "):
        # new signal
        dat = sg_regexp.match(l)
        go = 0
        if dat is None:
          dat = sgm_regexp.match(l)
          go = 1

        if dat is None:
          print("bad SG {0}".format(l))

        sgname = dat.group(1)
        start_bit = int(dat.group(go + 2))
        signal_size = int(dat.group(go + 3))
        is_little_endian = int(dat.group(go + 4)) == 1
        is_signed = dat.group(go + 5) == '-'
        factor = int_or_float(dat.group(go + 6))
        offset = int_or_float(dat.group(go + 7))
        tmin = int_or_float(dat.group(go + 8))
        tmax = int_or_float(dat.group(go + 9))
        units = dat.group(go + 10)

        self.msgs[ids][1].append(
          DBCSignal(sgname, start_bit, signal_size, is_little_endian,
                    is_signed, factor, offset, tmin, tmax, units))

      if l.startswith("VAL_ "):
        # new signal value/definition
        dat = val_regexp.match(l)

        if dat is None:
          print("bad VAL {0}".format(l))

        ids = int(dat.group(1), 0)  # could be hex
        sgname = dat.group(2)
        defvals = dat.group(3)

        defvals = defvals.replace("?", r"\?")  # escape sequence in C++
        defvals = defvals.split('"')[:-1]

        # convert strings to UPPER_CASE_WITH_UNDERSCORES
        defvals[1::2] = [d.strip().upper().replace(" ", "_") for d in defvals[1::2]]
        defvals = '"' + "".join(str(i) for i in defvals) + '"'

        self.def_vals[ids].append((sgname, defvals))

    for msg in self.msgs.values():
      msg[1].sort(key=lambda x: x.start_bit)

    self.msg_name_to_address = {}
    for address, m in self.msgs.items():
      name = m[0][0]
      self.msg_name_to_address[name] = address

  def lookup_msg_id(self, msg_id):
    if not isinstance(msg_id, numbers.Number):
      msg_id = self.msg_name_to_address[msg_id]
    return msg_id

  def reverse_bytes(self, x):
    return ((x & 0xff00000000000000) >> 56) | \
           ((x & 0x00ff000000000000) >> 40) | \
           ((x & 0x0000ff0000000000) >> 24) | \
           ((x & 0x000000ff00000000) >> 8) | \
           ((x & 0x00000000ff000000) << 8) | \
           ((x & 0x0000000000ff0000) << 24) | \
           ((x & 0x000000000000ff00) << 40) | \
           ((x & 0x00000000000000ff) << 56)

  def encode(self, msg_id, dd):
    """Encode a CAN message using the dbc.

       Inputs:
        msg_id: The message ID.
        dd: A dictionary mapping signal name to signal data.
    """
    msg_id = self.lookup_msg_id(msg_id)

    msg_def = self.msgs[msg_id]
    size = msg_def[0][1]

    result = 0
    for s in msg_def[1]:
      ival = dd.get(s.name)
      if ival is not None:

        ival = (ival - s.offset) / s.factor
        ival = int(round(ival))

        if s.is_signed and ival < 0:
          ival = (1 << s.size) + ival

        if s.is_little_endian:
          shift = s.start_bit
        else:
          b1 = (s.start_bit // 8) * 8 + (-s.start_bit - 1) % 8
          shift = 64 - (b1 + s.size)

        mask = ((1 << s.size) - 1) << shift
        dat = (ival & ((1 << s.size) - 1)) << shift

        if s.is_little_endian:
          mask = self.reverse_bytes(mask)
          dat = self.reverse_bytes(dat)

        result &= ~mask
        result |= dat

    result = struct.pack('>Q', result)
    return result[:size]

  def decode(self, x, arr=None, debug=False):
    """Decode a CAN message using the dbc.

       Inputs:
        x: A collection with elements (address, time, data), where address is
           the CAN address, time is the bus time, and data is the CAN data as a
           hex string.
        arr: Optional list of signals which should be decoded and returned.
        debug: True to print debugging statements.

       Returns:
        A tuple (name, data), where name is the name of the CAN message and data
        is the decoded result. If arr is None, data is a dict of properties.
        Otherwise data is a list of the same length as arr.

        Returns (None, None) if the message could not be decoded.
    """

    if arr is None:
      out = {}
    else:
      out = [None] * len(arr)

    msg = self.msgs.get(x[0])
    if msg is None:
      if x[0] not in self._warned_addresses:
        # print("WARNING: Unknown message address {}".format(x[0]))
        self._warned_addresses.add(x[0])
      return None, None

    name = msg[0][0]
    if debug:
      print(name)

    st = x[2].ljust(8, b'\x00')
    le, be = None, None

    for s in msg[1]:
      if arr is not None and s[0] not in arr:
        continue

      start_bit = s[1]
      signal_size = s[2]
      little_endian = s[3]
      signed = s[4]
      factor = s[5]
      offset = s[6]

      if little_endian:
        if le is None:
          le = struct.unpack("<Q", st)[0]
        tmp = le
        shift_amount = start_bit
      else:
        if be is None:
          be = struct.unpack(">Q", st)[0]
        tmp = be
        b1 = (start_bit // 8) * 8 + (-start_bit - 1) % 8
        shift_amount = 64 - (b1 + signal_size)

      if shift_amount < 0:
        continue

      tmp = (tmp >> shift_amount) & ((1 << signal_size) - 1)
      if signed and (tmp >> (signal_size - 1)):
        tmp -= (1 << signal_size)

      tmp = tmp * factor + offset

      # if debug:
      #   print("%40s  %2d %2d  %7.2f %s" % (s[0], s[1], s[2], tmp, s[-1]))

      if arr is None:
        out[s[0]] = tmp
      else:
        out[arr.index(s[0])] = tmp
    return name, out

  def get_signals(self, msg):
    msg = self.lookup_msg_id(msg)
    return [sgs.name for sgs in self.msgs[msg][1]]
