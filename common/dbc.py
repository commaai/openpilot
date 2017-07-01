import re
import bitstring
import struct
from collections import namedtuple

def int_or_float(s):
  # return number, trying to maintain int format
  try:
    return int(s)
  except ValueError:
    return float(s)

DBCSignal = namedtuple(
  "DBCSignal", ["name", "start_bit", "size", "is_little_endian", "is_signed",
                "factor", "offset", "tmin", "tmax", "units"])

class dbc(object):
  def __init__(self, fn):
    with open(fn) as f:
      self.txt = f.read().split("\n")
    self._warned_addresses = set()

    # regexps from https://github.com/ebroecker/canmatrix/blob/master/canmatrix/importdbc.py
    bo_regexp = re.compile("^BO\_ (\w+) (\w+) *: (\w+) (\w+)")
    sg_regexp = re.compile("^SG\_ (\w+) : (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*)")
    sgm_regexp = re.compile("^SG\_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*)")

    # A dictionary which maps message ids to tuples ((name, size), signals).
    #   name is the ASCII name of the message.
    #   size is the size of the message in bytes.
    #   signals is a list signals contained in the message.
    # signals is a list of DBCSignal in order of increasing start_bit.
    self.msgs = {}

    self.bits = []
    for i in range(0, 64, 8):
      for j in range(7, -1, -1):
        self.bits.append(i+j)
    self.bits_index = dict(zip(self.bits, range(64)))

    for l in self.txt:
      l = l.strip()

      if l.startswith("BO_ "):
        # new group
        dat = bo_regexp.match(l)

        if dat is None:
          print "bad BO", l
        name = dat.group(2)
        size = int(dat.group(3))
        ids = int(dat.group(1), 0) # could be hex

        self.msgs[ids] = ((name, size), [])

      if l.startswith("SG_ "):
        # new signal
        dat = sg_regexp.match(l)
        go = 0
        if dat is None:
          dat = sgm_regexp.match(l)
          go = 1
        if dat is None:
          print "bad SG", l

        sgname = dat.group(1)
        start_bit = int(dat.group(go+2))
        signal_size = int(dat.group(go+3))
        is_little_endian = int(dat.group(go+4))==1
        is_signed = dat.group(go+5)=='-'
        factor = int_or_float(dat.group(go+6))
        offset = int_or_float(dat.group(go+7))
        tmin = int_or_float(dat.group(go+8))
        tmax = int_or_float(dat.group(go+9))
        units = dat.group(go+10)

        self.msgs[ids][1].append(
          DBCSignal(sgname, start_bit, signal_size, is_little_endian,
                    is_signed, factor, offset, tmin, tmax, units))

    for msg in self.msgs.viewvalues():
      msg[1].sort(key=lambda x: x.start_bit)

  def encode(self, msg_id, dd):
    """Encode a CAN message using the dbc.

       Inputs:
        msg_id: The message ID.
        dd: A dictionary mapping signal name to signal data.
    """
    # TODO: Stop using bitstring, which is super slow.
    msg_def = self.msgs[msg_id]
    size = msg_def[0][1]

    bsf = bitstring.Bits(hex="00"*size)
    for s in msg_def[1]:
      ival = dd.get(s.name)
      if ival is not None:
        ival = (ival / s.factor) - s.offset
        ival = int(round(ival))

        # should pack this
        if s.is_little_endian:
          ss = s.start_bit
        else:
          ss = self.bits_index[s.start_bit]


        if s.is_signed:
          tbs = bitstring.Bits(int=ival, length=s.size)
        else:
          tbs = bitstring.Bits(uint=ival, length=s.size)

        lpad = bitstring.Bits(bin="0b"+"0"*ss)
        rpad = bitstring.Bits(bin="0b"+"0"*(8*size-(ss+s.size)))
        tbs = lpad+tbs+rpad

        bsf |= tbs
    return bsf.tobytes()

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
      out = [None]*len(arr)

    msg = self.msgs.get(x[0])
    if msg is None:
      if x[0] not in self._warned_addresses:
        print("WARNING: Unknown message address {}".format(x[0]))
        self._warned_addresses.add(x[0])
      return None, None

    name = msg[0][0]
    if debug:
      print name

    blen = 8*len(x[2])

    st = x[2].rjust(8, '\x00')
    le, be = None, None

    for s in msg[1]:
      if arr is not None and s[0] not in arr:
        continue

      # big or little endian?
      #   see http://vi-firmware.openxcplatform.com/en/master/config/bit-numbering.html
      if s[3] is False:
        ss = self.bits_index[s[1]]
        if be is None:
          be = struct.unpack(">Q", st)[0]
        x2_int = be
        data_bit_pos = (blen - (ss + s[2]))
      else:
        if le is None:
          le = struct.unpack("<Q", st)[0]
        x2_int = le
        ss = s[1]
        data_bit_pos = ss

      if data_bit_pos < 0:
        continue
      ival = (x2_int >> data_bit_pos) & ((1 << (s[2])) - 1)

      if s[4] and (ival & (1<<(s[2]-1))): # signed
        ival -= (1<<s[2])

      # control the offset
      ival = (ival * s[5]) + s[6]
      #if debug:
      #  print "%40s  %2d %2d  %7.2f %s" % (s[0], s[1], s[2], ival, s[-1])

      if arr is None:
        out[s[0]] = ival
      else:
        out[arr.index(s[0])] = ival
    return name, out
    
