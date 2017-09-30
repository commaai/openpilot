import struct

from selfdrive.can.libdbc_py import libdbc, ffi

class CANPacker(object):
  def __init__(self, dbc_name):
    self.packer = libdbc.canpack_init(dbc_name)
    self.sig_names = {}

  def pack(self, addr, values):
    # values: [(signal_name, signal_value)]

    values_thing = []
    for name, value in values:
      if name not in self.sig_names:
        self.sig_names[name] = ffi.new("char[]", name)

      values_thing.append({
        'name': self.sig_names[name],
        'value': value
      })

    values_c = ffi.new("SignalPackValue[]", values_thing)

    return libdbc.canpack_pack(self.packer, addr, len(values_thing), values_c)

  def pack_bytes(self, addr, values):
    return struct.pack(">Q", self.pack(addr, values))


if __name__ == "__main__":
  cp = CANPacker("honda_civic_touring_2016_can")
  s = cp.pack_bytes(0x30c, [
    ("PCM_SPEED", 123),
    ("PCM_GAS", 10),
  ])
  print s.encode("hex")
