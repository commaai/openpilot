import struct
import numbers

from selfdrive.can.libdbc_py import libdbc, ffi


class CANPacker(object):
  def __init__(self, dbc_name):
    self.packer = libdbc.canpack_init(dbc_name)
    self.dbc = libdbc.dbc_lookup(dbc_name)
    self.sig_names = {}
    self.name_to_address_and_size = {}
    self.address_to_size = {}

    num_msgs = self.dbc[0].num_msgs
    for i in range(num_msgs):
      msg = self.dbc[0].msgs[i]

      name = ffi.string(msg.name)
      address = msg.address
      self.name_to_address_and_size[name] = (address, msg.size)
      self.address_to_size[address] = msg.size

  def pack(self, addr, values, counter):
    # values: [(signal_name, signal_value)]

    values_thing = []
    if isinstance(values, dict):
      values = values.items()

    for name, value in values:
      if name not in self.sig_names:
        self.sig_names[name] = ffi.new("char[]", name)

      values_thing.append({
        'name': self.sig_names[name],
        'value': value
      })

    values_c = ffi.new("SignalPackValue[]", values_thing)

    return libdbc.canpack_pack(self.packer, addr, len(values_thing), values_c, counter)

  def pack_bytes(self, addr, values, counter=-1):
    if isinstance(addr, numbers.Number):
      size = self.address_to_size[addr]
    else:
      addr, size = self.name_to_address_and_size[addr]

    val = self.pack(addr, values, counter)
    r = struct.pack(">Q", val)
    return addr, r[:size]

  def make_can_msg(self, addr, bus, values, counter=-1):
    addr, msg = self.pack_bytes(addr, values, counter)
    return [addr, 0, msg, bus]


if __name__ == "__main__":
  cp = CANPacker("honda_civic_touring_2016_can_generated")
  s = cp.pack_bytes(0x30c, [
    ("PCM_SPEED", 123),
    ("PCM_GAS", 10),
  ])
  print s.encode("hex")
