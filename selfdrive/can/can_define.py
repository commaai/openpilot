from collections import defaultdict
from selfdrive.can.libdbc_py import libdbc, ffi

class CANDefine():
  def __init__(self, dbc_name):
    self.dv = defaultdict(dict)
    self.dbc_name = dbc_name
    self.dbc = libdbc.dbc_lookup(dbc_name.encode('utf8'))

    num_vals = self.dbc[0].num_vals

    self.address_to_msg_name = {}
    num_msgs = self.dbc[0].num_msgs
    for i in range(num_msgs):
      msg = self.dbc[0].msgs[i]
      name = ffi.string(msg.name).decode('utf8')
      address = msg.address
      self.address_to_msg_name[address] = name

    for i in range(num_vals):
      val = self.dbc[0].vals[i]

      sgname = ffi.string(val.name).decode('utf8')
      address = val.address
      def_val = ffi.string(val.def_val).decode('utf8')

      #separate definition/value pairs
      def_val = def_val.split()
      values = [int(v) for v in def_val[::2]]
      defs = def_val[1::2]

      if address not in self.dv:
        self.dv[address] = {}
        msgname = self.address_to_msg_name[address]
        self.dv[msgname] = {}

      # two ways to lookup: address or msg name
      self.dv[address][sgname] = {v: d for v, d in zip(values, defs)} #build dict
      self.dv[msgname][sgname] = self.dv[address][sgname]
