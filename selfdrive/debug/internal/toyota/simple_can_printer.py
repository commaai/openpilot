#!/usr/bin/env python3
import zmq
from collections import OrderedDict
import cereal.messaging as messaging
from cereal.services import service_list

can = messaging.sub_sock('can')

addr = OrderedDict()

while True:
  c = messaging.recv_one(can)
  for msg in c.can:
    s = map(ord, msg.dat)
    s = "".join(["\\x{:02X}".format(b) for b in s])
    s = "\"" + s + "\","

    if msg.address == 1872:
      print("s:", s)
    if msg.address == 1880:
      print("r:", s)

    if msg.address not in addr:
       addr[msg.address] = list()
    if msg.dat not in addr[msg.address]:
      addr[msg.address].append(s)
