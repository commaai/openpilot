#!/usr/bin/env python3
import zmq
import time
from collections import defaultdict, OrderedDict

from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.toyota.toyotacan import make_can_msg
import cereal.messaging as messaging
from cereal.services import service_list

can = messaging.sub_sock('can')
sendcan = messaging.pub_sock('sendcan')


BEFORE = [
"\x10\x15\x30\x0B\x00\x00\x00\x00",
"\x21\x00\x00\x00\x00\x00\x00\x00",
]

LEFT = "\x22\x00\x00\x08\x00\x00\x00\x00"
RIGHT = "\x22\x00\x00\x04\x00\x00\x00\x00"
OFF = "\x22\x00\x00\x00\x00\x00\x00\x00"

AFTER = "\x23\x00\x00\x00\x00\x00\x00\x00"

i = 0
j = 0
while True:
   i += 1

   if i % 10 == 0:
     j += 1

   cur = RIGHT if j % 2 == 0 else OFF
   can_list = [make_can_msg(1984, d, 0, False) for d in BEFORE]
   can_list.append(make_can_msg(1984, cur, 0, False))
   can_list.append(make_can_msg(1984, AFTER, 0, False))

   for m in can_list:
     sendcan.send(can_list_to_can_capnp([m], msgtype='sendcan').to_bytes())
     time.sleep(0.01)
