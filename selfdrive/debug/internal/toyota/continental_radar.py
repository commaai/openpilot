#!/usr/bin/env python3
import zmq
import time
from collections import defaultdict, OrderedDict

from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.toyota.toyotacan import make_can_msg
import cereal.messaging as messaging
from cereal.services import service_list


def send(sendcan, addr, m):
   packet = make_can_msg(addr, m, 0, False)
   packets = can_list_to_can_capnp([packet], msgtype='sendcan')
   sendcan.send(packets.to_bytes())


def recv_timeout(can, addr):
   received = False
   r = []
   t = time.time()

   while not received:
      c = messaging.recv_one_or_none(can)

      if c is not None:
         for msg in c.can:
            if msg.address == addr:
               r.append(msg)
               received = True

      if time.time() - t > 0.1:
         received = True

   return r


can = messaging.sub_sock('can')
sendcan = messaging.pub_sock('sendcan')

PUBLIC = 0
PRIVATE = 1

time.sleep(0.5)

# 1, 112

TEST_ON = "\xFF\x02\x10\x70\x00\x00\x00\x00"
POLL = "\xFF\x02\x21\x69\x00\x00\x00\x00"
send(sendcan, 1872, TEST_ON)
r = recv_timeout(can, 1880)
print(r)


for i in range(0, 256):
   send(sendcan, 1872, POLL.replace('\x69', chr(i)))
   r = recv_timeout(can, 1880)
   if len(r):
      print(i, end=' ')
      for m in r:
         print(m.dat.encode('hex'))
