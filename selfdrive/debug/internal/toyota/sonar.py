#!/usr/bin/env python3
import sys
import zmq
import os
import time
import random
from collections import defaultdict, OrderedDict

from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.toyota.toyotacan import make_can_msg
import cereal.messaging as messaging
from cereal.services import service_list

changing = []
fields = range(0, 256)
# fields = [225, 50, 39, 40]
fields = [50]
field_results = defaultdict(lambda: "\x00\x00")
cur_field = 97

def send(sendcan, addr, m):
   packet = make_can_msg(addr, m, 0, False)
   packets = can_list_to_can_capnp([packet], msgtype='sendcan')
   sendcan.send(packets.to_bytes())


def recv(can, addr):
   received = False
   r = []

   while not received:
      c = messaging.recv_one(can)
      for msg in c.can:
         if msg.address == addr:
            r.append(msg)
            received = True
   return r


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

      if time.time() - t > 0.05:
         received = True

   return r


def print_hex(d):
   s = map(ord, d)
   s = "".join(["{:02X}".format(b) for b in s])
   print(s)


TYPES = {
   0: 'single',
   1: 'first',
   2: 'consecutive',
   3: 'flow'
}

CONTINUE = "\x67\x30\x01\x00\x00\x00\x00\x00"
TEST_ON = "\x67\x02\x10\x74\x00\x00\x00\x00"
POLL = "\x67\x02\x21\x69\x00\x00\x00\x00"
# POLL = "\x67\x02\x10\x69\x00\x00\x00\x00"

prev_rcv_t = ""
recv_data = []
l = 0
index = 0


can = messaging.sub_sock('can')
sendcan = messaging.pub_sock('sendcan')

time.sleep(0.5)

results = []

test_mode = False

while True:
   # Send flow control if necessary
   if prev_rcv_t == 'first' or prev_rcv_t == 'consecutive':
      send(sendcan, 1872, CONTINUE)

   received = recv_timeout(can, 1880)

   if len(received) == 0:
      sys.stdout.flush()
      print(chr(27) + "[2J")
      print(time.time())
      print(changing)

      if len(results):
         if results[0] != "\x7F\x21\x31":
            old = field_results[cur_field]
            if old != '\x00\x00' and old != results[0] and cur_field not in changing:
               changing.append(cur_field)
            field_results[cur_field] = results[0]
         else:
            fields.remove(cur_field)

      for k in fields:
         # if field_results[k] == "\x00\x00":
         #    continue
         print(k, end=' ')
         print_hex(field_results[k])
      results = []

      if not test_mode:
         send(sendcan, 1872, TEST_ON)
         test_mode = True
      else:
         cur_field = random.choice(fields)
         send(sendcan, 1872, POLL.replace('\x69', chr(cur_field)))

   for r in received:
      data = r.dat

      # Check message type
      t = TYPES[ord(data[1]) >> 4]
      if t == 'single':
         l = ord(data[1]) & 0x0F
      elif t == 'first':
         a = ord(data[1]) & 0x0F
         b = ord(data[2])
         l = b + (a << 8)
         recv_data = []

      prev_rcv_t = t

      if t == 'single':
         recv_data = data[2: 2 + l]
         results.append(recv_data)
      if t == 'first':
         index = 0
         recv_data += data[3: min(8, 3 + l)]
      if t == 'consecutive':
         index += 1
         assert index == ord(data[1]) & 0x0F

         pending_l = l - len(recv_data)
         recv_data += data[2: min(8, 2 + pending_l)]

         if len(recv_data) == l:
            prev_rcv_t = ""
            results.append(recv_data)
