#!/usr/bin/env python
import selfdrive.messaging as messaging
from selfdrive.boardd.boardd import can_list_to_can_capnp

VIN_UNKNOWN = "0" * 17

# sanity checks on response messages from vin query
def is_vin_response_valid(can_dat, step, cnt):

  can_dat = [ord(i) for i in can_dat]

  if len(can_dat) != 8:
    # ISO-TP meesages are all 8 bytes
    return False

  if step == 0:
    # VIN does not fit in a single message and it's 20 bytes of data
    if can_dat[0] != 0x10 or can_dat[1] != 0x14:
       return False

  if step == 1 and cnt == 0:
    # first response after a CONTINUE query is sent
    if can_dat[0] != 0x21:
       return False

  if step == 1 and cnt == 1:
    # second response after a CONTINUE query is sent
    if can_dat[0] != 0x22:
       return False

  return True


class VinQuery():
  def __init__(self, bus):
    self.bus = bus
    # works on standard 11-bit addresses for diagnostic. Tested on Toyota and Subaru;
    # Honda uses the extended 29-bit addresses, and unfortunately only works from OBDII
    self.query_ext_msgs = [[0x18DB33F1, 0, '\x02\x09\x02'.ljust(8, "\x00"), bus],
                           [0x18DA10f1, 0, '\x30'.ljust(8, "\x00"), bus]]
    self.query_nor_msgs = [[0x7df, 0, '\x02\x09\x02'.ljust(8, "\x00"), bus],
                           [0x7e0, 0, '\x30'.ljust(8, "\x00"), bus]]

    self.cnts = [1, 2]  # number of messages to wait for at each iteration
    self.step = 0
    self.cnt = 0
    self.responded = False
    self.never_responded = True
    self.dat = []
    self.vin = VIN_UNKNOWN

  def check_response(self, msg):
    # have we got a VIN query response?
    if msg.src == self.bus and msg.address in [0x18daf110, 0x7e8]:
      self.never_responded = False
      # basic sanity checks on ISO-TP response
      if is_vin_response_valid(msg.dat, self.step, self.cnt):
        self.dat += msg.dat[2:] if self.step == 0 else msg.dat[1:]
        self.cnt += 1
        if self.cnt == self.cnts[self.step]:
          self.responded = True
          self.step += 1

  def send_query(self, sendcan):
    # keep sending VIN qury if ECU isn't responsing.
    # sendcan is probably not ready due to the zmq slow joiner syndrome
    if self.never_responded or (self.responded and self.step < len(self.cnts)):
      sendcan.send(can_list_to_can_capnp([self.query_ext_msgs[self.step]], msgtype='sendcan'))
      sendcan.send(can_list_to_can_capnp([self.query_nor_msgs[self.step]], msgtype='sendcan'))
      self.responded = False
      self.cnt = 0

  def get_vin(self):
    # only report vin if procedure is finished
    if self.step == len(self.cnts) and self.cnt == self.cnts[-1]:
      self.vin = "".join(self.dat[3:])
    return self.vin


def get_vin(logcan, sendcan, bus, query_time=1.):
  vin_query = VinQuery(bus)
  frame = 0

  # 1s max of VIN query time
  while frame < query_time * 100:
    a = messaging.recv_one(logcan)

    for can in a.can:
      vin_query.check_response(can)

    vin_query.send_query(sendcan)
    frame += 1

  return vin_query.get_vin()


if __name__ == "__main__":
  from selfdrive.services import service_list
  logcan = messaging.sub_sock(service_list['can'].port)
  sendcan = messaging.pub_sock(service_list['sendcan'].port)
  print get_vin(logcan, sendcan, 0)
