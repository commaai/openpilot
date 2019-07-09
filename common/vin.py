#!/usr/bin/env python
import time
from common.realtime import sec_since_boot
import selfdrive.messaging as messaging
from selfdrive.boardd.boardd import can_list_to_can_capnp

def get_vin(logcan, sendcan):

  # works on standard 11-bit addresses for diagnostic. Tested on Toyota and Subaru;
  # Honda uses the extended 29-bit addresses, and unfortunately only works from OBDII
  query_msg = [[0x7df, 0, '\x02\x09\x02'.ljust(8, "\x00"), 0],
               [0x7e0, 0, '\x30'.ljust(8, "\x00"), 0]]

  cnts = [1, 2]  # Number of messages to wait for at each iteration
  vin_valid = True

  dat = []
  for i in range(len(query_msg)):
    cnt = 0
    sendcan.send(can_list_to_can_capnp([query_msg[i]], msgtype='sendcan'))
    got_response = False
    t_start = sec_since_boot()
    while sec_since_boot() - t_start < 0.05 and not got_response:
      for a in messaging.drain_sock(logcan):
        for can in a.can:
          if can.src == 0 and can.address == 0x7e8:
            vin_valid = vin_valid and is_vin_response_valid(can.dat, i, cnt)
            dat += can.dat[2:] if i == 0 else can.dat[1:]
            cnt += 1
            if cnt == cnts[i]:
              got_response = True
      time.sleep(0.01)

  return "".join(dat[3:]) if vin_valid else ""

"""
if 'vin' not in gctx:
  print "getting vin"
  gctx['vin'] = query_vin()[3:]
  print "got VIN %s" % (gctx['vin'],)
  cloudlog.info("got VIN %s" % (gctx['vin'],))

# *** determine platform based on VIN ****
if vin.startswith("19UDE2F36G"):
  print "ACURA ILX 2016"
  self.civic = False
else:
  # TODO: add Honda check explicitly
  print "HONDA CIVIC 2016"
  self.civic = True

# *** special case VIN of Acura test platform
if vin == "19UDE2F36GA001322":
  print "comma.ai test platform detected"
  # it has a gas interceptor and a torque mod
  self.torque_mod = True
"""


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


if __name__ == "__main__":
  import zmq
  from selfdrive.services import service_list
  context = zmq.Context()
  logcan = messaging.sub_sock(context, service_list['can'].port)
  sendcan = messaging.pub_sock(context, service_list['sendcan'].port)
  time.sleep(1.)   # give time to sendcan socket to start

  print get_vin(logcan, sendcan)
