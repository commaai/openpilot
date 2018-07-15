#!/usr/bin/env python
import os
import struct
import zmq
import time

import selfdrive.messaging as messaging
from common.realtime import Ratekeeper
from selfdrive.services import service_list
from selfdrive.swaglog import cloudlog

# USB is optional
try:
  import usb1
  from usb1 import USBErrorIO, USBErrorOverflow  #pylint: disable=no-name-in-module
except Exception:
  pass

# TODO: rewrite in C to save CPU

SAFETY_NOOUTPUT = 0
SAFETY_HONDA = 1
SAFETY_TOYOTA = 2
SAFETY_TOYOTA_NOLIMITS = 0x1336
SAFETY_ALLOUTPUT = 0x1337

# *** serialization functions ***
def can_list_to_can_capnp(can_msgs, msgtype='can'):
  dat = messaging.new_message()
  dat.init(msgtype, len(can_msgs))
  for i, can_msg in enumerate(can_msgs):
    if msgtype == 'sendcan':
      cc = dat.sendcan[i]
    else:
      cc = dat.can[i]
    cc.address = can_msg[0]
    cc.busTime = can_msg[1]
    cc.dat = str(can_msg[2])
    cc.src = can_msg[3]
  return dat

def can_capnp_to_can_list(can, src_filter=None):
  ret = []
  for msg in can:
    if src_filter is None or msg.src in src_filter:
      ret.append((msg.address, msg.busTime, msg.dat, msg.src))
  return ret

# *** can driver ***
def can_health():
  while 1:
    try:
      dat = handle.controlRead(usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE, 0xd2, 0, 0, 0x10)
      break
    except (USBErrorIO, USBErrorOverflow):
      cloudlog.exception("CAN: BAD HEALTH, RETRYING")
  v, i, started = struct.unpack("IIB", dat[0:9])
  # TODO: units
  return {"voltage": v, "current": i, "started": bool(started)}

def __parse_can_buffer(dat):
  ret = []
  for j in range(0, len(dat), 0x10):
    ddat = dat[j:j+0x10]
    f1, f2 = struct.unpack("II", ddat[0:8])
    ret.append((f1 >> 21, f2>>16, ddat[8:8+(f2&0xF)], (f2>>4)&0xF))
  return ret

def can_send_many(arr):
  snds = []
  for addr, _, dat, alt in arr:
    snd = struct.pack("II", ((addr << 21) | 1), len(dat) | (alt << 4)) + dat
    snd = snd.ljust(0x10, '\x00')
    snds.append(snd)
  while 1:
    try:
      handle.bulkWrite(3, ''.join(snds))
      break
    except (USBErrorIO, USBErrorOverflow):
      cloudlog.exception("CAN: BAD SEND MANY, RETRYING")

def can_recv():
  dat = ""
  while 1:
    try:
      dat = handle.bulkRead(1, 0x10*256)
      break
    except (USBErrorIO, USBErrorOverflow):
      cloudlog.exception("CAN: BAD RECV, RETRYING")
  return __parse_can_buffer(dat)

def can_init():
  global handle, context
  handle = None
  cloudlog.info("attempting can init")

  context = usb1.USBContext()
  #context.setDebug(9)

  for device in context.getDeviceList(skip_on_error=True):
    if device.getVendorID() == 0xbbaa and device.getProductID() == 0xddcc:
      handle = device.open()
      handle.claimInterface(0)
      handle.controlWrite(0x40, 0xdc, SAFETY_ALLOUTPUT, 0, b'')

  if handle is None:
    cloudlog.warn("CAN NOT FOUND")
    exit(-1)

  cloudlog.info("got handle")
  cloudlog.info("can init done")

def boardd_mock_loop():
  context = zmq.Context()
  can_init()
  handle.controlWrite(0x40, 0xdc, SAFETY_ALLOUTPUT, 0, b'')

  logcan = messaging.sub_sock(context, service_list['can'].port)
  sendcan = messaging.pub_sock(context, service_list['sendcan'].port)

  while 1:
    tsc = messaging.drain_sock(logcan, wait_for_one=True)
    snds = map(lambda x: can_capnp_to_can_list(x.can), tsc)
    snd = []
    for s in snds:
      snd += s
    snd = filter(lambda x: x[-1] <= 1, snd)
    can_send_many(snd)

    # recv @ 100hz
    can_msgs = can_recv()
    print("sent %d got %d" % (len(snd), len(can_msgs)))
    m = can_list_to_can_capnp(can_msgs)
    sendcan.send(m.to_bytes())

def boardd_test_loop():
  can_init()
  cnt = 0
  while 1:
    can_send_many([[0xbb,0,"\xaa\xaa\xaa\xaa",0], [0xaa,0,"\xaa\xaa\xaa\xaa"+struct.pack("!I", cnt),1]])
    #can_send_many([[0xaa,0,"\xaa\xaa\xaa\xaa",0]])
    #can_send_many([[0xaa,0,"\xaa\xaa\xaa\xaa",1]])
    # recv @ 100hz
    can_msgs = can_recv()
    print("got %d" % (len(can_msgs)))
    time.sleep(0.01)
    cnt += 1

# *** main loop ***
def boardd_loop(rate=200):
  rk = Ratekeeper(rate)
  context = zmq.Context()

  can_init()

  # *** publishes can and health
  logcan = messaging.pub_sock(context, service_list['can'].port)
  health_sock = messaging.pub_sock(context, service_list['health'].port)

  # *** subscribes to can send
  sendcan = messaging.sub_sock(context, service_list['sendcan'].port)

  while 1:
    # health packet @ 1hz
    if (rk.frame%rate) == 0:
      health = can_health()
      msg = messaging.new_message()
      msg.init('health')

      # store the health to be logged
      msg.health.voltage = health['voltage']
      msg.health.current = health['current']
      msg.health.started = health['started']

      health_sock.send(msg.to_bytes())

    # recv @ 100hz
    can_msgs = can_recv()

    # publish to logger
    # TODO: refactor for speed
    if len(can_msgs) > 0:
      dat = can_list_to_can_capnp(can_msgs)
      logcan.send(dat.to_bytes())

    # send can if we have a packet
    tsc = messaging.recv_sock(sendcan)
    if tsc is not None:
      can_send_many(can_capnp_to_can_list(tsc.sendcan))

    rk.keep_time()

# *** main loop ***
def boardd_proxy_loop(rate=200, address="192.168.2.251"):
  rk = Ratekeeper(rate)
  context = zmq.Context()

  can_init()

  # *** subscribes can
  logcan = messaging.sub_sock(context, service_list['can'].port, addr=address)
  # *** publishes to can send
  sendcan = messaging.pub_sock(context, service_list['sendcan'].port)

  while 1:
    # recv @ 100hz
    can_msgs = can_recv()
    #for m in can_msgs:
    #  print "R:",hex(m[0]), str(m[2]).encode("hex")

    # publish to logger
    # TODO: refactor for speed
    if len(can_msgs) > 0:
      dat = can_list_to_can_capnp(can_msgs, "sendcan")
      sendcan.send(dat.to_bytes())

    # send can if we have a packet
    tsc = messaging.recv_sock(logcan)
    if tsc is not None:
      cl = can_capnp_to_can_list(tsc.can)
      #for m in cl:
      #  print "S:",hex(m[0]), str(m[2]).encode("hex")
      can_send_many(cl)

    rk.keep_time()

def main(gctx=None):
  if os.getenv("MOCK") is not None:
    boardd_mock_loop()
  elif os.getenv("PROXY") is not None:
    boardd_proxy_loop()
  elif os.getenv("BOARDTEST") is not None:
    boardd_test_loop()
  else:
    boardd_loop()

if __name__ == "__main__":
  main()
