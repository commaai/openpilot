#!/usr/bin/env python3
# pylint: skip-file

# This file is not used by openpilot. Only boardd.cc is used.
# The python version is slower, but has more options for development.

# TODO: merge the extra functionalities of this file (like MOCK) in boardd.c and
# delete this python version of boardd

import os
import struct
import time

import cereal.messaging as messaging
from common.realtime import Ratekeeper
from selfdrive.swaglog import cloudlog
from selfdrive.boardd.boardd import can_capnp_to_can_list
from cereal import car

SafetyModel = car.CarParams.SafetyModel

# USB is optional
try:
  import usb1
  from usb1 import USBErrorIO, USBErrorOverflow  # pylint: disable=no-name-in-module
except Exception:
  pass

# *** serialization functions ***
def can_list_to_can_capnp(can_msgs, msgtype='can'):
  dat = messaging.new_message(msgtype, len(can_msgs))
  for i, can_msg in enumerate(can_msgs):
    if msgtype == 'sendcan':
      cc = dat.sendcan[i]
    else:
      cc = dat.can[i]
    cc.address = can_msg[0]
    cc.busTime = can_msg[1]
    cc.dat = bytes(can_msg[2])
    cc.src = can_msg[3]
  return dat


# *** can driver ***
def can_health():
  while 1:
    try:
      dat = handle.controlRead(usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE, 0xd2, 0, 0, 0x16)
      break
    except (USBErrorIO, USBErrorOverflow):
      cloudlog.exception("CAN: BAD HEALTH, RETRYING")
  v, i = struct.unpack("II", dat[0:8])
  ign_line, ign_can = struct.unpack("BB", dat[20:22])
  return {"voltage": v, "current": i, "ignition_line": bool(ign_line), "ignition_can": bool(ign_can)}

def __parse_can_buffer(dat):
  ret = []
  for j in range(0, len(dat), 0x10):
    ddat = dat[j:j+0x10]
    f1, f2 = struct.unpack("II", ddat[0:8])
    ret.append((f1 >> 21, f2 >> 16, ddat[8:8 + (f2 & 0xF)], (f2 >> 4) & 0xFF))
  return ret

def can_send_many(arr):
  snds = []
  for addr, _, dat, alt in arr:
    if addr < 0x800:  # only support 11 bit addr
      snd = struct.pack("II", ((addr << 21) | 1), len(dat) | (alt << 4)) + dat
      snd = snd.ljust(0x10, b'\x00')
      snds.append(snd)
  while 1:
    try:
      handle.bulkWrite(3, b''.join(snds))
      break
    except (USBErrorIO, USBErrorOverflow):
      cloudlog.exception("CAN: BAD SEND MANY, RETRYING")

def can_recv():
  dat = b""
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
      handle.controlWrite(0x40, 0xdc, SafetyModel.allOutput, 0, b'')

  if handle is None:
    cloudlog.warning("CAN NOT FOUND")
    exit(-1)

  cloudlog.info("got handle")
  cloudlog.info("can init done")

def boardd_mock_loop():
  can_init()
  handle.controlWrite(0x40, 0xdc, SafetyModel.allOutput, 0, b'')

  logcan = messaging.sub_sock('can')
  sendcan = messaging.pub_sock('sendcan')

  while 1:
    tsc = messaging.drain_sock(logcan, wait_for_one=True)
    snds = map(lambda x: can_capnp_to_can_list(x.can), tsc)
    snd = []
    for s in snds:
      snd += s
    snd = list(filter(lambda x: x[-1] <= 2, snd))
    snd_0 = len(list(filter(lambda x: x[-1] == 0, snd)))
    snd_1 = len(list(filter(lambda x: x[-1] == 1, snd)))
    snd_2 = len(list(filter(lambda x: x[-1] == 2, snd)))
    can_send_many(snd)

    # recv @ 100hz
    can_msgs = can_recv()
    got_0 = len(list(filter(lambda x: x[-1] == 0+0x80, can_msgs)))
    got_1 = len(list(filter(lambda x: x[-1] == 1+0x80, can_msgs)))
    got_2 = len(list(filter(lambda x: x[-1] == 2+0x80, can_msgs)))
    print("sent %3d (%3d/%3d/%3d) got %3d (%3d/%3d/%3d)" %
      (len(snd), snd_0, snd_1, snd_2, len(can_msgs), got_0, got_1, got_2))
    m = can_list_to_can_capnp(can_msgs, msgtype='sendcan')
    sendcan.send(m.to_bytes())

def boardd_test_loop():
  can_init()
  cnt = 0
  while 1:
    can_send_many([[0xbb, 0, "\xaa\xaa\xaa\xaa", 0], [0xaa, 0, f"ªªªª{struct.pack('!I', cnt)}", 1]])
    #can_send_many([[0xaa,0,"\xaa\xaa\xaa\xaa",0]])
    #can_send_many([[0xaa,0,"\xaa\xaa\xaa\xaa",1]])
    # recv @ 100hz
    can_msgs = can_recv()
    print("got %d" % (len(can_msgs)))
    time.sleep(0.01)
    cnt += 1

# *** main loop ***
def boardd_loop(rate=100):
  rk = Ratekeeper(rate)

  can_init()

  # *** publishes can and health
  logcan = messaging.pub_sock('can')
  health_sock = messaging.pub_sock('pandaState')

  # *** subscribes to can send
  sendcan = messaging.sub_sock('sendcan')

  # drain sendcan to delete any stale messages from previous runs
  messaging.drain_sock(sendcan)

  while 1:
    # health packet @ 2hz
    if (rk.frame % (rate // 2)) == 0:
      health = can_health()
      msg = messaging.new_message('pandaState')

      # store the health to be logged
      msg.pandaState.voltage = health['voltage']
      msg.pandaState.current = health['current']
      msg.pandaState.ignitionLine = health['ignition_line']
      msg.pandaState.ignitionCan = health['ignition_can']
      msg.pandaState.controlsAllowed = True

      health_sock.send(msg.to_bytes())

    # recv @ 100hz
    can_msgs = can_recv()

    # publish to logger
    # TODO: refactor for speed
    if len(can_msgs) > 0:
      dat = can_list_to_can_capnp(can_msgs).to_bytes()
      logcan.send(dat)

    # send can if we have a packet
    tsc = messaging.recv_sock(sendcan)
    if tsc is not None:
      can_send_many(can_capnp_to_can_list(tsc.sendcan))

    rk.keep_time()

# *** main loop ***
def boardd_proxy_loop(rate=100, address="192.168.2.251"):
  rk = Ratekeeper(rate)

  can_init()

  # *** subscribes can
  logcan = messaging.sub_sock('can', addr=address)
  # *** publishes to can send
  sendcan = messaging.pub_sock('sendcan')

  # drain sendcan to delete any stale messages from previous runs
  messaging.drain_sock(sendcan)

  while 1:
    # recv @ 100hz
    can_msgs = can_recv()
    #for m in can_msgs:
    #  print("R: {0} {1}".format(hex(m[0]), str(m[2]).encode("hex")))

    # publish to logger
    # TODO: refactor for speed
    if len(can_msgs) > 0:
      dat = can_list_to_can_capnp(can_msgs, "sendcan")
      sendcan.send(dat)

    # send can if we have a packet
    tsc = messaging.recv_sock(logcan)
    if tsc is not None:
      cl = can_capnp_to_can_list(tsc.can)
      #for m in cl:
      #  print("S: {0} {1}".format(hex(m[0]), str(m[2]).encode("hex")))
      can_send_many(cl)

    rk.keep_time()

def main():
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
