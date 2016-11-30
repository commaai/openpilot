#!/usr/bin/env python
import os
import json
import zmq

import common.realtime as realtime
from common.services import service_list
from selfdrive.swaglog import cloudlog
import selfdrive.messaging as messaging

import uploader
from logger import Logger

from selfdrive.loggerd.config import ROOT, SEGMENT_LENGTH


def gen_init_data(gctx):
  msg = messaging.new_message()

  kernel_args = open("/proc/cmdline", "r").read().strip().split(" ")
  msg.initData.kernelArgs = kernel_args

  msg.initData.gctx = json.dumps(gctx)
  if os.getenv('DONGLE_ID'):
    msg.initData.dongleId = os.getenv('DONGLE_ID')

  return msg.to_bytes()

def main(gctx=None):
  logger = Logger(ROOT, gen_init_data(gctx))

  context = zmq.Context()
  poller = zmq.Poller()

  # we push messages to visiond to rotate image recordings
  vision_control_sock = context.socket(zmq.PUSH)
  vision_control_sock.connect("tcp://127.0.0.1:8001")

  # register listeners for all services
  for service in service_list.itervalues():
    if service.should_log and service.port is not None:
      messaging.sub_sock(context, service.port, poller)

  uploader.clear_locks(ROOT)

  cur_dir, cur_part = logger.start()
  try:
    cloudlog.info("starting in dir %r", cur_dir)

    rotate_msg = messaging.log.LogRotate.new_message()
    rotate_msg.segmentNum = cur_part
    rotate_msg.path = cur_dir
    vision_control_sock.send(rotate_msg.to_bytes())

    last_rotate = realtime.sec_since_boot()
    while True:
      polld = poller.poll(timeout=1000)
      for sock, mode in polld:
        if mode != zmq.POLLIN:
          continue
        dat = sock.recv()

        # print "got", len(dat), realtime.sec_since_boot()
        # logevent = log_capnp.Event.from_bytes(dat)
        # print str(logevent)
        logger.log_data(dat)

      t = realtime.sec_since_boot()
      if (t - last_rotate) > SEGMENT_LENGTH:
        last_rotate += SEGMENT_LENGTH

        cur_dir, cur_part = logger.rotate()
        cloudlog.info("rotated to %r", cur_dir)

        rotate_msg = messaging.log.LogRotate.new_message()
        rotate_msg.segmentNum = cur_part
        rotate_msg.path = cur_dir

        vision_control_sock.send(rotate_msg.to_bytes())

  finally:
    logger.stop()

if __name__ == "__main__":
  main()

