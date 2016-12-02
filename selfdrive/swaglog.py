import os
import logging

import zmq

from common.logging_extra import SwagLogger, SwagFormatter

class LogMessageHandler(logging.Handler):
  def __init__(self, formatter):
    logging.Handler.__init__(self)
    self.setFormatter(formatter)
    self.pid = None

  def connect(self):
    self.zctx = zmq.Context()
    self.sock = self.zctx.socket(zmq.PUSH)
    self.sock.connect("ipc:///tmp/logmessage")
    self.pid = os.getpid()

  def emit(self, record):
    if os.getpid() != self.pid:
      self.connect()

    msg = self.format(record).rstrip('\n')
    try:
      self.sock.send(chr(record.levelno)+msg, zmq.NOBLOCK)
    except zmq.error.Again:
      # If we are unable to send the message, we are as of now forced to drop it
      pass

cloudlog = log = SwagLogger()
log.setLevel(logging.DEBUG)

outhandler = logging.StreamHandler()
log.addHandler(outhandler)

log.addHandler(LogMessageHandler(SwagFormatter(log)))
