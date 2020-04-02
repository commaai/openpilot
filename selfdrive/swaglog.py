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
    self.sock.setsockopt(zmq.LINGER, 10)
    self.sock.connect("ipc:///tmp/logmessage")
    self.pid = os.getpid()

  def emit(self, record):
    if os.getpid() != self.pid:
      self.connect()

    msg = self.format(record).rstrip('\n')
    # print("SEND".format(repr(msg)))
    try:
      s = chr(record.levelno)+msg
      self.sock.send(s.encode('utf8'), zmq.NOBLOCK)
    except zmq.error.Again:
      # drop :/
      pass


def add_logentries_handler(log):
  """Function to add the logentries handler to swaglog.
  This can be used to send logs when logmessaged is not running."""
  from selfdrive.logmessaged import get_le_handler
  handler = get_le_handler()
  handler.setFormatter(SwagFormatter(log))
  log.addHandler(handler)


cloudlog = log = SwagLogger()
log.setLevel(logging.DEBUG)

outhandler = logging.StreamHandler()
log.addHandler(outhandler)
log.addHandler(LogMessageHandler(SwagFormatter(log)))
