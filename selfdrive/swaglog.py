import os
from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler

import zmq

from common.logging_extra import SwagLogger, SwagFormatter
from selfdrive.hardware import PC

if PC:
  SWAGLOG_DIR = os.path.join(str(Path.home()), ".comma", "log")
else:
  SWAGLOG_DIR = "/data/log/"

def get_file_handler():
  Path(SWAGLOG_DIR).mkdir(parents=True, exist_ok=True)
  file_name = os.path.join(SWAGLOG_DIR, "swaglog")
  handler = TimedRotatingFileHandler(file_name, when="M", interval=1, backupCount=2000)
  return handler

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


def add_file_handler(log):
  """
  Function to add the file log handler to swaglog.
  This can be used to store logs when logmessaged is not running.
  """
  handler = get_file_handler()
  handler.setFormatter(SwagFormatter(log))
  log.addHandler(handler)


cloudlog = log = SwagLogger()
log.setLevel(logging.DEBUG)

outhandler = logging.StreamHandler()
log.addHandler(outhandler)
log.addHandler(LogMessageHandler(SwagFormatter(log)))
