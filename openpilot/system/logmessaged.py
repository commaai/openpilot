#!/usr/bin/env python3
import os
import socket
from typing import NoReturn

import openpilot.cereal.messaging as messaging
from openpilot.common.logging_extra import SwagLogFileFormatter
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import SWAGLOG_MAX_SIZE, cloudlog, get_file_handler


def main() -> NoReturn:
  log_handler = get_file_handler()
  log_handler.setFormatter(SwagLogFileFormatter(None))
  log_level = 20  # logging.INFO

  path = Paths.swaglog_ipc()
  if os.path.exists(path):
    os.unlink(path)
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SWAGLOG_MAX_SIZE)
  sock.bind(path)
  buf = bytearray(SWAGLOG_MAX_SIZE)

  # senders' SO_SNDBUF caps the log size, and the kernel may clamp it (net.core.wmem_max on Linux)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SWAGLOG_MAX_SIZE)
  max_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
  if max_size < 256*1024:
    cloudlog.warning(f"logs bigger than {max_size} bytes will be dropped")

  # and we publish them
  log_message_sock = messaging.pub_sock('logMessage')
  error_log_message_sock = messaging.pub_sock('errorLogMessage')

  try:
    while True:
      n = sock.recv_into(buf)
      if n == 0:
        continue
      level = buf[0]
      record = buf[1:n].decode("utf-8")
      if level >= log_level:
        log_handler.emit(record)

      if len(record) > 2*1024*1024:
        print("WARNING: log too big to publish", len(record))
        print(record[:100])
        continue

      # then we publish them
      msg = messaging.new_message(None, valid=True, logMessage=record)
      log_message_sock.send(msg.to_bytes())

      if level >= 40:  # logging.ERROR
        msg = messaging.new_message(None, valid=True, errorLogMessage=record)
        error_log_message_sock.send(msg.to_bytes())
  finally:
    sock.close()

    # can hit this if interrupted during a rollover
    try:
      log_handler.close()
    except ValueError:
      pass

if __name__ == "__main__":
  main()
