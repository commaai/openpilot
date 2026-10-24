#!/usr/bin/env python3
import os
import socket
from typing import NoReturn

import openpilot.cereal.messaging as messaging
from openpilot.common.logging_extra import SwagLogFileFormatter
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import SWAGLOG_BUF_SIZE, SWAGLOG_MAX_INLINE, get_file_handler


def main() -> NoReturn:
  log_handler = get_file_handler()
  log_handler.setFormatter(SwagLogFileFormatter(None))
  log_level = 20  # logging.INFO

  path = Paths.swaglog_ipc()
  if os.path.exists(path):
    os.unlink(path)
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SWAGLOG_BUF_SIZE)
  sock.bind(path)

  # and we publish them
  log_message_sock = messaging.pub_sock('logMessage')
  error_log_message_sock = messaging.pub_sock('errorLogMessage')

  try:
    while True:
      dat, fds, _, _ = socket.recv_fds(sock, SWAGLOG_MAX_INLINE, 1)
      if fds:
        # big logs come as a memfd
        dat = os.pread(fds[0], os.fstat(fds[0]).st_size, 0)
        for fd in fds:
          os.close(fd)
      if len(dat) == 0:
        continue
      level = dat[0]
      record = dat[1:].decode("utf-8")
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
