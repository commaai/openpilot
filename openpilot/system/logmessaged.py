#!/usr/bin/env python3
import zmq
from typing import NoReturn

import openpilot.cereal.messaging as messaging
from openpilot.common.logging_extra import SwagLogFileFormatter
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import get_file_handler


# msgq queues are 256KiB and msgq asserts that a single message fits in a third
# of the queue. Records over this limit must be dropped before publishing, or
# logmessaged dies with SIGABRT - and the crash of the logging daemon is the
# one crash that can never be logged.
MAX_PUBLISH_BYTES = 80 * 1024


def main() -> NoReturn:
  log_handler = get_file_handler()
  log_handler.setFormatter(SwagLogFileFormatter(None))
  log_level = 20  # logging.INFO

  ctx = zmq.Context.instance()
  sock = ctx.socket(zmq.PULL)
  sock.bind(Paths.swaglog_ipc())

  # and we publish them
  log_message_sock = messaging.pub_sock('logMessage')
  error_log_message_sock = messaging.pub_sock('errorLogMessage')

  try:
    while True:
      dat = b''.join(sock.recv_multipart())
      level = dat[0]
      record = dat[1:].decode("utf-8")
      if level >= log_level:
        log_handler.emit(record)

      if len(record) > MAX_PUBLISH_BYTES:
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
    ctx.term()

    # can hit this if interrupted during a rollover
    try:
      log_handler.close()
    except ValueError:
      pass

if __name__ == "__main__":
  main()
