#!/usr/bin/env python3
import zmq
import cereal.messaging as messaging
from selfdrive.swaglog import get_file_handler


def main():
  log_handler = get_file_handler()
  le_level = 20  # logging.INFO

  ctx = zmq.Context().instance()
  sock = ctx.socket(zmq.PULL)
  sock.bind("ipc:///tmp/logmessage")

  # and we publish them
  pub_sock = messaging.pub_sock('logMessage')

  while True:
    dat = b''.join(sock.recv_multipart())
    dat = dat.decode('utf8')

    levelnum = ord(dat[0])
    dat = dat[1:]

    if levelnum >= le_level:
      log_handler.emit(dat)

    # then we publish them
    msg = messaging.new_message()
    msg.logMessage = dat
    pub_sock.send(msg.to_bytes())


if __name__ == "__main__":
  main()
