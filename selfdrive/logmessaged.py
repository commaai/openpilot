#!/usr/bin/env python3
import zmq
from logentries import LogentriesHandler
import cereal.messaging as messaging


def get_le_handler():
  # setup logentries. we forward log messages to it
  le_token = "e8549616-0798-4d7e-a2ca-2513ae81fa17"
  return LogentriesHandler(le_token, use_tls=False, verbose=False)


def main():
  le_handler = get_le_handler()
  le_level = 20  # logging.INFO

  ctx = zmq.Context().instance()
  sock = ctx.socket(zmq.PULL)
  sock.bind("ipc:///tmp/logmessage")

  # and we publish them
  pub_sock = messaging.pub_sock('logMessage')

  while True:
    dat = b''.join(sock.recv_multipart())
    dat = dat.decode('utf8')

    # print "RECV", repr(dat)

    levelnum = ord(dat[0])
    dat = dat[1:]

    if levelnum >= le_level:
      # push to logentries
      # TODO: push to athena instead
      le_handler.emit_raw(dat)

    # then we publish them
    msg = messaging.new_message()
    msg.logMessage = dat
    pub_sock.send(msg.to_bytes())


if __name__ == "__main__":
  main()
