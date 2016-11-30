#!/usr/bin/env python
import zmq
from logentries import LogentriesHandler
from common.services import service_list
import selfdrive.messaging as messaging

def main(gctx):
  # setup logentries. we forward log messages to it
  le_token = "bc65354a-b887-4ef4-8525-15dd51230e8c"
  le_handler = LogentriesHandler(le_token, use_tls=False)

  le_level = 20 #logging.INFO

  ctx = zmq.Context()
  sock = ctx.socket(zmq.PULL)
  sock.bind("ipc:///tmp/logmessage")

  # and we publish them
  pub_sock = messaging.pub_sock(ctx, service_list['logMessage'].port)

  while True:
    dat = ''.join(sock.recv_multipart())

    # print "RECV", repr(dat)

    levelnum = ord(dat[0])
    dat = dat[1:]

    if levelnum >= le_level:
      # push to logentries
      le_handler.emit_raw(dat)

    # then we publish them
    msg = messaging.new_message()
    msg.logMessage = dat
    pub_sock.send(msg.to_bytes())

if __name__ == "__main__":
  main(None)
