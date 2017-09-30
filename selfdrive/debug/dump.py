#!/usr/bin/env python
import sys
import argparse
import zmq
from hexdump import hexdump

import selfdrive.messaging as messaging
from selfdrive.services import service_list

if __name__ == "__main__":
  context = zmq.Context()
  poller = zmq.Poller()

  parser = argparse.ArgumentParser(description='Sniff a communcation socket')
  parser.add_argument('--raw', action='store_true')
  parser.add_argument("socket", type=str, nargs='*', help="socket name")
  args = parser.parse_args()

  for m in args.socket if len(args.socket) > 0 else service_list:
    messaging.sub_sock(context, service_list[m].port, poller)

  while 1:
    polld = poller.poll(timeout=1000)
    for sock, mode in polld:
      if mode != zmq.POLLIN:
        continue
      if args.raw:
        hexdump(sock.recv())
      else:
        print messaging.recv_one(sock)

