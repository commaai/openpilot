#!/usr/bin/env python
import sys
import argparse
import zmq
import json
from hexdump import hexdump

import selfdrive.messaging as messaging
from selfdrive.services import service_list

if __name__ == "__main__":
  context = zmq.Context()
  poller = zmq.Poller()

  parser = argparse.ArgumentParser(description='Sniff a communcation socket')
  parser.add_argument('--raw', action='store_true')
  parser.add_argument('--json', action='store_true')
  parser.add_argument('--addr', default='127.0.0.1')
  parser.add_argument("socket", type=str, nargs='*', help="socket name")
  args = parser.parse_args()

  for m in args.socket if len(args.socket) > 0 else service_list:
    if m in service_list:
      messaging.sub_sock(context, service_list[m].port, poller, addr=args.addr)
    elif m.isdigit():
      messaging.sub_sock(context, int(m), poller, addr=args.addr)
    else:
      print("service not found")
      exit(-1)

  while 1:
    polld = poller.poll(timeout=1000)
    for sock, mode in polld:
      if mode != zmq.POLLIN:
        continue
      if args.raw:
        hexdump(sock.recv())
      elif args.json:
        print(json.loads(sock.recv()))
      else:
        print messaging.recv_one(sock)

