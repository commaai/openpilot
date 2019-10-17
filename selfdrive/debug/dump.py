#!/usr/bin/env python3
import sys
import argparse
import zmq
import json
from hexdump import hexdump
from threading import Thread

from cereal import log
import selfdrive.messaging as messaging
from selfdrive.services import service_list

def run_server(socketio):
  socketio.run(app, host='0.0.0.0', port=4000)

if __name__ == "__main__":
  poller = zmq.Poller()

  parser = argparse.ArgumentParser(description='Sniff a communcation socket')
  parser.add_argument('--pipe', action='store_true')
  parser.add_argument('--raw', action='store_true')
  parser.add_argument('--json', action='store_true')
  parser.add_argument('--dump-json', action='store_true')
  parser.add_argument('--no-print', action='store_true')
  parser.add_argument('--proxy', action='store_true', help='republish on localhost')
  parser.add_argument('--map', action='store_true')
  parser.add_argument('--addr', default='127.0.0.1')
  parser.add_argument('--values', help='values to monitor (instead of entire event)')
  parser.add_argument("socket", type=str, nargs='*', help="socket name")
  args = parser.parse_args()

  republish_socks = {}

  for m in args.socket if len(args.socket) > 0 else service_list:
    if m in service_list:
      port = service_list[m].port
    elif m.isdigit():
      port = int(m)
    else:
      print("service not found")
      sys.exit(-1)
    sock = messaging.sub_sock(port, poller, addr=args.addr)
    if args.proxy:
      republish_socks[sock] = messaging.pub_sock(port)

  if args.map:
    from flask.ext.socketio import SocketIO  #pylint: disable=no-name-in-module, import-error
    from flask import Flask
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode='threading')
    server_thread = Thread(target=run_server, args=(socketio,))
    server_thread.daemon = True
    server_thread.start()
    print('server running')

  values = None
  if args.values:
    values = [s.strip().split(".") for s in args.values.split(",")]

  while 1:
    polld = poller.poll(timeout=1000)
    for sock, mode in polld:
      if mode != zmq.POLLIN:
        continue
      msg = sock.recv()
      evt = log.Event.from_bytes(msg)
      if sock in republish_socks:
        republish_socks[sock].send(msg)
      if args.map and evt.which() == 'liveLocation':
        print('send loc')
        socketio.emit('location', {
          'lat': evt.liveLocation.lat,
          'lon': evt.liveLocation.lon,
          'alt': evt.liveLocation.alt,
        })
      if not args.no_print:
        if args.pipe:
          sys.stdout.write(msg)
          sys.stdout.flush()
        elif args.raw:
          hexdump(msg)
        elif args.json:
          print(json.loads(msg))
        elif args.dump_json:
          print(json.dumps(evt.to_dict()))
        elif values:
          print("logMonotime = {}".format(evt.logMonoTime))
          for value in values:
            if hasattr(evt, value[0]):
              item = evt
              for key in value:
                item = getattr(item, key)
              print("{} = {}".format(".".join(value), item))
          print("")
        else:
          print(evt)
