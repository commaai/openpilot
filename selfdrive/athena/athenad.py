#!/usr/bin/env python2.7
import json
import jwt
import os
import random
import re
import select
import subprocess
import socket
import time
import threading
import traceback
import zmq
import requests
import six.moves.queue
from datetime import datetime, timedelta
from functools import partial
from jsonrpc import JSONRPCResponseManager, dispatcher
from websocket import create_connection, WebSocketTimeoutException, ABNF
from selfdrive.loggerd.config import ROOT

import selfdrive.crash as crash
import selfdrive.messaging as messaging
from common.api import Api
from common.params import Params
from selfdrive.services import service_list
from selfdrive.swaglog import cloudlog
from selfdrive.version import version, dirty

ATHENA_HOST = os.getenv('ATHENA_HOST', 'wss://athena.comma.ai')
HANDLER_THREADS = os.getenv('HANDLER_THREADS', 4)
LOCAL_PORT_WHITELIST = set([8022])

dispatcher["echo"] = lambda s: s
payload_queue = six.moves.queue.Queue()
response_queue = six.moves.queue.Queue()

def handle_long_poll(ws):
  end_event = threading.Event()

  threads = [
    threading.Thread(target=ws_recv, args=(ws, end_event)),
    threading.Thread(target=ws_send, args=(ws, end_event))
  ] + [
    threading.Thread(target=jsonrpc_handler, args=(end_event,))
    for x in xrange(HANDLER_THREADS)
  ]

  map(lambda thread: thread.start(), threads)
  try:
    while not end_event.is_set():
      time.sleep(0.1)
  except (KeyboardInterrupt, SystemExit):
    end_event.set()
    raise
  finally:
    for i, thread in enumerate(threads):
      thread.join()

def jsonrpc_handler(end_event):
  dispatcher["startLocalProxy"] = partial(startLocalProxy, end_event)
  while not end_event.is_set():
    try:
      data = payload_queue.get(timeout=1)
      response = JSONRPCResponseManager.handle(data, dispatcher)
      response_queue.put_nowait(response)
    except six.moves.queue.Empty:
      pass
    except Exception as e:
      cloudlog.exception("athena jsonrpc handler failed")
      traceback.print_exc()
      response_queue.put_nowait(json.dumps({"error": str(e)}))

# security: user should be able to request any message from their car
# TODO: add service to, for example, start visiond and take a picture
@dispatcher.add_method
def getMessage(service=None, timeout=1000):
  if service is None or service not in service_list:
    raise Exception("invalid service")
  socket = messaging.sub_sock(service_list[service].port)
  socket.setsockopt(zmq.RCVTIMEO, timeout)
  ret = messaging.recv_one(socket)
  return ret.to_dict()

@dispatcher.add_method
def listDataDirectory():
  files = [os.path.relpath(os.path.join(dp, f), ROOT) for dp, dn, fn in os.walk(ROOT) for f in fn]
  return files

@dispatcher.add_method
def uploadFileToUrl(fn, url, headers):
  if len(fn) == 0 or fn[0] == '/' or '..' in fn:
    return 500
  with open(os.path.join(ROOT, fn), "rb") as f:
    ret = requests.put(url, data=f, headers=headers, timeout=10)
  return ret.status_code

def startLocalProxy(global_end_event, remote_ws_uri, local_port):
  try:
    cloudlog.event("athena startLocalProxy", remote_ws_uri=remote_ws_uri, local_port=local_port)

    if local_port not in LOCAL_PORT_WHITELIST:
      raise Exception("Requested local port not whitelisted")

    params = Params()
    dongle_id = params.get("DongleId")
    private_key = open("/persist/comma/id_rsa").read()
    identity_token = jwt.encode({'identity':dongle_id, 'exp': datetime.utcnow() + timedelta(hours=1)}, private_key, algorithm='RS256')

    ws = create_connection(remote_ws_uri,
                           cookie="jwt=" + identity_token,
                           enable_multithread=True)

    ssock, csock = socket.socketpair()
    local_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_sock.connect(('127.0.0.1', local_port))
    local_sock.setblocking(0)

    proxy_end_event = threading.Event()
    threads = [
      threading.Thread(target=ws_proxy_recv, args=(ws, local_sock, ssock, proxy_end_event, global_end_event)),
      threading.Thread(target=ws_proxy_send, args=(ws, local_sock, csock, proxy_end_event))
    ]

    map(lambda thread: thread.start(), threads)

    return {"success": 1}
  except Exception as e:
    traceback.print_exc()
    raise e

@dispatcher.add_method
def getPublicKey():
  if not os.path.isfile('/persist/comma/id_rsa.pub'):
    return None

  with open('/persist/comma/id_rsa.pub', 'r') as f:
    return f.read()

@dispatcher.add_method
def getSshAuthorizedKeys():
  with open('/system/comma/home/.ssh/authorized_keys', 'r') as f:
    return f.read()

@dispatcher.add_method
def getSimInfo():
  sim_state = subprocess.check_output(['getprop', 'gsm.sim.state']).strip().split(',')
  network_type = subprocess.check_output(['getprop', 'gsm.network.type']).strip().split(',')
  mcc_mnc = subprocess.check_output(['getprop', 'gsm.sim.operator.numeric']).strip() or None

  sim_id_aidl_out = subprocess.check_output(['service', 'call', 'iphonesubinfo', '11'])
  sim_id_aidl_lines = sim_id_aidl_out.split('\n')
  if len(sim_id_aidl_lines) > 3:
    sim_id_lines = sim_id_aidl_lines[1:4]
    sim_id_fragments = [re.search(r"'([0-9\.]+)'", line).group(1) for line in sim_id_lines]
    sim_id = reduce(lambda frag1, frag2: frag1.replace('.', '') + frag2.replace('.', ''), sim_id_fragments)
  else:
    sim_id = None

  return {
    'sim_id': sim_id,
    'mcc_mnc': mcc_mnc,
    'network_type': network_type,
    'sim_state': sim_state
  }

def ws_proxy_recv(ws, local_sock, ssock, end_event, global_end_event):
  while not (end_event.is_set() or global_end_event.is_set()):
    try:
      data = ws.recv()
      local_sock.sendall(data)
    except WebSocketTimeoutException:
      pass
    except Exception:
      traceback.print_exc()
      break

  ssock.close()
  end_event.set()

def ws_proxy_send(ws, local_sock, signal_sock, end_event):
  while not end_event.is_set():
    try:
      r, _, _ = select.select((local_sock, signal_sock), (), ())
      if r:
        if r[0].fileno() == signal_sock.fileno():
          # got end signal from ws_proxy_recv
          end_event.set()
          break
        data = local_sock.recv(4096)
        if not data:
          # local_sock is dead
          end_event.set()
          break

        ws.send(data, ABNF.OPCODE_BINARY)
    except Exception:
        traceback.print_exc()
        end_event.set()

def ws_recv(ws, end_event):
  while not end_event.is_set():
    try:
      data = ws.recv()
      payload_queue.put_nowait(data)
    except WebSocketTimeoutException:
      pass
    except Exception:
      traceback.print_exc()
      end_event.set()

def ws_send(ws, end_event):
  while not end_event.is_set():
    try:
      response = response_queue.get(timeout=1)
      ws.send(response.json)
    except six.moves.queue.Empty:
      pass
    except Exception:
      traceback.print_exc()
      end_event.set()

def backoff(retries):
  return random.randrange(0, min(128, int(2 ** retries)))

def main(gctx=None):
  params = Params()
  dongle_id = params.get("DongleId")
  ws_uri = ATHENA_HOST + "/ws/v2/" + dongle_id

  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty, is_eon=True)
  crash.install()

  private_key = open("/persist/comma/id_rsa").read()
  api = Api(dongle_id, private_key)

  conn_retries = 0
  while 1:
    try:
      print("connecting to %s" % ws_uri)
      ws = create_connection(ws_uri,
                             cookie="jwt=" + api.get_token(),
                             enable_multithread=True)
      ws.settimeout(1)
      conn_retries = 0
      handle_long_poll(ws)
    except (KeyboardInterrupt, SystemExit):
      break
    except Exception:
      conn_retries += 1
      traceback.print_exc()

    time.sleep(backoff(conn_retries))

  params.delete("AthenadPid")

if __name__ == "__main__":
  main()
