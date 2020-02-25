#!/usr/bin/env python3.7
import json
import os
import hashlib
import io
import random
import select
import socket
import time
import threading
import base64
import requests
import queue
from collections import namedtuple
from functools import partial
from jsonrpc import JSONRPCResponseManager, dispatcher
from websocket import create_connection, WebSocketTimeoutException, ABNF
from selfdrive.loggerd.config import ROOT

import cereal.messaging as messaging
from common import android
from common.basedir import PERSIST
from common.api import Api
from common.params import Params
from cereal.services import service_list
from selfdrive.swaglog import cloudlog

ATHENA_HOST = os.getenv('ATHENA_HOST', 'wss://athena.comma.ai')
HANDLER_THREADS = os.getenv('HANDLER_THREADS', 4)
LOCAL_PORT_WHITELIST = set([8022])

dispatcher["echo"] = lambda s: s
payload_queue = queue.Queue()
response_queue = queue.Queue()
upload_queue = queue.Queue()
cancelled_uploads = set()
UploadItem = namedtuple('UploadItem', ['path', 'url', 'headers', 'created_at', 'id'])

def handle_long_poll(ws):
  end_event = threading.Event()

  threads = [
    threading.Thread(target=ws_recv, args=(ws, end_event)),
    threading.Thread(target=ws_send, args=(ws, end_event)),
    threading.Thread(target=upload_handler, args=(end_event,))
  ] + [
    threading.Thread(target=jsonrpc_handler, args=(end_event,))
    for x in range(HANDLER_THREADS)
  ]

  for thread in threads:
    thread.start()
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
    except queue.Empty:
      pass
    except Exception as e:
      cloudlog.exception("athena jsonrpc handler failed")
      response_queue.put_nowait(json.dumps({"error": str(e)}))

def upload_handler(end_event):
  while not end_event.is_set():
    try:
      item = upload_queue.get(timeout=1)
      if item.id in cancelled_uploads:
        cancelled_uploads.remove(item.id)
        continue
      _do_upload(item)
    except queue.Empty:
      pass
    except Exception:
      cloudlog.exception("athena.upload_handler.exception")

def _do_upload(upload_item):
  with open(upload_item.path, "rb") as f:
    size = os.fstat(f.fileno()).st_size
    return requests.put(upload_item.url,
                        data=f,
                        headers={**upload_item.headers, 'Content-Length': str(size)},
                        timeout=10)

# security: user should be able to request any message from their car
@dispatcher.add_method
def getMessage(service=None, timeout=1000):
  if service is None or service not in service_list:
    raise Exception("invalid service")

  socket = messaging.sub_sock(service, timeout=timeout)
  ret = messaging.recv_one(socket)

  if ret is None:
    raise TimeoutError

  return ret.to_dict()

@dispatcher.add_method
def listDataDirectory():
  files = [os.path.relpath(os.path.join(dp, f), ROOT) for dp, dn, fn in os.walk(ROOT) for f in fn]
  return files

@dispatcher.add_method
def reboot():
  thermal_sock = messaging.sub_sock("thermal", timeout=1000)
  ret = messaging.recv_one(thermal_sock)
  if ret is None or ret.thermal.started:
    raise Exception("Reboot unavailable")

  def do_reboot():
    time.sleep(2)
    android.reboot()

  threading.Thread(target=do_reboot).start()

  return {"success": 1}

@dispatcher.add_method
def uploadFileToUrl(fn, url, headers):
  if len(fn) == 0 or fn[0] == '/' or '..' in fn:
    return 500
  path = os.path.join(ROOT, fn)
  if not os.path.exists(path):
    return 404

  item = UploadItem(path=path, url=url, headers=headers, created_at=int(time.time()*1000), id=None)
  upload_id = hashlib.sha1(str(item).encode()).hexdigest()
  item = item._replace(id=upload_id)

  upload_queue.put_nowait(item)

  return {"enqueued": 1, "item": item._asdict()}

@dispatcher.add_method
def listUploadQueue():
  return [item._asdict() for item in list(upload_queue.queue)]

@dispatcher.add_method
def cancelUpload(upload_id):
  upload_ids = set(item.id for item in list(upload_queue.queue))
  if upload_id not in upload_ids:
    return 404

  cancelled_uploads.add(upload_id)
  return {"success": 1}

def startLocalProxy(global_end_event, remote_ws_uri, local_port):
  try:
    if local_port not in LOCAL_PORT_WHITELIST:
      raise Exception("Requested local port not whitelisted")

    params = Params()
    dongle_id = params.get("DongleId").decode('utf8')
    identity_token = Api(dongle_id).get_token()
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
    for thread in threads:
      thread.start()

    return {"success": 1}
  except Exception as e:
    cloudlog.exception("athenad.startLocalProxy.exception")
    raise e

@dispatcher.add_method
def getPublicKey():
  if not os.path.isfile(PERSIST+'/comma/id_rsa.pub'):
    return None

  with open(PERSIST+'/comma/id_rsa.pub', 'r') as f:
    return f.read()

@dispatcher.add_method
def getSshAuthorizedKeys():
  return Params().get("GithubSshKeys", encoding='utf8') or ''

@dispatcher.add_method
def getSimInfo():
  sim_state = android.getprop("gsm.sim.state").split(",")
  network_type = android.getprop("gsm.network.type").split(',')
  mcc_mnc = android.getprop("gsm.sim.operator.numeric") or None

  sim_id = android.parse_service_call_string(android.service_call(['iphonesubinfo', '11']))
  cell_data_state = android.parse_service_call_unpack(android.service_call(['phone', '46']), ">q")
  cell_data_connected = (cell_data_state == 2)

  return {
    'sim_id': sim_id,
    'mcc_mnc': mcc_mnc,
    'network_type': network_type,
    'sim_state': sim_state,
    'data_connected': cell_data_connected
  }

@dispatcher.add_method
def takeSnapshot():
  from selfdrive.camerad.snapshot.snapshot import snapshot, jpeg_write
  ret = snapshot()
  if ret is not None:
    def b64jpeg(x):
      if x is not None:
        f = io.BytesIO()
        jpeg_write(f, x)
        return base64.b64encode(f.getvalue()).decode("utf-8")
      else:
        return None
    return {'jpegBack': b64jpeg(ret[0]),
            'jpegFront': b64jpeg(ret[1])}
  else:
    raise Exception("not available while camerad is started")

def ws_proxy_recv(ws, local_sock, ssock, end_event, global_end_event):
  while not (end_event.is_set() or global_end_event.is_set()):
    try:
      data = ws.recv()
      local_sock.sendall(data)
    except WebSocketTimeoutException:
      pass
    except Exception:
      cloudlog.exception("athenad.ws_proxy_recv.exception")
      break

  ssock.close()
  local_sock.close()
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
      cloudlog.exception("athenad.ws_proxy_send.exception")
      end_event.set()

def ws_recv(ws, end_event):
  while not end_event.is_set():
    try:
      data = ws.recv()
      payload_queue.put_nowait(data)
    except WebSocketTimeoutException:
      pass
    except Exception:
      cloudlog.exception("athenad.ws_recv.exception")
      end_event.set()

def ws_send(ws, end_event):
  while not end_event.is_set():
    try:
      response = response_queue.get(timeout=1)
      ws.send(response.json)
    except queue.Empty:
      pass
    except Exception:
      cloudlog.exception("athenad.ws_send.exception")
      end_event.set()

def backoff(retries):
  return random.randrange(0, min(128, int(2 ** retries)))

def main(gctx=None):
  params = Params()
  dongle_id = params.get("DongleId").decode('utf-8')
  ws_uri = ATHENA_HOST + "/ws/v2/" + dongle_id

  api = Api(dongle_id)

  conn_retries = 0
  while 1:
    try:
      ws = create_connection(ws_uri,
                             cookie="jwt=" + api.get_token(),
                             enable_multithread=True)
      cloudlog.event("athenad.main.connected_ws", ws_uri=ws_uri)
      ws.settimeout(1)
      conn_retries = 0
      handle_long_poll(ws)
    except (KeyboardInterrupt, SystemExit):
      break
    except Exception:
      cloudlog.exception("athenad.main.exception")
      conn_retries += 1

    time.sleep(backoff(conn_retries))

if __name__ == "__main__":
  main()
