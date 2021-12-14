#!/usr/bin/env python3
import base64
import hashlib
import io
import json
import os
import sys
import queue
import random
import select
import socket
import threading
import time
from collections import namedtuple
from functools import partial
from typing import Any

import requests
from jsonrpc import JSONRPCResponseManager, dispatcher
from websocket import ABNF, WebSocketTimeoutException, WebSocketException, create_connection

import cereal.messaging as messaging
from cereal.services import service_list
from common.api import Api
from common.file_helpers import CallbackReader
from common.basedir import PERSIST
from common.params import Params
from common.realtime import sec_since_boot
from selfdrive.hardware import HARDWARE, PC
from selfdrive.loggerd.config import ROOT
from selfdrive.loggerd.xattr_cache import getxattr, setxattr
from selfdrive.swaglog import cloudlog, SWAGLOG_DIR
from selfdrive.version import get_version, get_origin, get_short_branch, get_commit

ATHENA_HOST = os.getenv('ATHENA_HOST', 'wss://athena.comma.ai')
HANDLER_THREADS = int(os.getenv('HANDLER_THREADS', "4"))
LOCAL_PORT_WHITELIST = set([8022])

LOG_ATTR_NAME = 'user.upload'
LOG_ATTR_VALUE_MAX_UNIX_TIME = int.to_bytes(2147483647, 4, sys.byteorder)
RECONNECT_TIMEOUT_S = 70

RETRY_DELAY = 10  # seconds
MAX_RETRY_COUNT = 30  # Try for at most 5 minutes if upload fails immediately
WS_FRAME_SIZE = 4096

dispatcher["echo"] = lambda s: s
recv_queue: Any = queue.Queue()
send_queue: Any = queue.Queue()
upload_queue: Any = queue.Queue()
log_send_queue: Any = queue.Queue()
log_recv_queue: Any = queue.Queue()
cancelled_uploads: Any = set()
UploadItem = namedtuple('UploadItem', ['path', 'url', 'headers', 'created_at', 'id', 'retry_count', 'current', 'progress'], defaults=(0, False, 0))

cur_upload_items = {}


class UploadQueueCache():
  params = Params()

  @staticmethod
  def initialize(upload_queue):
    upload_queue_json = UploadQueueCache.params.get("AthenadUploadQueue")
    try:
      for item in json.loads(upload_queue_json):
        upload_queue.put(UploadItem(**item))
    except json.JSONDecodeError:
      cloudlog.exception("athena.UploadQueueCache.initialize.exception")

  @staticmethod
  def cache(upload_queue):
    items = [i._asdict() for i in upload_queue.queue if i.id not in cancelled_uploads]
    UploadQueueCache.params.put("AthenadUploadQueue", json.dumps(items))


def handle_long_poll(ws):
  end_event = threading.Event()

  threads = [
    threading.Thread(target=ws_recv, args=(ws, end_event), name='ws_recv'),
    threading.Thread(target=ws_send, args=(ws, end_event), name='ws_send'),
    threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler'),
    threading.Thread(target=log_handler, args=(end_event,), name='log_handler'),
  ] + [
    threading.Thread(target=jsonrpc_handler, args=(end_event,), name=f'worker_{x}')
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
    for thread in threads:
      cloudlog.debug(f"athena.joining {thread.name}")
      thread.join()


def jsonrpc_handler(end_event):
  dispatcher["startLocalProxy"] = partial(startLocalProxy, end_event)
  while not end_event.is_set():
    try:
      data = recv_queue.get(timeout=1)
      if "method" in data:
        cloudlog.debug(f"athena.jsonrpc_handler.call_method {data}")
        response = JSONRPCResponseManager.handle(data, dispatcher)
        send_queue.put_nowait(response.json)
      elif "id" in data and ("result" in data or "error" in data):
        log_recv_queue.put_nowait(data)
      else:
        raise Exception("not a valid request or response")
    except queue.Empty:
      pass
    except Exception as e:
      cloudlog.exception("athena jsonrpc handler failed")
      send_queue.put_nowait(json.dumps({"error": str(e)}))


def upload_handler(end_event):
  tid = threading.get_ident()

  while not end_event.is_set():
    cur_upload_items[tid] = None

    try:
      cur_upload_items[tid] = upload_queue.get(timeout=1)._replace(current=True)

      if cur_upload_items[tid].id in cancelled_uploads:
        cancelled_uploads.remove(cur_upload_items[tid].id)
        continue

      try:
        def cb(sz, cur):
          cur_upload_items[tid] = cur_upload_items[tid]._replace(progress=cur / sz if sz else 1)

        _do_upload(cur_upload_items[tid], cb)
        UploadQueueCache.cache(upload_queue)
      except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
        cloudlog.warning(f"athena.upload_handler.retry {e} {cur_upload_items[tid]}")

        if cur_upload_items[tid].retry_count < MAX_RETRY_COUNT:
          item = cur_upload_items[tid]
          item = item._replace(
            retry_count=item.retry_count + 1,
            progress=0,
            current=False
          )
          upload_queue.put_nowait(item)
          UploadQueueCache.cache(upload_queue)

          cur_upload_items[tid] = None

          for _ in range(RETRY_DELAY):
            time.sleep(1)
            if end_event.is_set():
              break

    except queue.Empty:
      pass
    except Exception:
      cloudlog.exception("athena.upload_handler.exception")


def _do_upload(upload_item, callback=None):
  with open(upload_item.path, "rb") as f:
    size = os.fstat(f.fileno()).st_size

    if callback:
      f = CallbackReader(f, callback, size)

    return requests.put(upload_item.url,
                        data=f,
                        headers={**upload_item.headers, 'Content-Length': str(size)},
                        timeout=30)


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
def getVersion():
  return {
    "version": get_version(),
    "remote": get_origin(),
    "branch": get_short_branch(),
    "commit": get_commit(),
  }


@dispatcher.add_method
def setNavDestination(latitude=0, longitude=0, place_name=None, place_details=None):
  destination = {
    "latitude": latitude,
    "longitude": longitude,
    "place_name": place_name,
    "place_details": place_details,
  }
  Params().put("NavDestination", json.dumps(destination))

  return {"success": 1}


def scan_dir(path, prefix):
  files = list()
  # only walk directories that match the prefix
  # (glob and friends traverse entire dir tree)
  with os.scandir(path) as i:
    for e in i:
      rel_path = os.path.relpath(e.path, ROOT)
      if e.is_dir(follow_symlinks=False):
        # add trailing slash
        rel_path = os.path.join(rel_path, '')
        # if prefix is a partial dir name, current dir will start with prefix
        # if prefix is a partial file name, prefix with start with dir name
        if rel_path.startswith(prefix) or prefix.startswith(rel_path):
          files.extend(scan_dir(e.path, prefix))
      else:
        if rel_path.startswith(prefix):
          files.append(rel_path)
  return files

@dispatcher.add_method
def listDataDirectory(prefix=''):
  return scan_dir(ROOT, prefix)


@dispatcher.add_method
def reboot():
  sock = messaging.sub_sock("deviceState", timeout=1000)
  ret = messaging.recv_one(sock)
  if ret is None or ret.deviceState.started:
    raise Exception("Reboot unavailable")

  def do_reboot():
    time.sleep(2)
    HARDWARE.reboot()

  threading.Thread(target=do_reboot).start()

  return {"success": 1}


@dispatcher.add_method
def uploadFileToUrl(fn, url, headers):
  if len(fn) == 0 or fn[0] == '/' or '..' in fn:
    return 500
  path = os.path.join(ROOT, fn)
  if not os.path.exists(path):
    return 404

  item = UploadItem(path=path, url=url, headers=headers, created_at=int(time.time() * 1000), id=None)
  upload_id = hashlib.sha1(str(item).encode()).hexdigest()
  item = item._replace(id=upload_id)

  upload_queue.put_nowait(item)
  UploadQueueCache.cache(upload_queue)

  return {"enqueued": 1, "item": item._asdict()}


@dispatcher.add_method
def listUploadQueue():
  items = list(upload_queue.queue) + list(cur_upload_items.values())
  return [i._asdict() for i in items if (i is not None) and (i.id not in cancelled_uploads)]


@dispatcher.add_method
def cancelUpload(upload_id):
  upload_ids = set(item.id for item in list(upload_queue.queue))
  if upload_id not in upload_ids:
    return 404

  cancelled_uploads.add(upload_id)
  return {"success": 1}


@dispatcher.add_method
def primeActivated(activated):
  return {"success": 1}


def startLocalProxy(global_end_event, remote_ws_uri, local_port):
  try:
    if local_port not in LOCAL_PORT_WHITELIST:
      raise Exception("Requested local port not whitelisted")

    cloudlog.debug("athena.startLocalProxy.starting")

    dongle_id = Params().get("DongleId").decode('utf8')
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

    cloudlog.debug("athena.startLocalProxy.started")
    return {"success": 1}
  except Exception as e:
    cloudlog.exception("athenad.startLocalProxy.exception")
    raise e


@dispatcher.add_method
def getPublicKey():
  if not os.path.isfile(PERSIST + '/comma/id_rsa.pub'):
    return None

  with open(PERSIST + '/comma/id_rsa.pub', 'r') as f:
    return f.read()


@dispatcher.add_method
def getSshAuthorizedKeys():
  return Params().get("GithubSshKeys", encoding='utf8') or ''


@dispatcher.add_method
def getSimInfo():
  return HARDWARE.get_sim_info()


@dispatcher.add_method
def getNetworkType():
  return HARDWARE.get_network_type()


@dispatcher.add_method
def getNetworks():
  return HARDWARE.get_networks()


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


def get_logs_to_send_sorted():
  # TODO: scan once then use inotify to detect file creation/deletion
  curr_time = int(time.time())
  logs = []
  for log_entry in os.listdir(SWAGLOG_DIR):
    log_path = os.path.join(SWAGLOG_DIR, log_entry)
    try:
      time_sent = int.from_bytes(getxattr(log_path, LOG_ATTR_NAME), sys.byteorder)
    except (ValueError, TypeError):
      time_sent = 0
    # assume send failed and we lost the response if sent more than one hour ago
    if not time_sent or curr_time - time_sent > 3600:
      logs.append(log_entry)
  # excluding most recent (active) log file
  return sorted(logs)[:-1]


def log_handler(end_event):
  if PC:
    return

  log_files = []
  last_scan = 0
  while not end_event.is_set():
    try:
      curr_scan = sec_since_boot()
      if curr_scan - last_scan > 10:
        log_files = get_logs_to_send_sorted()
        last_scan = curr_scan

      # send one log
      curr_log = None
      if len(log_files) > 0:
        log_entry = log_files.pop() # newest log file
        cloudlog.debug(f"athena.log_handler.forward_request {log_entry}")
        try:
          curr_time = int(time.time())
          log_path = os.path.join(SWAGLOG_DIR, log_entry)
          setxattr(log_path, LOG_ATTR_NAME, int.to_bytes(curr_time, 4, sys.byteorder))
          with open(log_path, "r") as f:
            jsonrpc = {
              "method": "forwardLogs",
              "params": {
                "logs": f.read()
              },
              "jsonrpc": "2.0",
              "id": log_entry
            }
            log_send_queue.put_nowait(json.dumps(jsonrpc))
            curr_log = log_entry
        except OSError:
          pass  # file could be deleted by log rotation

      # wait for response up to ~100 seconds
      # always read queue at least once to process any old responses that arrive
      for _ in range(100):
        if end_event.is_set():
          break
        try:
          log_resp = json.loads(log_recv_queue.get(timeout=1))
          log_entry = log_resp.get("id")
          log_success = "result" in log_resp and log_resp["result"].get("success")
          cloudlog.debug(f"athena.log_handler.forward_response {log_entry} {log_success}")
          if log_entry and log_success:
            log_path = os.path.join(SWAGLOG_DIR, log_entry)
            try:
              setxattr(log_path, LOG_ATTR_NAME, LOG_ATTR_VALUE_MAX_UNIX_TIME)
            except OSError:
              pass  # file could be deleted by log rotation
          if curr_log == log_entry:
            break
        except queue.Empty:
          if curr_log is None:
            break

    except Exception:
      cloudlog.exception("athena.log_handler.exception")


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

  cloudlog.debug("athena.ws_proxy_recv closing sockets")
  ssock.close()
  local_sock.close()
  cloudlog.debug("athena.ws_proxy_recv done closing sockets")

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

  cloudlog.debug("athena.ws_proxy_send closing sockets")
  signal_sock.close()
  cloudlog.debug("athena.ws_proxy_send done closing sockets")


def ws_recv(ws, end_event):
  last_ping = int(sec_since_boot() * 1e9)
  while not end_event.is_set():
    try:
      opcode, data = ws.recv_data(control_frame=True)
      if opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
        if opcode == ABNF.OPCODE_TEXT:
          data = data.decode("utf-8")
        recv_queue.put_nowait(data)
      elif opcode == ABNF.OPCODE_PING:
        last_ping = int(sec_since_boot() * 1e9)
        Params().put("LastAthenaPingTime", str(last_ping))
    except WebSocketTimeoutException:
      ns_since_last_ping = int(sec_since_boot() * 1e9) - last_ping
      if ns_since_last_ping > RECONNECT_TIMEOUT_S * 1e9:
        cloudlog.exception("athenad.ws_recv.timeout")
        end_event.set()
    except Exception:
      cloudlog.exception("athenad.ws_recv.exception")
      end_event.set()


def ws_send(ws, end_event):
  while not end_event.is_set():
    try:
      try:
        data = send_queue.get_nowait()
      except queue.Empty:
        data = log_send_queue.get(timeout=1)
      for i in range(0, len(data), WS_FRAME_SIZE):
        frame = data[i:i+WS_FRAME_SIZE]
        last = i + WS_FRAME_SIZE >= len(data)
        opcode = ABNF.OPCODE_TEXT if i == 0 else ABNF.OPCODE_CONT
        ws.send_frame(ABNF.create_frame(frame, opcode, last))
    except queue.Empty:
      pass
    except Exception:
      cloudlog.exception("athenad.ws_send.exception")
      end_event.set()


def backoff(retries):
  return random.randrange(0, min(128, int(2 ** retries)))


def main():
  params = Params()
  dongle_id = params.get("DongleId", encoding='utf-8')
  UploadQueueCache.initialize(upload_queue)

  ws_uri = ATHENA_HOST + "/ws/v2/" + dongle_id
  api = Api(dongle_id)

  conn_retries = 0
  while 1:
    try:
      cloudlog.event("athenad.main.connecting_ws", ws_uri=ws_uri)
      ws = create_connection(ws_uri,
                             cookie="jwt=" + api.get_token(),
                             enable_multithread=True,
                             timeout=30.0)
      cloudlog.event("athenad.main.connected_ws", ws_uri=ws_uri)
      params.delete("PrimeRedirected")

      conn_retries = 0
      cur_upload_items.clear()

      handle_long_poll(ws)
    except (KeyboardInterrupt, SystemExit):
      break
    except (ConnectionError, TimeoutError, WebSocketException):
      conn_retries += 1
      params.delete("PrimeRedirected")
      params.delete("LastAthenaPingTime")
    except socket.timeout:
      try:
        r = requests.get("http://api.commadotai.com/v1/me", allow_redirects=False,
                         headers={"User-Agent": f"openpilot-{get_version()}"}, timeout=15.0)
        if r.status_code == 302 and r.headers['Location'].startswith("http://u.web2go.com"):
          params.put_bool("PrimeRedirected", True)
      except Exception:
        cloudlog.exception("athenad.socket_timeout.exception")
      params.delete("LastAthenaPingTime")
    except Exception:
      cloudlog.exception("athenad.main.exception")

      conn_retries += 1
      params.delete("PrimeRedirected")
      params.delete("LastAthenaPingTime")

    time.sleep(backoff(conn_retries))


if __name__ == "__main__":
  main()
