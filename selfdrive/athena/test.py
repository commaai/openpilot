#!/usr/bin/env python3
import json
import os
import requests
import tempfile
import time
import threading
import queue
import unittest

from multiprocessing import Process
from pathlib import Path
from unittest import mock
from websocket import ABNF
from websocket._exceptions import WebSocketConnectionClosedException

from selfdrive.athena import athenad
from selfdrive.athena.athenad import dispatcher
from selfdrive.athena.test_helpers import MockWebsocket, MockParams, MockApi, EchoSocket, with_http_server
from cereal import messaging

class TestAthenadMethods(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.SOCKET_PORT = 45454
    athenad.ROOT = tempfile.mkdtemp()
    athenad.Params = MockParams
    athenad.Api = MockApi
    athenad.LOCAL_PORT_WHITELIST = set([cls.SOCKET_PORT])

  def test_echo(self):
    assert dispatcher["echo"]("bob") == "bob"

  def test_getMessage(self):
    try:
      dispatcher["getMessage"]("controlsState")
    except TimeoutError:
      pass

    def send_thermal():
      messaging.context = messaging.Context()
      pub_sock = messaging.pub_sock("thermal")
      start = time.time()

      while time.time() - start < 1:
        msg = messaging.new_message()
        msg.init('thermal')
        pub_sock.send(msg.to_bytes())
        time.sleep(0.01)

    p = Process(target=send_thermal)
    p.start()
    try:
      thermal = dispatcher["getMessage"]("thermal")
      assert thermal['thermal']
    finally:
      p.terminate()

  def test_listDataDirectory(self):
    print(dispatcher["listDataDirectory"]())

  @with_http_server
  def test_do_upload(self):
    fn = os.path.join(athenad.ROOT, 'qlog.bz2')
    Path(fn).touch()

    try:
      item = athenad.UploadItem(path=fn, url="http://localhost:1238", headers={}, created_at=int(time.time()*1000), id='')
      try:
        athenad._do_upload(item)
      except requests.exceptions.ConnectionError:
        pass

      item = athenad.UploadItem(path=fn, url="http://localhost:44444/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='')
      resp = athenad._do_upload(item)
      self.assertEqual(resp.status_code, 201)
    finally:
      os.unlink(fn)

  @with_http_server
  def test_uploadFileToUrl(self):
    not_exists_resp = dispatcher["uploadFileToUrl"]("does_not_exist.bz2", "http://localhost:1238", {})
    self.assertEqual(not_exists_resp, 404)

    fn = os.path.join(athenad.ROOT, 'qlog.bz2')
    Path(fn).touch()

    try:
      resp = dispatcher["uploadFileToUrl"]("qlog.bz2", "http://localhost:44444/qlog.bz2", {})
      self.assertEqual(resp['enqueued'], 1)
      self.assertDictContainsSubset({"path": fn, "url": "http://localhost:44444/qlog.bz2", "headers": {}}, resp['item'])
      self.assertIsNotNone(resp['item'].get('id'))
      self.assertEqual(athenad.upload_queue.qsize(), 1)
    finally:
      athenad.upload_queue = queue.Queue()
      os.unlink(fn)

  @with_http_server
  def test_upload_handler(self):
    fn = os.path.join(athenad.ROOT, 'qlog.bz2')
    Path(fn).touch()
    item = athenad.UploadItem(path=fn, url="http://localhost:44444/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='')

    end_event = threading.Event()
    thread = threading.Thread(target=athenad.upload_handler, args=(end_event,))
    thread.start()

    athenad.upload_queue.put_nowait(item)
    try:
      now = time.time()
      while time.time() - now < 5:
        if athenad.upload_queue.qsize() == 0:
          break
      self.assertEqual(athenad.upload_queue.qsize(), 0)
    finally:
      end_event.set()
      athenad.upload_queue = queue.Queue()
      os.unlink(fn)

  def test_cancelUpload(self):
    item = athenad.UploadItem(path="qlog.bz2", url="http://localhost:44444/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='id')
    athenad.upload_queue.put_nowait(item)
    dispatcher["cancelUpload"](item.id)

    self.assertIn(item.id, athenad.cancelled_uploads)

    end_event = threading.Event()
    thread = threading.Thread(target=athenad.upload_handler, args=(end_event,))
    thread.start()
    try:
      now = time.time()
      while time.time() - now < 5:
        if athenad.upload_queue.qsize() == 0 and len(athenad.cancelled_uploads) == 0:
          break
      self.assertEqual(athenad.upload_queue.qsize(), 0)
      self.assertEqual(len(athenad.cancelled_uploads), 0)
    finally:
      end_event.set()
      athenad.upload_queue = queue.Queue()

  def test_listUploadQueue(self):
    item = athenad.UploadItem(path="qlog.bz2", url="http://localhost:44444/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='id')
    athenad.upload_queue.put_nowait(item)

    try:
      items = dispatcher["listUploadQueue"]()
      self.assertEqual(len(items), 1)
      self.assertDictEqual(items[0], item._asdict())
    finally:
      athenad.upload_queue = queue.Queue()

  @mock.patch('selfdrive.athena.athenad.create_connection')
  def test_startLocalProxy(self, mock_create_connection):
    end_event = threading.Event()

    ws_recv = queue.Queue()
    ws_send = queue.Queue()
    mock_ws = MockWebsocket(ws_recv, ws_send)
    mock_create_connection.return_value = mock_ws

    echo_socket = EchoSocket(self.SOCKET_PORT)
    socket_thread = threading.Thread(target=echo_socket.run)
    socket_thread.start()

    athenad.startLocalProxy(end_event, 'ws://localhost:1234', self.SOCKET_PORT)

    ws_recv.put_nowait(b'ping')
    try:
      recv = ws_send.get(timeout=5)
      assert recv == (b'ping', ABNF.OPCODE_BINARY), recv
    finally:
      # signal websocket close to athenad.ws_proxy_recv
      ws_recv.put_nowait(WebSocketConnectionClosedException())
      socket_thread.join()

  def test_getSshAuthorizedKeys(self):
    keys = dispatcher["getSshAuthorizedKeys"]()
    self.assertEqual(keys, MockParams().params["GithubSshKeys"].decode('utf-8'))

  def test_jsonrpc_handler(self):
    end_event = threading.Event()
    thread = threading.Thread(target=athenad.jsonrpc_handler, args=(end_event,))
    thread.daemon = True
    thread.start()
    athenad.payload_queue.put_nowait(json.dumps({"method": "echo", "params": ["hello"], "jsonrpc": "2.0", "id": 0}))
    try:
      resp = athenad.response_queue.get(timeout=3)
      self.assertDictEqual(resp.data, {'result': 'hello', 'id': 0, 'jsonrpc': '2.0'})
    finally:
      end_event.set()
      thread.join()

if __name__ == '__main__':
  unittest.main()
