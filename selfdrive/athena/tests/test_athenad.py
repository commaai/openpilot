#!/usr/bin/env python3
import json
import os
import requests
import shutil
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

from selfdrive import swaglog
from selfdrive.athena import athenad
from selfdrive.athena.athenad import MAX_RETRY_COUNT, dispatcher
from selfdrive.athena.tests.helpers import MockWebsocket, MockParams, MockApi, EchoSocket, with_http_server
from cereal import messaging

class TestAthenadMethods(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.SOCKET_PORT = 45454
    athenad.params = MockParams()
    athenad.ROOT = tempfile.mkdtemp()
    athenad.SWAGLOG_DIR = swaglog.SWAGLOG_DIR = tempfile.mkdtemp()
    athenad.Api = MockApi
    athenad.LOCAL_PORT_WHITELIST = set([cls.SOCKET_PORT])

  def setUp(self):
    MockParams.restore_defaults()
    athenad.upload_queue = queue.Queue()
    athenad.cur_upload_items.clear()
    athenad.cancelled_uploads.clear()

    for i in os.listdir(athenad.ROOT):
      p = os.path.join(athenad.ROOT, i)
      if os.path.isdir(p):
        shutil.rmtree(p)
      else:
        os.unlink(p)

  def wait_for_upload(self):
    now = time.time()
    while time.time() - now < 5:
      if athenad.upload_queue.qsize() == 0:
        break

  def test_echo(self):
    assert dispatcher["echo"]("bob") == "bob"

  def test_getMessage(self):
    with self.assertRaises(TimeoutError) as _:
      dispatcher["getMessage"]("controlsState")

    def send_deviceState():
      messaging.context = messaging.Context()
      pub_sock = messaging.pub_sock("deviceState")
      start = time.time()

      while time.time() - start < 1:
        msg = messaging.new_message('deviceState')
        pub_sock.send(msg.to_bytes())
        time.sleep(0.01)

    p = Process(target=send_deviceState)
    p.start()
    time.sleep(0.1)
    try:
      deviceState = dispatcher["getMessage"]("deviceState")
      assert deviceState['deviceState']
    finally:
      p.terminate()

  def test_listDataDirectory(self):
    route = '2021-03-29--13-32-47'
    segments = [0, 1, 2, 3, 11]
    filenames = ['qlog.bz2', 'qcamera.ts', 'rlog.bz2', 'fcamera.hevc', 'ecamera.hevc', 'dcamera.hevc']
    files = [f'{route}--{s}/{f}' for s in segments for f in filenames]
    for file in files:
      fn = os.path.join(athenad.ROOT, file)
      os.makedirs(os.path.dirname(fn), exist_ok=True)
      Path(fn).touch()

    resp = dispatcher["listDataDirectory"]()
    self.assertTrue(resp, 'list empty!')
    self.assertCountEqual(resp, files)

    resp = dispatcher["listDataDirectory"](f'{route}--123')
    self.assertCountEqual(resp, [])

    prefix = f'{route}'
    expected = filter(lambda f: f.startswith(prefix), files)
    resp = dispatcher["listDataDirectory"](prefix)
    self.assertTrue(resp, 'list empty!')
    self.assertCountEqual(resp, expected)

    prefix = f'{route}--1'
    expected = filter(lambda f: f.startswith(prefix), files)
    resp = dispatcher["listDataDirectory"](prefix)
    self.assertTrue(resp, 'list empty!')
    self.assertCountEqual(resp, expected)

    prefix = f'{route}--1/'
    expected = filter(lambda f: f.startswith(prefix), files)
    resp = dispatcher["listDataDirectory"](prefix)
    self.assertTrue(resp, 'list empty!')
    self.assertCountEqual(resp, expected)

    prefix = f'{route}--1/q'
    expected = filter(lambda f: f.startswith(prefix), files)
    resp = dispatcher["listDataDirectory"](prefix)
    self.assertTrue(resp, 'list empty!')
    self.assertCountEqual(resp, expected)

  @with_http_server
  def test_do_upload(self, host):
    fn = os.path.join(athenad.ROOT, 'qlog.bz2')
    Path(fn).touch()

    item = athenad.UploadItem(path=fn, url="http://localhost:1238", headers={}, created_at=int(time.time()*1000), id='')
    with self.assertRaises(requests.exceptions.ConnectionError):
      athenad._do_upload(item)

    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='')
    resp = athenad._do_upload(item)
    self.assertEqual(resp.status_code, 201)

  @with_http_server
  def test_uploadFileToUrl(self, host):
    not_exists_resp = dispatcher["uploadFileToUrl"]("does_not_exist.bz2", "http://localhost:1238", {})
    self.assertEqual(not_exists_resp, 404)

    fn = os.path.join(athenad.ROOT, 'qlog.bz2')
    Path(fn).touch()

    resp = dispatcher["uploadFileToUrl"]("qlog.bz2", f"{host}/qlog.bz2", {})
    self.assertEqual(resp['enqueued'], 1)
    self.assertDictContainsSubset({"path": fn, "url": f"{host}/qlog.bz2", "headers": {}}, resp['item'])
    self.assertIsNotNone(resp['item'].get('id'))
    self.assertEqual(athenad.upload_queue.qsize(), 1)

  @with_http_server
  def test_upload_handler(self, host):
    fn = os.path.join(athenad.ROOT, 'qlog.bz2')
    Path(fn).touch()
    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='')

    end_event = threading.Event()
    thread = threading.Thread(target=athenad.upload_handler, args=(end_event,))
    thread.start()

    athenad.upload_queue.put_nowait(item)
    try:
      self.wait_for_upload()
      time.sleep(0.1)

      # TODO: verify that upload actually succeeded
      self.assertEqual(athenad.upload_queue.qsize(), 0)
    finally:
      end_event.set()

  def test_upload_handler_timeout(self):
    """When an upload times out or fails to connect it should be placed back in the queue"""
    fn = os.path.join(athenad.ROOT, 'qlog.bz2')
    Path(fn).touch()
    item = athenad.UploadItem(path=fn, url="http://localhost:44444/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='')
    item_no_retry = item._replace(retry_count=MAX_RETRY_COUNT)

    end_event = threading.Event()
    thread = threading.Thread(target=athenad.upload_handler, args=(end_event,))
    thread.start()

    try:
      athenad.upload_queue.put_nowait(item_no_retry)
      self.wait_for_upload()
      time.sleep(0.1)

      # Check that upload with retry count exceeded is not put back
      self.assertEqual(athenad.upload_queue.qsize(), 0)

      athenad.upload_queue.put_nowait(item)
      self.wait_for_upload()
      time.sleep(0.1)

      # Check that upload item was put back in the queue with incremented retry count
      self.assertEqual(athenad.upload_queue.qsize(), 1)
      self.assertEqual(athenad.upload_queue.get().retry_count, 1)

    finally:
      end_event.set()

  def test_cancelUpload(self):
    item = athenad.UploadItem(path="qlog.bz2", url="http://localhost:44444/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='id')
    athenad.upload_queue.put_nowait(item)
    dispatcher["cancelUpload"](item.id)

    self.assertIn(item.id, athenad.cancelled_uploads)

    end_event = threading.Event()
    thread = threading.Thread(target=athenad.upload_handler, args=(end_event,))
    thread.start()
    try:
      self.wait_for_upload()
      time.sleep(0.1)

      self.assertEqual(athenad.upload_queue.qsize(), 0)
      self.assertEqual(len(athenad.cancelled_uploads), 0)
    finally:
      end_event.set()

  def test_listUploadQueueEmpty(self):
    items = dispatcher["listUploadQueue"]()
    self.assertEqual(len(items), 0)

  @with_http_server
  def test_listUploadQueueCurrent(self, host):
    fn = os.path.join(athenad.ROOT, 'qlog.bz2')
    Path(fn).touch()
    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='')

    end_event = threading.Event()
    thread = threading.Thread(target=athenad.upload_handler, args=(end_event,))
    thread.start()

    try:
      athenad.upload_queue.put_nowait(item)
      self.wait_for_upload()

      items = dispatcher["listUploadQueue"]()
      self.assertEqual(len(items), 1)
      self.assertTrue(items[0]['current'])

    finally:
      end_event.set()

  def test_listUploadQueue(self):
    item = athenad.UploadItem(path="qlog.bz2", url="http://localhost:44444/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='id')
    athenad.upload_queue.put_nowait(item)

    items = dispatcher["listUploadQueue"]()
    self.assertEqual(len(items), 1)
    self.assertDictEqual(items[0], item._asdict())
    self.assertFalse(items[0]['current'])

    athenad.cancelled_uploads.add(item.id)
    items = dispatcher["listUploadQueue"]()
    self.assertEqual(len(items), 0)
  
  def test_upload_queue_persistence(self):
    item = athenad.UploadItem(path="_", url="_", headers={}, created_at=int(time.time()), id='id')

    # serialize item
    athenad.upload_queue.put_nowait(item)
    athenad.cache_upload_queue(athenad.upload_queue)

    # deserialize item 
    athenad.upload_queue.queue.clear() 
    athenad.init_upload_queue(athenad.upload_queue) 

    self.assertEqual(athenad.upload_queue.qsize(), 1)
    self.assertDictEqual(athenad.upload_queue.queue[-1]._asdict(), item._asdict())

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

  def test_getVersion(self):
    resp = dispatcher["getVersion"]()
    keys = ["version", "remote", "branch", "commit"]
    self.assertEqual(list(resp.keys()), keys)
    for k in keys:
      self.assertIsInstance(resp[k], str, f"{k} is not a string")
      self.assertTrue(len(resp[k]) > 0, f"{k} has no value")

  def test_jsonrpc_handler(self):
    end_event = threading.Event()
    thread = threading.Thread(target=athenad.jsonrpc_handler, args=(end_event,))
    thread.daemon = True
    thread.start()
    try:
      # with params
      athenad.recv_queue.put_nowait(json.dumps({"method": "echo", "params": ["hello"], "jsonrpc": "2.0", "id": 0}))
      resp = athenad.send_queue.get(timeout=3)
      self.assertDictEqual(json.loads(resp), {'result': 'hello', 'id': 0, 'jsonrpc': '2.0'})
      # without params
      athenad.recv_queue.put_nowait(json.dumps({"method": "getNetworkType", "jsonrpc": "2.0", "id": 0}))
      resp = athenad.send_queue.get(timeout=3)
      self.assertDictEqual(json.loads(resp), {'result': 1, 'id': 0, 'jsonrpc': '2.0'})
      # log forwarding
      athenad.recv_queue.put_nowait(json.dumps({'result': {'success': 1}, 'id': 0, 'jsonrpc': '2.0'}))
      resp = athenad.log_recv_queue.get(timeout=3)
      self.assertDictEqual(json.loads(resp), {'result': {'success': 1}, 'id': 0, 'jsonrpc': '2.0'})
    finally:
      end_event.set()
      thread.join()

  def test_get_logs_to_send_sorted(self):
    fl = list()
    for i in range(10):
      fn = os.path.join(swaglog.SWAGLOG_DIR, f'swaglog.{i:010}')
      Path(fn).touch()
      fl.append(os.path.basename(fn))

    # ensure the list is all logs except most recent
    sl = athenad.get_logs_to_send_sorted()
    self.assertListEqual(sl, fl[:-1])

if __name__ == '__main__':
  unittest.main()