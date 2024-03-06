#!/usr/bin/env python3
from functools import partial, wraps
import json
import multiprocessing
import os
import requests
import shutil
import time
import threading
import queue
import unittest
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from parameterized import parameterized

from unittest import mock
from websocket import ABNF
from websocket._exceptions import WebSocketConnectionClosedException

from cereal import messaging

from openpilot.common.params import Params
from openpilot.common.timeout import Timeout
from openpilot.selfdrive.athena import athenad
from openpilot.selfdrive.athena.athenad import MAX_RETRY_COUNT, dispatcher
from openpilot.selfdrive.athena.tests.helpers import HTTPRequestHandler, MockWebsocket, MockApi, EchoSocket
from openpilot.selfdrive.test.helpers import with_http_server
from openpilot.system.hardware.hw import Paths


def seed_athena_server(host, port):
  with Timeout(2, 'HTTP Server seeding failed'):
    while True:
      try:
        requests.put(f'http://{host}:{port}/qlog.bz2', data='', timeout=10)
        break
      except requests.exceptions.ConnectionError:
        time.sleep(0.1)


with_mock_athena = partial(with_http_server, handler=HTTPRequestHandler, setup=seed_athena_server)


def with_upload_handler(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    end_event = threading.Event()
    thread = threading.Thread(target=athenad.upload_handler, args=(end_event,))
    thread.start()
    try:
      return func(*args, **kwargs)
    finally:
      end_event.set()
      thread.join()
  return wrapper


class TestAthenadMethods(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.SOCKET_PORT = 45454
    athenad.Api = MockApi
    athenad.LOCAL_PORT_WHITELIST = {cls.SOCKET_PORT}

  def setUp(self):
    self.default_params = {
      "DongleId": "0000000000000000",
      "GithubSshKeys": b"ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC307aE+nuHzTAgaJhzSf5v7ZZQW9gaperjhCmyPyl4PzY7T1mDGenTlVTN7yoVFZ9UfO9oMQqo0n1OwDIiqbIFxqnhrHU0cYfj88rI85m5BEKlNu5RdaVTj1tcbaPpQc5kZEolaI1nDDjzV0lwS7jo5VYDHseiJHlik3HH1SgtdtsuamGR2T80q1SyW+5rHoMOJG73IH2553NnWuikKiuikGHUYBd00K1ilVAK2xSiMWJp55tQfZ0ecr9QjEsJ+J/efL4HqGNXhffxvypCXvbUYAFSddOwXUPo5BTKevpxMtH+2YrkpSjocWA04VnTYFiPG6U4ItKmbLOTFZtPzoez private", # noqa: E501
      "GithubUsername": b"commaci",
      "AthenadUploadQueue": '[]',
    }

    self.params = Params()
    for k, v in self.default_params.items():
      self.params.put(k, v)
    self.params.put_bool("GsmMetered", True)

    athenad.upload_queue = queue.Queue()
    athenad.cur_upload_items.clear()
    athenad.cancelled_uploads.clear()

    for i in os.listdir(Paths.log_root()):
      p = os.path.join(Paths.log_root(), i)
      if os.path.isdir(p):
        shutil.rmtree(p)
      else:
        os.unlink(p)

  # *** test helpers ***

  @staticmethod
  def _wait_for_upload():
    now = time.time()
    while time.time() - now < 5:
      if athenad.upload_queue.qsize() == 0:
        break

  @staticmethod
  def _create_file(file: str, parent: str = None, data: bytes = b'') -> str:
    fn = os.path.join(Paths.log_root() if parent is None else parent, file)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'wb') as f:
      f.write(data)
    return fn


  # *** test cases ***

  def test_echo(self):
    assert dispatcher["echo"]("bob") == "bob"

  def test_getMessage(self):
    with self.assertRaises(TimeoutError) as _:
      dispatcher["getMessage"]("controlsState")

    end_event = multiprocessing.Event()

    pub_sock = messaging.pub_sock("deviceState")

    def send_deviceState():
      while not end_event.is_set():
        msg = messaging.new_message('deviceState')
        pub_sock.send(msg.to_bytes())
        time.sleep(0.01)

    p = multiprocessing.Process(target=send_deviceState)
    p.start()
    time.sleep(0.1)
    try:
      deviceState = dispatcher["getMessage"]("deviceState")
      assert deviceState['deviceState']
    finally:
      end_event.set()
      p.join()

  def test_listDataDirectory(self):
    route = '2021-03-29--13-32-47'
    segments = [0, 1, 2, 3, 11]

    filenames = ['qlog', 'qcamera.ts', 'rlog', 'fcamera.hevc', 'ecamera.hevc', 'dcamera.hevc']
    files = [f'{route}--{s}/{f}' for s in segments for f in filenames]
    for file in files:
      self._create_file(file)

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

  def test_strip_bz2_extension(self):
    fn = self._create_file('qlog.bz2')
    if fn.endswith('.bz2'):
      self.assertEqual(athenad.strip_bz2_extension(fn), fn[:-4])

  @parameterized.expand([(True,), (False,)])
  @with_mock_athena
  def test_do_upload(self, compress, host):
    # random bytes to ensure rather large object post-compression
    fn = self._create_file('qlog', data=os.urandom(10000 * 1024))

    upload_fn = fn + ('.bz2' if compress else '')
    item = athenad.UploadItem(path=upload_fn, url="http://localhost:1238", headers={}, created_at=int(time.time()*1000), id='')
    with self.assertRaises(requests.exceptions.ConnectionError):
      athenad._do_upload(item)

    item = athenad.UploadItem(path=upload_fn, url=f"{host}/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='')
    resp = athenad._do_upload(item)
    self.assertEqual(resp.status_code, 201)

  @with_mock_athena
  def test_uploadFileToUrl(self, host):
    fn = self._create_file('qlog.bz2')

    resp = dispatcher["uploadFileToUrl"]("qlog.bz2", f"{host}/qlog.bz2", {})
    self.assertEqual(resp['enqueued'], 1)
    self.assertNotIn('failed', resp)
    self.assertLessEqual({"path": fn, "url": f"{host}/qlog.bz2", "headers": {}}.items(), resp['items'][0].items())
    self.assertIsNotNone(resp['items'][0].get('id'))
    self.assertEqual(athenad.upload_queue.qsize(), 1)

  @with_mock_athena
  def test_uploadFileToUrl_duplicate(self, host):
    self._create_file('qlog.bz2')

    url1 = f"{host}/qlog.bz2?sig=sig1"
    dispatcher["uploadFileToUrl"]("qlog.bz2", url1, {})

    # Upload same file again, but with different signature
    url2 = f"{host}/qlog.bz2?sig=sig2"
    resp = dispatcher["uploadFileToUrl"]("qlog.bz2", url2, {})
    self.assertEqual(resp, {'enqueued': 0, 'items': []})

  @with_mock_athena
  def test_uploadFileToUrl_does_not_exist(self, host):
    not_exists_resp = dispatcher["uploadFileToUrl"]("does_not_exist.bz2", "http://localhost:1238", {})
    self.assertEqual(not_exists_resp, {'enqueued': 0, 'items': [], 'failed': ['does_not_exist.bz2']})

  @with_mock_athena
  @with_upload_handler
  def test_upload_handler(self, host):
    fn = self._create_file('qlog.bz2')
    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='', allow_cellular=True)

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()
    time.sleep(0.1)

    # TODO: verify that upload actually succeeded
    # TODO: also check that end_event and metered network raises AbortTransferException
    self.assertEqual(athenad.upload_queue.qsize(), 0)

  @parameterized.expand([(500, True), (412, False)])
  @with_mock_athena
  @mock.patch('requests.put')
  @with_upload_handler
  def test_upload_handler_retry(self, status, retry, mock_put, host):
    mock_put.return_value.status_code = status
    fn = self._create_file('qlog.bz2')
    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='', allow_cellular=True)

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()
    time.sleep(0.1)

    self.assertEqual(athenad.upload_queue.qsize(), 1 if retry else 0)

    if retry:
      self.assertEqual(athenad.upload_queue.get().retry_count, 1)

  @with_upload_handler
  def test_upload_handler_timeout(self):
    """When an upload times out or fails to connect it should be placed back in the queue"""
    fn = self._create_file('qlog.bz2')
    item = athenad.UploadItem(path=fn, url="http://localhost:44444/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='', allow_cellular=True)
    item_no_retry = replace(item, retry_count=MAX_RETRY_COUNT)

    athenad.upload_queue.put_nowait(item_no_retry)
    self._wait_for_upload()
    time.sleep(0.1)

    # Check that upload with retry count exceeded is not put back
    self.assertEqual(athenad.upload_queue.qsize(), 0)

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()
    time.sleep(0.1)

    # Check that upload item was put back in the queue with incremented retry count
    self.assertEqual(athenad.upload_queue.qsize(), 1)
    self.assertEqual(athenad.upload_queue.get().retry_count, 1)

  @with_upload_handler
  def test_cancelUpload(self):
    item = athenad.UploadItem(path="qlog.bz2", url="http://localhost:44444/qlog.bz2", headers={},
                              created_at=int(time.time()*1000), id='id', allow_cellular=True)
    athenad.upload_queue.put_nowait(item)
    dispatcher["cancelUpload"](item.id)

    self.assertIn(item.id, athenad.cancelled_uploads)

    self._wait_for_upload()
    time.sleep(0.1)

    self.assertEqual(athenad.upload_queue.qsize(), 0)
    self.assertEqual(len(athenad.cancelled_uploads), 0)

  @with_upload_handler
  def test_cancelExpiry(self):
    t_future = datetime.now() - timedelta(days=40)
    ts = int(t_future.strftime("%s")) * 1000

    # Item that would time out if actually uploaded
    fn = self._create_file('qlog.bz2')
    item = athenad.UploadItem(path=fn, url="http://localhost:44444/qlog.bz2", headers={}, created_at=ts, id='', allow_cellular=True)

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()
    time.sleep(0.1)

    self.assertEqual(athenad.upload_queue.qsize(), 0)

  def test_listUploadQueueEmpty(self):
    items = dispatcher["listUploadQueue"]()
    self.assertEqual(len(items), 0)

  @with_http_server
  @with_upload_handler
  def test_listUploadQueueCurrent(self, host: str):
    fn = self._create_file('qlog.bz2')
    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.bz2", headers={}, created_at=int(time.time()*1000), id='', allow_cellular=True)

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()

    items = dispatcher["listUploadQueue"]()
    self.assertEqual(len(items), 1)
    self.assertTrue(items[0]['current'])

  def test_listUploadQueue(self):
    item = athenad.UploadItem(path="qlog.bz2", url="http://localhost:44444/qlog.bz2", headers={},
                              created_at=int(time.time()*1000), id='id', allow_cellular=True)
    athenad.upload_queue.put_nowait(item)

    items = dispatcher["listUploadQueue"]()
    self.assertEqual(len(items), 1)
    self.assertDictEqual(items[0], asdict(item))
    self.assertFalse(items[0]['current'])

    athenad.cancelled_uploads.add(item.id)
    items = dispatcher["listUploadQueue"]()
    self.assertEqual(len(items), 0)

  def test_upload_queue_persistence(self):
    item1 = athenad.UploadItem(path="_", url="_", headers={}, created_at=int(time.time()), id='id1')
    item2 = athenad.UploadItem(path="_", url="_", headers={}, created_at=int(time.time()), id='id2')

    athenad.upload_queue.put_nowait(item1)
    athenad.upload_queue.put_nowait(item2)

    # Ensure cancelled items are not persisted
    athenad.cancelled_uploads.add(item2.id)

    # serialize item
    athenad.UploadQueueCache.cache(athenad.upload_queue)

    # deserialize item
    athenad.upload_queue.queue.clear()
    athenad.UploadQueueCache.initialize(athenad.upload_queue)

    self.assertEqual(athenad.upload_queue.qsize(), 1)
    self.assertDictEqual(asdict(athenad.upload_queue.queue[-1]), asdict(item1))

  @mock.patch('openpilot.selfdrive.athena.athenad.create_connection')
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
    self.assertEqual(keys, self.default_params["GithubSshKeys"].decode('utf-8'))

  def test_getGithubUsername(self):
    keys = dispatcher["getGithubUsername"]()
    self.assertEqual(keys, self.default_params["GithubUsername"].decode('utf-8'))

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
      file = f'swaglog.{i:010}'
      self._create_file(file, Paths.swaglog_root())
      fl.append(file)

    # ensure the list is all logs except most recent
    sl = athenad.get_logs_to_send_sorted()
    self.assertListEqual(sl, fl[:-1])


if __name__ == '__main__':
  unittest.main()
