import pytest
from functools import wraps
import json
import multiprocessing
import os
import requests
import shutil
import time
import threading
import queue
from dataclasses import asdict, replace
from datetime import datetime, timedelta

from websocket import ABNF
from websocket._exceptions import WebSocketConnectionClosedException

from cereal import messaging

from openpilot.common.params import Params
from openpilot.common.timeout import Timeout
from openpilot.system.athena import athenad
from openpilot.system.athena.athenad import MAX_RETRY_COUNT, UPLOAD_SESS, dispatcher
from openpilot.system.athena.tests.helpers import HTTPRequestHandler, MockWebsocket, MockApi, EchoSocket
from openpilot.selfdrive.test.helpers import http_server_context
from openpilot.system.hardware.hw import Paths


def seed_athena_server(host, port):
  with Timeout(2, 'HTTP Server seeding failed'):
    while True:
      try:
        UPLOAD_SESS.put(f'http://{host}:{port}/qlog.zst', data='', timeout=10)
        break
      except requests.exceptions.ConnectionError:
        time.sleep(0.1)

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

@pytest.fixture
def mock_create_connection(mocker):
    return mocker.patch('openpilot.system.athena.athenad.create_connection')

@pytest.fixture
def host():
  with http_server_context(handler=HTTPRequestHandler, setup=seed_athena_server) as (host, port):
    yield f"http://{host}:{port}"

class TestAthenadMethods:
  @classmethod
  def setup_class(cls):
    cls.SOCKET_PORT = 45454
    athenad.Api = MockApi
    athenad.LOCAL_PORT_WHITELIST = {cls.SOCKET_PORT}

  def setup_method(self):
    self.default_params = {
      "DongleId": "0000000000000000",
      "GithubSshKeys": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC307aE+nuHzTAgaJhzSf5v7ZZQW9gaperjhCmyPyl4PzY7T1mDGenTlVTN7yoVFZ9UfO9oMQqo0n1OwDIiqbIFxqnhrHU0cYfj88rI85m5BEKlNu5RdaVTj1tcbaPpQc5kZEolaI1nDDjzV0lwS7jo5VYDHseiJHlik3HH1SgtdtsuamGR2T80q1SyW+5rHoMOJG73IH2553NnWuikKiuikGHUYBd00K1ilVAK2xSiMWJp55tQfZ0ecr9QjEsJ+J/efL4HqGNXhffxvypCXvbUYAFSddOwXUPo5BTKevpxMtH+2YrkpSjocWA04VnTYFiPG6U4ItKmbLOTFZtPzoez private", # noqa: E501
      "GithubUsername": "commaci",
      "AthenadUploadQueue": [],
    }

    self.params = Params()
    for k, v in self.default_params.items():
      self.params.put(k, v)
    self.params.put_bool("GsmMetered", True)

    athenad.upload_queue = queue.PriorityQueue()
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
    now = time.monotonic()
    while time.monotonic() - now < 5:
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

  def test_get_message(self):
    with pytest.raises(TimeoutError) as _:
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

  def test_list_data_directory(self):
    route = '2021-03-29--13-32-47'
    segments = [0, 1, 2, 3, 11]

    filenames = ['qlog.zst', 'qcamera.ts', 'rlog.zst', 'fcamera.hevc', 'ecamera.hevc', 'dcamera.hevc']
    files = [f'{route}--{s}/{f}' for s in segments for f in filenames]
    for file in files:
      self._create_file(file)

    resp = dispatcher["listDataDirectory"]()
    assert resp, 'list empty!'
    assert len(resp) == len(files)

    resp = dispatcher["listDataDirectory"](f'{route}--123')
    assert len(resp) == 0

    prefix = f'{route}'
    expected = list(filter(lambda f: f.startswith(prefix), files))
    resp = dispatcher["listDataDirectory"](prefix)
    assert resp, 'list empty!'
    assert len(resp) == len(expected)

    prefix = f'{route}--1'
    expected = list(filter(lambda f: f.startswith(prefix), files))
    resp = dispatcher["listDataDirectory"](prefix)
    assert resp, 'list empty!'
    assert len(resp) == len(expected)

    prefix = f'{route}--1/'
    expected = list(filter(lambda f: f.startswith(prefix), files))
    resp = dispatcher["listDataDirectory"](prefix)
    assert resp, 'list empty!'
    assert len(resp) == len(expected)

    prefix = f'{route}--1/q'
    expected = list(filter(lambda f: f.startswith(prefix), files))
    resp = dispatcher["listDataDirectory"](prefix)
    assert resp, 'list empty!'
    assert len(resp) == len(expected)

  def test_strip_extension(self):
    # any requested log file with an invalid extension won't return as existing
    fn = self._create_file('qlog.bz2')
    if fn.endswith('.bz2'):
      assert athenad.strip_zst_extension(fn) == fn

    fn = self._create_file('qlog.zst')
    if fn.endswith('.zst'):
      assert athenad.strip_zst_extension(fn) == fn[:-4]

  @pytest.mark.parametrize("compress", [True, False])
  def test_do_upload(self, host, compress):
    # random bytes to ensure rather large object post-compression
    fn = self._create_file('qlog', data=os.urandom(10000 * 1024))

    upload_fn = fn + ('.zst' if compress else '')
    item = athenad.UploadItem(path=upload_fn, url="http://localhost:1238", headers={}, created_at=int(time.time()*1000), id='')  # noqa: TID251
    with pytest.raises(requests.exceptions.ConnectionError):
      athenad._do_upload(item)

    item = athenad.UploadItem(path=upload_fn, url=f"{host}/qlog.zst", headers={}, created_at=int(time.time()*1000), id='')  # noqa: TID251
    resp = athenad._do_upload(item)
    assert resp.status_code == 201

  def test_upload_file_to_url(self, host):
    fn = self._create_file('qlog.zst')

    resp = dispatcher["uploadFileToUrl"]("qlog.zst", f"{host}/qlog.zst", {})
    assert resp['enqueued'] == 1
    assert 'failed' not in resp
    assert {"path": fn, "url": f"{host}/qlog.zst", "headers": {}}.items() <= resp['items'][0].items()
    assert resp['items'][0].get('id') is not None
    assert athenad.upload_queue.qsize() == 1

  def test_upload_file_to_url_duplicate(self, host):
    self._create_file('qlog.zst')

    url1 = f"{host}/qlog.zst?sig=sig1"
    dispatcher["uploadFileToUrl"]("qlog.zst", url1, {})

    # Upload same file again, but with different signature
    url2 = f"{host}/qlog.zst?sig=sig2"
    resp = dispatcher["uploadFileToUrl"]("qlog.zst", url2, {})
    assert resp == {'enqueued': 0, 'items': []}

  def test_upload_file_to_url_does_not_exist(self, host):
    not_exists_resp = dispatcher["uploadFileToUrl"]("does_not_exist.zst", "http://localhost:1238", {})
    assert not_exists_resp == {'enqueued': 0, 'items': [], 'failed': ['does_not_exist.zst']}

  @with_upload_handler
  def test_upload_handler(self, host):
    fn = self._create_file('qlog.zst')
    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.zst", headers={}, created_at=int(time.time()*1000), id='', allow_cellular=True)  # noqa: TID251

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()
    time.sleep(0.1)

    # TODO: verify that upload actually succeeded
    # TODO: also check that end_event and metered network raises AbortTransferException
    assert athenad.upload_queue.qsize() == 0

  @pytest.mark.parametrize("status,retry", [(500,True), (412,False)])
  @with_upload_handler
  def test_upload_handler_retry(self, mocker, host, status, retry):
    mock_put = mocker.patch('openpilot.system.athena.athenad.UPLOAD_SESS.put')
    mock_put.return_value.__enter__.return_value.status_code = status
    fn = self._create_file('qlog.zst')
    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.zst", headers={}, created_at=int(time.time()*1000), id='', allow_cellular=True)  # noqa: TID251

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()
    time.sleep(0.1)

    assert athenad.upload_queue.qsize() == (1 if retry else 0)

    if retry:
      assert athenad.upload_queue.get().retry_count == 1

  @with_upload_handler
  def test_upload_handler_timeout(self):
    """When an upload times out or fails to connect it should be placed back in the queue"""
    fn = self._create_file('qlog.zst')
    item = athenad.UploadItem(path=fn, url="http://localhost:44444/qlog.zst", headers={}, created_at=int(time.time()*1000), id='', allow_cellular=True)  # noqa: TID251
    item_no_retry = replace(item, retry_count=MAX_RETRY_COUNT)

    athenad.upload_queue.put_nowait(item_no_retry)
    self._wait_for_upload()
    time.sleep(0.1)

    # Check that upload with retry count exceeded is not put back
    assert athenad.upload_queue.qsize() == 0

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()
    time.sleep(0.1)

    # Check that upload item was put back in the queue with incremented retry count
    assert athenad.upload_queue.qsize() == 1
    assert athenad.upload_queue.get().retry_count == 1

  @with_upload_handler
  def test_cancel_upload(self):
    item = athenad.UploadItem(path="qlog.zst", url="http://localhost:44444/qlog.zst", headers={},
                              created_at=int(time.time()*1000), id='id', allow_cellular=True)  # noqa: TID251
    athenad.upload_queue.put_nowait(item)
    dispatcher["cancelUpload"](item.id)

    assert item.id in athenad.cancelled_uploads

    self._wait_for_upload()
    time.sleep(0.1)

    assert athenad.upload_queue.qsize() == 0
    assert len(athenad.cancelled_uploads) == 0

  @with_upload_handler
  def test_cancel_expiry(self):
    t_future = datetime.now() - timedelta(days=40)
    ts = int(t_future.strftime("%s")) * 1000

    # Item that would time out if actually uploaded
    fn = self._create_file('qlog.zst')
    item = athenad.UploadItem(path=fn, url="http://localhost:44444/qlog.zst", headers={}, created_at=ts, id='', allow_cellular=True)

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()
    time.sleep(0.1)

    assert athenad.upload_queue.qsize() == 0

  def test_list_upload_queue_empty(self):
    items = dispatcher["listUploadQueue"]()
    assert len(items) == 0

  @with_upload_handler
  def test_list_upload_queue_current(self, host: str):
    fn = self._create_file('qlog.zst')
    item = athenad.UploadItem(path=fn, url=f"{host}/qlog.zst", headers={}, created_at=int(time.time()*1000), id='', allow_cellular=True)  # noqa: TID251

    athenad.upload_queue.put_nowait(item)
    self._wait_for_upload()

    items = dispatcher["listUploadQueue"]()
    assert len(items) == 1
    assert items[0]['current']

  def test_list_upload_queue_priority(self):
    priorities = (25, 50, 99, 75, 0)

    for i in priorities:
      fn = f'qlog_{i}.zst'
      fp = self._create_file(fn)
      item = athenad.UploadItem(
        path=fp,
        url=f"http://localhost:44444/{fn}",
        headers={},
        created_at=int(time.time()*1000),  # noqa: TID251
        id='',
        allow_cellular=True,
        priority=i
      )
      athenad.upload_queue.put_nowait(item)

    for i in sorted(priorities):
      assert athenad.upload_queue.get_nowait().priority == i

  def test_list_upload_queue(self):
    item = athenad.UploadItem(path="qlog.zst", url="http://localhost:44444/qlog.zst", headers={},
                              created_at=int(time.time()*1000), id='id', allow_cellular=True)  # noqa: TID251
    athenad.upload_queue.put_nowait(item)

    items = dispatcher["listUploadQueue"]()
    assert len(items) == 1
    assert items[0] == asdict(item)
    assert not items[0]['current']

    athenad.cancelled_uploads.add(item.id)
    items = dispatcher["listUploadQueue"]()
    assert len(items) == 0

  def test_upload_queue_persistence(self):
    item1 = athenad.UploadItem(path="_", url="_", headers={}, created_at=int(time.time()), id='id1')  # noqa: TID251
    item2 = athenad.UploadItem(path="_", url="_", headers={}, created_at=int(time.time()), id='id2')  # noqa: TID251

    athenad.upload_queue.put_nowait(item1)
    athenad.upload_queue.put_nowait(item2)

    # Ensure canceled items are not persisted
    athenad.cancelled_uploads.add(item2.id)

    # serialize item
    athenad.UploadQueueCache.cache(athenad.upload_queue)

    # deserialize item
    athenad.upload_queue.queue.clear()
    athenad.UploadQueueCache.initialize(athenad.upload_queue)

    assert athenad.upload_queue.qsize() == 1
    assert asdict(athenad.upload_queue.queue[-1]) == asdict(item1)

  def test_start_local_proxy(self, mock_create_connection):
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

  def test_get_ssh_authorized_keys(self):
    keys = dispatcher["getSshAuthorizedKeys"]()
    assert keys == self.default_params["GithubSshKeys"]

  def test_get_github_username(self):
    keys = dispatcher["getGithubUsername"]()
    assert keys == self.default_params["GithubUsername"]

  def test_get_version(self):
    resp = dispatcher["getVersion"]()
    keys = ["version", "remote", "branch", "commit"]
    assert list(resp.keys()) == keys
    for k in keys:
      assert isinstance(resp[k], str), f"{k} is not a string"
      assert len(resp[k]) > 0, f"{k} has no value"

  def test_jsonrpc_handler(self):
    end_event = threading.Event()
    thread = threading.Thread(target=athenad.jsonrpc_handler, args=(end_event,))
    thread.daemon = True
    thread.start()
    try:
      # with params
      athenad.recv_queue.put_nowait(json.dumps({"method": "echo", "params": ["hello"], "jsonrpc": "2.0", "id": 0}))
      resp = athenad.send_queue.get(timeout=3)
      assert json.loads(resp) == {'result': 'hello', 'id': 0, 'jsonrpc': '2.0'}
      # without params
      athenad.recv_queue.put_nowait(json.dumps({"method": "getNetworkType", "jsonrpc": "2.0", "id": 0}))
      resp = athenad.send_queue.get(timeout=3)
      assert json.loads(resp) == {'result': 1, 'id': 0, 'jsonrpc': '2.0'}
      # log forwarding
      athenad.recv_queue.put_nowait(json.dumps({'result': {'success': 1}, 'id': 0, 'jsonrpc': '2.0'}))
      resp = athenad.log_recv_queue.get(timeout=3)
      assert json.loads(resp) == {'result': {'success': 1}, 'id': 0, 'jsonrpc': '2.0'}
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
    assert sl == fl[:-1]
