import http.server
import requests
import socket
import time
from functools import wraps
from multiprocessing import Process

class EchoSocket():
  def __init__(self, port):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind(('127.0.0.1', port))
    self.socket.listen(1)

  def run(self):
    conn, client_address = self.socket.accept()
    conn.settimeout(5.0)

    try:
      while True:
        data = conn.recv(4096)
        if data:
          print(f'EchoSocket got {data}')
          conn.sendall(data)
        else:
          break
    finally:
      conn.shutdown(0)
      conn.close()
      self.socket.shutdown(0)
      self.socket.close()

class MockApi():
  def __init__(self, dongle_id):
    pass

  def get_token(self):
    return "fake-token"

class MockParams():
  def __init__(self):
    self.params = {
      "DongleId": b"0000000000000000",
      "GithubSshKeys": b"ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC307aE+nuHzTAgaJhzSf5v7ZZQW9gaperjhCmyPyl4PzY7T1mDGenTlVTN7yoVFZ9UfO9oMQqo0n1OwDIiqbIFxqnhrHU0cYfj88rI85m5BEKlNu5RdaVTj1tcbaPpQc5kZEolaI1nDDjzV0lwS7jo5VYDHseiJHlik3HH1SgtdtsuamGR2T80q1SyW+5rHoMOJG73IH2553NnWuikKiuikGHUYBd00K1ilVAK2xSiMWJp55tQfZ0ecr9QjEsJ+J/efL4HqGNXhffxvypCXvbUYAFSddOwXUPo5BTKevpxMtH+2YrkpSjocWA04VnTYFiPG6U4ItKmbLOTFZtPzoez private"
    }

  def get(self, k, encoding=None):
    ret = self.params.get(k)
    if ret is not None and encoding is not None:
      ret = ret.decode(encoding)
    return ret

class MockWebsocket():
  def __init__(self, recv_queue, send_queue):
    self.recv_queue = recv_queue
    self.send_queue = send_queue

  def recv(self):
    data = self.recv_queue.get()
    if isinstance(data, Exception):
      raise data
    return data

  def send(self, data, opcode):
    self.send_queue.put_nowait((data, opcode))

class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
  def do_PUT(self):
    length = int(self.headers['Content-Length'])
    self.rfile.read(length)
    self.send_response(201, "Created")
    self.end_headers()

def with_http_server(func):
  @wraps(func)
  def inner(*args, **kwargs):
    p = Process(target=http.server.test,
                kwargs={
                  'HandlerClass': HTTPRequestHandler,
                  'port': 44444,
                  'bind': '127.0.0.1'})
    p.start()
    now = time.time()
    while 1:
      if time.time() - now > 5:
        raise Exception('HTTP Server did not start')
      try:
        requests.put('http://localhost:44444/qlog.bz2', data='')
        break
      except requests.exceptions.ConnectionError:
        time.sleep(0.1)

    try:
      return func(*args, **kwargs)
    finally:
      p.terminate()

  return inner
