import http.server
import socket


class MockResponse:
  def __init__(self, json, status_code):
    self.json = json
    self.text = json
    self.status_code = status_code


class EchoSocket:
  def __init__(self, port):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind(('127.0.0.1', port))
    self.socket.listen(1)

  def run(self):
    conn, _ = self.socket.accept()
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


class MockApi:
  def __init__(self, dongle_id):
    pass

  def get_token(self):
    return "fake-token"


class MockWebsocket:
  sock = socket.socket()

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

  def close(self):
    pass


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
  def do_PUT(self):
    length = int(self.headers['Content-Length'])
    self.rfile.read(length)
    self.send_response(201, "Created")
    self.end_headers()
