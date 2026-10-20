"""Client for the AGNOS Wi-Fi service."""
import json
import socket

SOCKET = '/run/comma-wifi.sock'


def request(command, timeout=3, **args):
  with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
    sock.settimeout(timeout)
    sock.connect(SOCKET)
    sock.sendall(json.dumps({'command': command, **args}).encode() + b'\n')
    with sock.makefile('rb') as f:
      result = json.loads(f.readline(262144))
    if 'error' in result:
      raise ValueError(result['error'])
    return result

