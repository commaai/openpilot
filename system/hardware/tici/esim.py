#!/usr/bin/env python3
import os
import json
import math
import time
import binascii
import requests
import serial
import subprocess
from typing import Optional

from openpilot.system.swaglog import cloudlog

class ATCommandFailed(Exception):
  pass

def at(cmd: str) -> Optional[str]:
  for _ in range(1):
    try:
      return subprocess.check_output(f"mmcli -m any --timeout 30 --command='{cmd}'", shell=True, encoding='utf8')
    except subprocess.CalledProcessError:
      cloudlog.exception("at_command_failed")
      time.sleep(1.0)
  raise ATCommandFailed(f"failed to execute {cmd=}")

def post(url, payload):
  print("POST to", url)
  r = requests.post(
    url,
    data=payload,
    verify=False,
    headers={
      "Content-Type": "application/json",
      "X-Admin-Protocol": "gsma/rsp/v2.2.0",
      "charset": "utf-8",
      "User-Agent": "gsma-rsp-lpad",
    },
  )
  print("resp", r)
  print("resp text", repr(r.text))
  r.raise_for_status()

  #ret = f"HTTP/{r.raw.version.decode('utf-8')} {r.status_code}"
  #ret += ''.join(f"{k}: {v}" for k, v in r.headers.items())
  ret = f"HTTP/1.1 {r.status_code}"
  ret += ''.join(f"{k}: {v}" for k, v in r.headers.items() if k != 'Connection')
  return ret.encode() + r.content

def get_unsolicited_response():
  # why doesn't modem manager directly return this?
  time.sleep(0.1)
  out = subprocess.check_output("journalctl -o cat -u ModemManager | grep QESIM", shell=True, encoding='utf8')
  return out.split('+QESIM:')[-1].split('<CR><LF><CR><LF>OK<CR><LF>')[0]

class LPA:
  def __init__(self):
    self.dev = serial.Serial('/dev/ttyUSB2', baudrate=57600, timeout=1, bytesize=8)
    self.dev.reset_input_buffer()
    self.dev.reset_output_buffer()
    assert "OK" in self.at("AT")

  def at(self, cmd):
    print(f"==> sending {cmd}")
    self.dev.write(cmd.encode() + b'\r\n')
    r = self.dev.read(8192).strip()
    if b"OK" not in r and b"ERROR" not in r:
      time.sleep(7)
      r += self.dev.read(8192).strip()
    print(f"<== recv {repr(r)}")
    return r.decode()

  def download_ota(self, qr):
    return at(f'AT+QESIM="ota","{qr}"')

  def download(self, qr):
    # TODO: double check this against the QR code
    smdp = "smdp.io"
    out = self.at(f'AT+QESIM="download","{qr}"')
    for n in range(3):
      print("\n\n\n!!!!!!!!!!!!!!!!!!", n, "\n")
      line = out.split("+QESIM: ")[1].split("\r\n\r\nOK")[0]
      print("line", repr(line))

      parts = [x.strip().strip('"') for x in line.split(',', maxsplit=4)]
      print(repr(parts))
      trans, ret, url, payloadlen, payload = parts
      assert trans == "trans" and ret == "0"
      assert len(payload) == int(payloadlen)

      r = post(f"https://{smdp}/{url}", payload)
      to_send = binascii.hexlify(r).decode()
      print("\ngoing to module", repr(r), "\n", to_send, "\n")

      max_trans_len = 1400
      n_transfers = math.ceil(len(to_send) / max_trans_len)
      for i in range(n_transfers):
        print("****** doing ", i)
        state = 1 if i < n_transfers-1 else 0
        data = to_send[i * max_trans_len: (i+1)*max_trans_len]

        out = self.at(f'AT+QESIM="trans",{len(to_send)},{state},{i},{len(data)},"{data}"')
        if out.endswith('+QESIM:"download",1'):
          print("done, successfully loaded")
          break
        assert out.endswith('OK')

  def enable(self, profile):
    pass

  def disable(self, profile):
    pass

  def delete(self, profile):
    pass

  def list_profiles(self):
    out = self.at('AT+QESIM="list"')
    return out.strip().splitlines()[1:]


if __name__ == "__main__":
  import sys

  # restart first, easy to get it into a bad state
  subprocess.check_call("sudo systemctl stop ModemManager", shell=True)
  subprocess.check_call("/usr/comma/lte/lte.sh stop_blocking", shell=True)
  subprocess.check_call("/usr/comma/lte/lte.sh start", shell=True)
  while not os.path.exists('/dev/ttyUSB2'):
    time.sleep(1)

  lpa = LPA()
  print(lpa.list_profiles())
  if len(sys.argv) > 1:
    lpa.download(sys.argv[1])
    print(lpa.list_profiles())
