#!/usr/bin/env python3
import os
import math
import time
import binascii
import requests
import serial
import subprocess


def post(url, payload):
  print()
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
  print()
  r.raise_for_status()

  ret = f"HTTP/1.1 {r.status_code}"
  ret += ''.join(f"{k}: {v}" for k, v in r.headers.items() if k != 'Connection')
  return ret.encode() + r.content


class LPA:
  def __init__(self):
    self.dev = serial.Serial('/dev/ttyUSB2', baudrate=57600, timeout=1, bytesize=8)
    self.dev.reset_input_buffer()
    self.dev.reset_output_buffer()
    assert "OK" in self.at("AT")

  def at(self, cmd):
    print(f"==> {cmd}")
    self.dev.write(cmd.encode() + b'\r\n')

    r = b""
    cnt = 0
    while b"OK" not in r and b"ERROR" not in r and cnt < 20:
      r += self.dev.read(8192).strip()
      cnt += 1
    r = r.decode()
    print(f"<== {repr(r)}")
    return r

  def download_ota(self, qr):
    return self.at(f'AT+QESIM="ota","{qr}"')

  def download(self, qr):
    smdp = qr.split('$')[1]
    out = self.at(f'AT+QESIM="download","{qr}"')
    for _ in range(5):
      line = out.split("+QESIM: ")[1].split("\r\n\r\nOK")[0]

      parts = [x.strip().strip('"') for x in line.split(',', maxsplit=4)]
      print(repr(parts))
      trans, ret, url, payloadlen, payload = parts
      assert trans == "trans" and ret == "0"
      assert len(payload) == int(payloadlen)

      r = post(f"https://{smdp}/{url}", payload)
      to_send = binascii.hexlify(r).decode()

      chunk_len = 1400
      for i in range(math.ceil(len(to_send) / chunk_len)):
        state = 1 if (i+1)*chunk_len < len(to_send) else 0
        data = to_send[i * chunk_len : (i+1)*chunk_len]
        out = self.at(f'AT+QESIM="trans",{len(to_send)},{state},{i},{len(data)},"{data}"')
        assert "OK" in out

      if '+QESIM:"download",1' in out:
        raise Exception("profile install failed")
      elif '+QESIM:"download",0' in out:
        print("done, successfully loaded")
        break

  def enable(self, iccid):
    self.at(f'AT+QESIM="enable","{iccid}"')

  def disable(self, iccid):
    self.at(f'AT+QESIM="disable","{iccid}"')

  def delete(self, iccid):
    self.at(f'AT+QESIM="delete","{iccid}"')

  def list_profiles(self):
    out = self.at('AT+QESIM="list"')
    return out.strip().splitlines()[1:]


if __name__ == "__main__":
  import sys

  if "RESTART" in os.environ:
    subprocess.check_call("sudo systemctl stop ModemManager", shell=True)
    subprocess.check_call("/usr/comma/lte/lte.sh stop_blocking", shell=True)
    subprocess.check_call("/usr/comma/lte/lte.sh start", shell=True)
    while not os.path.exists('/dev/ttyUSB2'):
      time.sleep(1)
    time.sleep(3)

  lpa = LPA()
  print(lpa.list_profiles())
  if len(sys.argv) > 1:
    lpa.download(sys.argv[1])
    print(lpa.list_profiles())
