#!/usr/bin/env python3
import json
import time
import requests
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
  return r

def get_unsolicited_response():
  # why doesn't modem manager directly return this?
  time.sleep(0.1)
  out = subprocess.check_output("journalctl -o cat -u ModemManager | grep QESIM", shell=True, encoding='utf8')
  return out.split('+QESIM:')[-1].split('<CR><LF><CR><LF>OK<CR><LF>')[0]

class LPA:
  def download_ota(self, qr):
    return at(f'AT+QESIM="ota","{qr}"')

  def download(self, qr):
    out = at(f'AT+QESIM="download","{qr}"')
    print(repr(out))

    out = get_unsolicited_response()
    print("line", repr(out))

    parts = [x.strip().strip('"') for x in out.split(',', maxsplit=4)]
    print(repr(parts))
    trans, ret, url, payloadlen, payload = parts
    assert trans == "trans" and ret == "0"
    assert len(payload) == int(payloadlen)

    # TODO: double check this against the QR code
    smdp = json.loads(payload)['smdpAddress']
    r = post(f"https://{smdp}/{url}", payload)

    # do the download
    for i in range(1):
      at('AT+QESIM="trans"')

  def enable(self, profile):
    pass

  def disable(self, profile):
    pass

  def delete(self, profile):
    pass

  def list_profiles(self):
    out = at('AT+QESIM="list"')
    return out.strip().splitlines()[1:]


if __name__ == "__main__":
  import sys

  lpa = LPA()
  if len(sys.argv) > 1:
    # restart first, easy to get it into a bad state
    subprocess.check_call("/usr/comma/lte/lte.sh stop_blocking", shell=True)
    subprocess.check_call("/usr/comma/lte/lte.sh start", shell=True)
    subprocess.check_call("sudo systemctl restart ModemManager", shell=True)
    out = ""
    while "QUECTEL Mobile Broadband Module" not in out:
      try:
        out = subprocess.check_output("mmcli -L", shell=True, encoding='utf8')
      except subprocess.CalledProcessError:
        pass
    print("got modem")
    time.sleep(1)

    lpa.download(sys.argv[1])
    print(lpa.list_profiles())
