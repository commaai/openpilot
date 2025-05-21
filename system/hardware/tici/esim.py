#!/usr/bin/env python3
import json
import os
import math
import time
import binascii
import requests
import serial
import subprocess


class LPAError(RuntimeError):
  pass

class LPAProfileNotFoundError(LPAError):
  pass


class LPA2:
  def __init__(self):
    self.env = os.environ.copy()
    self.env['LPAC_APDU'] = 'qmi'
    self.env['QMI_DEVICE'] = '/dev/cdc-wdm0'

    self.timeout_sec = 45

  def list_profiles(self) -> list[dict[str, str]]:
    msgs = self._invoke('profile', 'list')
    self.validate_successful(msgs)

    profiles = []
    for profile in msgs[-1]['payload']['data']:
      profiles.append({
        'iccid': profile['iccid'],
        'nickname': profile['profileNickname'],
        'enabled': profile['profileState'] == 'enabled',
        'provider': profile['serviceProviderName'],
      })

    return profiles

  def get_active_profile(self) -> dict[str, str] | None:
    return next((p for p in self.list_profiles() if p['enabled']), None)

  def enable_profile(self, iccid: str) -> None:
    self.validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest is None:
      raise LPAError('no profile enabled')
    if latest['iccid'] == iccid:
      raise LPAError(f'profile {iccid} is already enabled')
    else:
      self.disable_profile(latest['iccid'])

    self.validate_successful(self._invoke('profile', 'enable', iccid))
    self.process_notifications()

  def disable_profile(self, iccid: str) -> None:
    self.validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest is None:
      raise LPAError('no profile enabled')
    if latest['iccid'] != iccid:
      raise LPAError(f'profile {iccid} is not enabled')

    self.validate_successful(self._invoke('profile', 'disable', iccid))
    self.process_notifications()

  def delete_profile(self, iccid: str) -> None:
    self.validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest is not None and latest['iccid'] == iccid:
      self.disable_profile(iccid)
    self.validate_successful(self._invoke('profile', 'delete', iccid))
    self.process_notifications()

  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    msgs = self._invoke('profile', 'download', '-a', qr)
    self.validate_successful(msgs)
    new_profile = next((m for m in msgs if m['payload']['message'] == 'es8p_meatadata_parse'), None)
    if new_profile is None:
      raise LPAError('no new profile found')
    if nickname:
      self.nickname_profile(new_profile['payload']['data']['iccid'], nickname)
    self.process_notifications()

  def nickname_profile(self, iccid: str, nickname: str) -> None:
    self.validate_profile_exists(iccid)
    self.validate_successful(self._invoke('profile', 'nickname', iccid, nickname))

  def process_notifications(self) -> None:
    """
    Process notifications stored on the eUICC, typically to activate/deactivate the profile with the carrier.
    """
    self.validate_successful(self._invoke('notification', 'process', '-a', '-r'))

  def validate_profile_exists(self, iccid: str) -> None:
    if not any(p['iccid'] == iccid for p in self.list_profiles()):
      raise LPAProfileNotFoundError(f'profile {iccid} does not exist')

  def validate_successful(self, msgs: list[dict]) -> None:
    assert msgs[-1]['payload']['message'] == 'success', 'expected success notification'

  def _invoke(self, *cmd: str):
    ret = subprocess.Popen(['sudo', '-E', 'lpac'] + list(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
    try:
      out, err = ret.communicate(timeout=self.timeout_sec)
    except subprocess.TimeoutExpired as e:
      ret.kill()
      raise LPAError(f"lpac {cmd} timed out after {self.timeout_sec} seconds") from e

    messages = []
    for line in out.decode().strip().splitlines():
      if line.startswith('{'):
        message = json.loads(line)

        # lpac response format validations
        assert 'type' in message, 'expected type in message'
        assert message['type'] == 'lpa' or message['type'] == 'progress', 'expected lpa or progress message type'
        assert 'payload' in message, 'expected payload in message'
        assert 'code' in message['payload'], 'expected code in message payload'

        if message['payload']['code'] != 0:
          raise LPAError(f"lpac {' '.join(cmd)} failed with code {message['payload']['code']}: <{message['payload']['message']}> {message['payload']['data']}")

        assert 'data' in message['payload'], 'expected data in message payload'

        messages.append(message)

    if len(messages) == 0:
      raise LPAError(f"lpac {cmd} returned no messages")

    return messages


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

  lpa = LPA2()
  print(lpa.list_profiles())

  if len(sys.argv) > 2:
    if sys.argv[1] == 'enable':
      lpa.enable_profile(sys.argv[2])
    elif sys.argv[1] == 'disable':
      lpa.disable_profile(sys.argv[2])
    elif sys.argv[1] == 'delete':
      lpa.delete_profile(sys.argv[2])
    elif sys.argv[1] == 'download':
      assert len(sys.argv) == 4, 'expected profile nickname'
      lpa.download_profile(sys.argv[2], sys.argv[3])
    else:
      raise Exception(f"invalid command: {sys.argv[1]}")

  if "RESTART" in os.environ:
    subprocess.check_call("sudo systemctl stop ModemManager", shell=True)
    subprocess.check_call("/usr/comma/lte/lte.sh stop_blocking", shell=True)
    subprocess.check_call("/usr/comma/lte/lte.sh start", shell=True)
    while not os.path.exists('/dev/ttyUSB2'):
      time.sleep(1)
