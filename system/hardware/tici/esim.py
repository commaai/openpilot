#!/usr/bin/env python3

import json
import os
import subprocess
import time


class LPAError(RuntimeError):
  pass

class LPAProfileNotFoundError(LPAError):
  pass


class LPA:
  def __init__(self):
    self.env = os.environ.copy()
    self.env['LPAC_APDU'] = 'qmi'
    self.env['QMI_DEVICE'] = '/dev/cdc-wdm0'

    self.timeout_sec = 45

  def list_profiles(self) -> list[dict[str, str]]:
    msgs = self._invoke('profile', 'list')
    self._validate_successful(msgs)
    return [{
      'iccid': p['iccid'],
      'nickname': p['profileNickname'],
      'enabled': p['profileState'] == 'enabled',
      'provider': p['serviceProviderName']
    } for p in msgs[-1]['payload']['data']]

  def get_active_profile(self) -> dict[str, str] | None:
    return next((p for p in self.list_profiles() if p['enabled']), None)

  def enable_profile(self, iccid: str) -> None:
    self._validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest:
      if latest['iccid'] == iccid:
        raise LPAError(f'profile {iccid} is already enabled')
      self.disable_profile(latest['iccid'])
    self._validate_successful(self._invoke('profile', 'enable', iccid))
    self.process_notifications()

  def disable_profile(self, iccid: str) -> None:
    self._validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest is not None and latest['iccid'] != iccid:
      return
    self._validate_successful(self._invoke('profile', 'disable', iccid))
    self.process_notifications()

  def delete_profile(self, iccid: str) -> None:
    self._validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest is not None and latest['iccid'] == iccid:
      self.disable_profile(iccid)
    self._validate_successful(self._invoke('profile', 'delete', iccid))
    self.process_notifications()

  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    msgs = self._invoke('profile', 'download', '-a', qr)
    self._validate_successful(msgs)
    new_profile = next((m for m in msgs if m['payload']['message'] == 'es8p_meatadata_parse'), None)
    if new_profile is None:
      raise LPAError('no new profile found')
    if nickname:
      self.nickname_profile(new_profile['payload']['data']['iccid'], nickname)
    self.process_notifications()

  def nickname_profile(self, iccid: str, nickname: str) -> None:
    self._validate_profile_exists(iccid)
    self._validate_successful(self._invoke('profile', 'nickname', iccid, nickname))

  def process_notifications(self) -> None:
    """
    Process notifications stored on the eUICC, typically to activate/deactivate the profile with the carrier.
    """
    self._validate_successful(self._invoke('notification', 'process', '-a', '-r'))

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

  def _validate_profile_exists(self, iccid: str) -> None:
    if not any(p['iccid'] == iccid for p in self.list_profiles()):
      raise LPAProfileNotFoundError(f'profile {iccid} does not exist')

  def _validate_successful(self, msgs: list[dict]) -> None:
    assert msgs[-1]['payload']['message'] == 'success', 'expected success notification'


if __name__ == "__main__":
  import sys

  lpa = LPA()
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
