import json
import os
import shutil
import subprocess
from typing import Literal

from openpilot.system.hardware.base import LPABase, LPAError, LPAProfileNotFoundError, Profile

class TiciLPA(LPABase):
  def __init__(self, interface: Literal['qmi', 'at'] = 'qmi'):
    self.env = os.environ.copy()
    self.env['LPAC_APDU'] = interface
    self.env['QMI_DEVICE'] = '/dev/cdc-wdm0'
    self.env['AT_DEVICE'] = '/dev/ttyUSB2'

    self.timeout_sec = 45

    if shutil.which('lpac') is None:
      raise LPAError('lpac not found, must be installed!')

  def list_profiles(self) -> list[Profile]:
    msgs = self._invoke('profile', 'list')
    self._validate_successful(msgs)
    return [Profile(
      iccid=p['iccid'],
      nickname=p['profileNickname'],
      enabled=p['profileState'] == 'enabled',
      provider=p['serviceProviderName']
    ) for p in msgs[-1]['payload']['data']]

  def get_active_profile(self) -> Profile | None:
    return next((p for p in self.list_profiles() if p.enabled), None)

  def delete_profile(self, iccid: str) -> None:
    self._validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest is not None and latest.iccid == iccid:
      raise LPAError('cannot delete active profile, switch to another profile first')
    self._validate_successful(self._invoke('profile', 'delete', iccid))
    self._process_notifications()

  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    msgs = self._invoke('profile', 'download', '-a', qr)
    self._validate_successful(msgs)
    new_profile = next((m for m in msgs if m['payload']['message'] == 'es8p_meatadata_parse'), None)
    if new_profile is None:
      raise LPAError('no new profile found')
    if nickname:
      self.nickname_profile(new_profile['payload']['data']['iccid'], nickname)
    self._process_notifications()

  def nickname_profile(self, iccid: str, nickname: str) -> None:
    self._validate_profile_exists(iccid)
    self._validate_successful(self._invoke('profile', 'nickname', iccid, nickname))

  def switch_profile(self, iccid: str) -> None:
    self._validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest and latest.iccid == iccid:
      return
    self._validate_successful(self._invoke('profile', 'enable', iccid))
    self._process_notifications()

  def _invoke(self, *cmd: str):
    proc = subprocess.Popen(['sudo', '-E', 'lpac'] + list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
    try:
      out, err = proc.communicate(timeout=self.timeout_sec)
    except subprocess.TimeoutExpired as e:
      proc.kill()
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
        assert 'data' in message['payload'], 'expected data in message payload'

        msg_ret_code = message['payload']['code']
        if msg_ret_code != 0:
          raise LPAError(f"lpac {' '.join(cmd)} failed with code {msg_ret_code}: <{message['payload']['message']}> {message['payload']['data']}")

        messages.append(message)

    if len(messages) == 0:
      raise LPAError(f"lpac {cmd} returned no messages")

    return messages

  def _process_notifications(self) -> None:
    """
    Process notifications stored on the eUICC, typically to activate/deactivate the profile with the carrier.
    """
    self._validate_successful(self._invoke('notification', 'process', '-a', '-r'))

  def _validate_profile_exists(self, iccid: str) -> None:
    if not any(p.iccid == iccid for p in self.list_profiles()):
      raise LPAProfileNotFoundError(f'profile {iccid} does not exist')

  def _validate_successful(self, msgs: list[dict]) -> None:
    assert msgs[-1]['payload']['message'] == 'success', 'expected success notification'
