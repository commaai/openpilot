#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

@dataclass
class Profile:
  iccid: str
  nickname: str
  enabled: bool
  provider: str

class LPAError(RuntimeError):
  pass

class LPAProfileNotFoundError(LPAError):
  pass


class LPA:
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

  def enable_profile(self, iccid: str) -> None:
    self._validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest:
      if latest.iccid == iccid:
        return
      self.disable_profile(latest.iccid)
    self._validate_successful(self._invoke('profile', 'enable', iccid))
    self.process_notifications()

  def disable_profile(self, iccid: str) -> None:
    self._validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest is not None and latest.iccid != iccid:
      return
    self._validate_successful(self._invoke('profile', 'disable', iccid))
    self.process_notifications()

  def delete_profile(self, iccid: str) -> None:
    self._validate_profile_exists(iccid)
    latest = self.get_active_profile()
    if latest is not None and latest.iccid == iccid:
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

  def _validate_profile_exists(self, iccid: str) -> None:
    if not any(p.iccid == iccid for p in self.list_profiles()):
      raise LPAProfileNotFoundError(f'profile {iccid} does not exist')

  def _validate_successful(self, msgs: list[dict]) -> None:
    assert msgs[-1]['payload']['message'] == 'success', 'expected success notification'


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='esim.py', description='manage eSIM profiles on your comma device', epilog='comma.ai')
  parser.add_argument('--backend', choices=['qmi', 'at'], default='qmi', help='use the specified backend, defaults to qmi')
  parser.add_argument('--enable', metavar='iccid', help='enable profile; will disable current profile')
  parser.add_argument('--disable', metavar='iccid', help='disable profile')
  parser.add_argument('--delete', metavar='iccid', help='delete profile (warning: this cannot be undone)')
  parser.add_argument('--download', nargs=2, metavar=('qr', 'name'), help='download a profile using QR code (format: LPA:1$rsp.truphone.com$QRF-SPEEDTEST)')
  parser.add_argument('--nickname', nargs=2, metavar=('iccid', 'name'), help='update the nickname for a profile')
  args = parser.parse_args()

  lpa = LPA(interface=args.backend)
  if args.enable:
    lpa.enable_profile(args.enable)
    print('enabled profile, please restart device to apply changes')
  elif args.disable:
    lpa.disable_profile(args.disable)
    print('disabled profile, please restart device to apply changes')
  elif args.delete:
    confirm = input('are you sure you want to delete this profile? (y/N) ')
    if confirm == 'y':
      lpa.delete_profile(args.delete)
      print('deleted profile, please restart device to apply changes')
    else:
      print('cancelled')
      exit(0)
  elif args.download:
    lpa.download_profile(args.download[0], args.download[1])
  elif args.nickname:
    lpa.nickname_profile(args.nickname[0], args.nickname[1])
  else:
    parser.print_help()

  profiles = lpa.list_profiles()
  print(f'\n{len(profiles)} profile{"s" if len(profiles) > 1 else ""}:')
  for p in profiles:
    print(f'- {p.iccid} (nickname: {p.nickname or "<none provided>"}) (provider: {p.provider}) - {"enabled" if p.enabled else "disabled"}')
