#!/usr/bin/env python3
import json
import os
import subprocess
import time
from dataclasses import dataclass

@dataclass
class Profile:
  iccid: str
  isdp_aid: str
  nickname: str
  enabled: bool
  provider: str

class LPAError(Exception):
  pass


class LPA:
  def __init__(self):
    self.env = os.environ.copy()
    self.env['LPAC_APDU'] = 'qmi'
    self.env['QMI_DEVICE'] = '/dev/cdc-wdm0'

    self.timeout_sec = 30

  def list_profiles(self):
    """
    List all profiles on the eUICC.
    """
    raw = self._invoke('profile', 'list')[-1] # only one message

    profiles = []
    for profile in raw['payload']['data']:
      profiles.append(Profile(
        iccid=profile['iccid'],
        isdp_aid=profile['isdpAid'],
        nickname=profile['profileNickname'],
        enabled=profile['profileState'] == 'enabled',
        provider=profile['serviceProviderName'],
      ))

    return profiles

  def profile_exists(self, iccid: str) -> bool:
    """
    Check if a profile exists on the eUICC.
    """
    profiles = self.list_profiles()
    return any(profile.iccid == iccid for profile in profiles)

  def get_active_profile(self):
    """
    Get the active profile on the eUICC.
    """
    profiles = self.list_profiles()
    for profile in profiles:
      if profile.enabled:
        return profile
    return None

  def process_notifications(self) -> None:
    """
    Process notifications from the LPA, typically to activate/deactivate the profile with the carrier.
    """
    msgs = self._invoke('notification', 'process', '-a', '-r')
    assert msgs[-1]['payload']['message'] == 'success', 'expected success notification'

  def enable_profile(self, iccid: str) -> None:
    """
    Enable the profile on the eUICC.
    """
    self.validate_profile(iccid)
    latest = self.get_active_profile()
    if latest is not None and latest.iccid == iccid:
      raise LPAError(f'profile {iccid} is already enabled')
    elif latest is not None:
      self.disable_profile(latest.iccid)

    msgs = self._invoke('profile', 'enable', iccid)
    assert msgs[-1]['payload']['message'] == 'success', 'expected success notification'
    self.process_notifications()

  def disable_profile(self, iccid: str) -> None:
    """
    Disable the profile on the eUICC.
    """
    self.validate_profile(iccid)
    latest = self.get_active_profile()
    if latest is None:
      return
    if latest.iccid != iccid:
      raise LPAError(f'profile {iccid} is not enabled')

    msgs = self._invoke('profile', 'disable', iccid)
    assert msgs[-1]['payload']['message'] == 'success', 'expected success notification'
    self.process_notifications()


  def validate_profile(self, iccid: str) -> None:
    """
    Validate the profile on the eUICC.
    """
    if not self.profile_exists(iccid):
      raise LPAError(f'profile {iccid} does not exist')


  def _invoke(self, *cmd: str):
    print(f"-> lpac {' '.join(list(cmd))}")
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


if __name__ == "__main__":
  import sys

  lpa = LPA()
  print(lpa.list_profiles())

  if len(sys.argv) > 2:
    if sys.argv[1] == 'enable':
      lpa.enable_profile(sys.argv[2])
    elif sys.argv[1] == 'disable':
      lpa.disable_profile(sys.argv[2])
    else:
      raise Exception(f"invalid command: {sys.argv[1]}")

  if "RESTART" in os.environ:
    subprocess.check_call("sudo systemctl stop ModemManager", shell=True)
    subprocess.check_call("/usr/comma/lte/lte.sh stop_blocking", shell=True)
    subprocess.check_call("/usr/comma/lte/lte.sh start", shell=True)
    while not os.path.exists('/dev/ttyUSB2'):
      time.sleep(1)
