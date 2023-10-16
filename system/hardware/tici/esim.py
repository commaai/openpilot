#!/usr/bin/env python3
import time
import subprocess
from typing import Optional


def at(cmd: str) -> Optional[str]:
  for _ in range(3):
    try:
      return subprocess.check_output(f"mmcli -m any --timeout 30 --command='{cmd}'", shell=True, encoding='utf8')
    except subprocess.CalledProcessError:
      cloudlog.exception("rawgps.mmcli_command_failed")
      time.sleep(1.0)
  raise Exception(f"failed to execute mmcli command {cmd=}")


class LPA:
  def download(self, qr):
    pass

  def enable(self, profile):
    pass

  def disable(self, profile):
    pass

  def delete(self, profile):
    pass

  def list_profiles(self):
    out = at('AT+QESIM="list"')
    profiles = out.strip().splitlines()[1:]
    return profiles

if __name__ == "__main__":
  lpa = LPA()
  print(lpa.list_profiles())
