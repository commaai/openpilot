#!/usr/bin/env python3
from openpilot.system.athena.manage_athenad import manage_athenad

if __name__ == '__main__':
  manage_athenad("SunnylinkDongleId", "SunnylinkdPid", 'sunnylinkd', 'sunnypilot.sunnylink.athena.sunnylinkd')
