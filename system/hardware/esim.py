#!/usr/bin/env python3

import argparse
import time
from openpilot.system.hardware import HARDWARE


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='esim.py', description='manage eSIM profiles on your comma device', epilog='comma.ai')
  parser.add_argument('--backend', choices=['qmi', 'at'], default='qmi', help='use the specified backend, defaults to qmi')
  parser.add_argument('--switch', metavar='iccid', help='switch to profile')
  parser.add_argument('--delete', metavar='iccid', help='delete profile (warning: this cannot be undone)')
  parser.add_argument('--download', nargs=2, metavar=('qr', 'name'), help='download a profile using QR code (format: LPA:1$rsp.truphone.com$QRF-SPEEDTEST)')
  parser.add_argument('--nickname', nargs=2, metavar=('iccid', 'name'), help='update the nickname for a profile')
  args = parser.parse_args()

  mutated = False
  lpa = HARDWARE.get_sim_lpa()
  if args.switch:
    lpa.switch_profile(args.switch)
    mutated = True
  elif args.delete:
    confirm = input('are you sure you want to delete this profile? (y/N) ')
    if confirm == 'y':
      lpa.delete_profile(args.delete)
      mutated = True
    else:
      print('cancelled')
      exit(0)
  elif args.download:
    lpa.download_profile(args.download[0], args.download[1])
  elif args.nickname:
    lpa.nickname_profile(args.nickname[0], args.nickname[1])
  else:
    parser.print_help()

  if mutated:
    HARDWARE.reboot_modem()
    # eUICC needs a small delay post-reboot before querying profiles
    time.sleep(.5)

  profiles = lpa.list_profiles()
  print(f'\n{len(profiles)} profile{"s" if len(profiles) > 1 else ""}:')
  for p in profiles:
    print(f'- {p.iccid} (nickname: {p.nickname or "<none provided>"}) (provider: {p.provider}) - {"enabled" if p.enabled else "disabled"}')
