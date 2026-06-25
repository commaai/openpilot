#!/usr/bin/env python3

import argparse
import sys
import time
from openpilot.common.hardware import HARDWARE
from openpilot.common.esim.base import LPABase, Profile


def sorted_profiles(lpa: LPABase) -> list[Profile]:
  return sorted(lpa.list_profiles(), key=lambda p: p.iccid)


def resolve_iccid(lpa: LPABase, ref: str) -> str:
  # ref is either a 1-based index into the sorted list, or a literal iccid
  if ref.isdigit():
    profiles = sorted_profiles(lpa)
    idx = int(ref) - 1
    if not 0 <= idx < len(profiles):
      raise SystemExit(f'no profile at index {ref} (have {len(profiles)})')
    return profiles[idx].iccid
  return ref


def print_profiles(lpa: LPABase) -> None:
  profiles = sorted_profiles(lpa)
  print(f'\n{len(profiles)} profile{"s" if len(profiles) != 1 else ""}:')
  for i, p in enumerate(profiles, start=1):
    print(f'{i}. {p.iccid} (nickname: {p.nickname or "<none provided>"}) (provider: {p.provider}) - {"enabled" if p.enabled else "disabled"}')


def execute_and_process_notifications(lpa: LPABase, operation) -> None:
  try:
    operation()
  finally:
    time.sleep(1)
    try:
      lpa.process_notifications()
    except Exception as e:
      print(f'failed to process eSIM notifications: {e}', file=sys.stderr)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='esim.py', description='manage eSIM profiles on your comma device', epilog='comma.ai')
  sub = parser.add_subparsers(dest='cmd')

  sub.add_parser('list', help='list profiles')

  p_switch = sub.add_parser('switch', help='switch to profile')
  p_switch.add_argument('profile', help='iccid or 1-based index from `list`')

  p_delete = sub.add_parser('delete', help='delete profile (warning: this cannot be undone)')
  p_delete.add_argument('profile', help='iccid or 1-based index from `list`')

  p_download = sub.add_parser('download', help='download a profile using QR code (format: LPA:1$rsp.truphone.com$QRF-SPEEDTEST)')
  p_download.add_argument('qr')
  p_download.add_argument('name')

  p_nickname = sub.add_parser('nickname', help='update the nickname for a profile')
  p_nickname.add_argument('profile', help='iccid or 1-based index from `list`')
  p_nickname.add_argument('name')

  args = parser.parse_args()

  lpa = HARDWARE.get_sim_lpa()
  if args.cmd == 'switch':
    iccid = resolve_iccid(lpa, args.profile)
    execute_and_process_notifications(lpa, lambda: lpa.switch_profile(iccid))
  elif args.cmd == 'delete':
    iccid = resolve_iccid(lpa, args.profile)
    confirm = input(f'are you sure you want to delete profile {iccid}? (y/N) ')
    if confirm == 'y':
      execute_and_process_notifications(lpa, lambda: lpa.delete_profile(iccid))
    else:
      print('cancelled')
      exit(0)
  elif args.cmd == 'download':
    execute_and_process_notifications(lpa, lambda: lpa.download_profile(args.qr, args.name))
  elif args.cmd == 'nickname':
    lpa.nickname_profile(resolve_iccid(lpa, args.profile), args.name)
  else:
    if args.cmd is None:
      parser.print_help()
    print_profiles(lpa)
