#!/usr/bin/env python3
import sys
import argparse
from subprocess import check_output, CalledProcessError
from opendbc.car.carlog import carlog
from opendbc.car.uds import UdsClient, MessageTimeoutError, SESSION_TYPE, DTC_GROUP_TYPE
from opendbc.car.structs import CarParams
from panda import Panda

parser = argparse.ArgumentParser(description="clear DTC status")
parser.add_argument("addr", type=lambda x: int(x,0), nargs="?", default=0x7DF) # default is functional (broadcast) address
parser.add_argument("--bus", type=int, default=0)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.debug:
  carlog.setLevel('DEBUG')

try:
  check_output(["pidof", "pandad"])
  print("pandad is running, please kill openpilot before running this script! (aborted)")
  sys.exit(1)
except CalledProcessError as e:
  if e.returncode != 1: # 1 == no process found (pandad not running)
    raise e

panda = Panda()
panda.set_safety_mode(CarParams.SafetyModel.elm327)
uds_client = UdsClient(panda, args.addr, bus=args.bus)
print("extended diagnostic session ...")
try:
  uds_client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
except MessageTimeoutError:
  # functional address isn't properly handled so a timeout occurs
  if args.addr != 0x7DF:
    raise
print("clear diagnostic info ...")
try:
  uds_client.clear_diagnostic_information(DTC_GROUP_TYPE.ALL)
except MessageTimeoutError:
  # functional address isn't properly handled so a timeout occurs
  if args.addr != 0x7DF:
    pass
print("")
print("you may need to power cycle your vehicle now")
