#!/usr/bin/env python3
import sys
from subprocess import check_output, CalledProcessError
from panda import Panda
from panda.python.uds import UdsClient, MessageTimeoutError, SESSION_TYPE, DTC_GROUP_TYPE

try:
  check_output(["pidof", "boardd"])
  print("boardd is running, please kill openpilot before running this script! (aborted)")
  sys.exit(1)
except CalledProcessError as e:
  if e.returncode != 1: # 1 == no process found (boardd not running)
    raise e

panda = Panda()
panda.set_safety_mode(Panda.SAFETY_ELM327)
address = 0x7DF # functional (broadcast) address
uds_client = UdsClient(panda, address, bus=0, debug=False)
print("extended diagnostic session ...")
try:
  uds_client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
except MessageTimeoutError:
  pass # functional address isn't properly handled so a timeout occurs
print("clear diagnostic info ...")
try:
  uds_client.clear_diagnostic_information(DTC_GROUP_TYPE.ALL)
except MessageTimeoutError:
  pass # functional address isn't properly handled so a timeout occurs
print("")
print("you may need to power cycle your vehicle now")
