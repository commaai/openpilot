#!/usr/bin/env python3
import sys
import argparse
from subprocess import check_output, CalledProcessError
from panda import Panda
from panda.python.uds import UdsClient, SESSION_TYPE, DTC_REPORT_TYPE, DTC_STATUS_MASK_TYPE
from panda.python.uds import get_dtc_num_as_str, get_dtc_status_names

parser = argparse.ArgumentParser(description="read DTC status")
parser.add_argument("addr", type=lambda x: int(x,0))
parser.add_argument("--bus", type=int, default=0)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

try:
  check_output(["pidof", "boardd"])
  print("boardd is running, please kill openpilot before running this script! (aborted)")
  sys.exit(1)
except CalledProcessError as e:
  if e.returncode != 1: # 1 == no process found (boardd not running)
    raise e

panda = Panda()
panda.set_safety_mode(Panda.SAFETY_ELM327)
uds_client = UdsClient(panda, args.addr, bus=args.bus, debug=args.debug)
print("extended diagnostic session ...")
uds_client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
print("read diagnostic codes ...")
data = uds_client.read_dtc_information(DTC_REPORT_TYPE.DTC_BY_STATUS_MASK, DTC_STATUS_MASK_TYPE.ALL)
print("status availability:", " ".join(get_dtc_status_names(data[0])))
print("DTC status:")
for i in range(1, len(data), 4):
  dtc_num = get_dtc_num_as_str(data[i:i+3])
  dtc_status = " ".join(get_dtc_status_names(data[i+3]))
  print(dtc_num, dtc_status)
