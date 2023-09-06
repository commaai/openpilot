#!/usr/bin/env python3
# simple boardd wrapper that updates the panda first
import os
import usb1
import time
import json
import subprocess
from typing import List, NoReturn
from functools import cmp_to_key

from panda import Panda, PandaDFU, PandaProtocolMismatch, FW_PATH
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.boardd.set_time import set_time
from openpilot.system.hardware import HARDWARE
from openpilot.system.swaglog import cloudlog


def get_expected_signature(panda: Panda) -> bytes:
  try:
    fn = os.path.join(FW_PATH, panda.get_mcu_type().config.app_fn)
    return Panda.get_signature_from_firmware(fn)
  except Exception:
    cloudlog.exception("Error computing expected signature")
    return b""

def read_panda_logs(panda: Panda) -> None:
  """
    Forward panda logs to the cloud
  """

  params = Params()
  serial = panda.get_usb_serial()

  log_state = {}
  try:
    l = json.loads(params.get("PandaLogState"))
    for k, v in l.items():
      if isinstance(k, str) and isinstance(v, int):
        log_state[k] = v
  except (TypeError, json.JSONDecodeError):
    cloudlog.exception("failed to parse PandaLogState")

  try:
    if serial in log_state:
      logs = panda.get_logs(last_id=log_state[serial])
    else:
      logs = panda.get_logs(get_all=True)

    # truncate logs to 100 entries if needed
    MAX_LOGS = 100
    if len(logs) > MAX_LOGS:
      cloudlog.warning(f"Panda {serial} has {len(logs)} logs, truncating to {MAX_LOGS}")
      logs = logs[-MAX_LOGS:]

    # update log state
    if len(logs) > 0:
      log_state[serial] = logs[-1]["id"]

    for log in logs:
      if log['timestamp'] is not None:
        log['timestamp'] = log['timestamp'].isoformat()
      cloudlog.event("panda_log", **log, serial=serial)

    params.put("PandaLogState", json.dumps(log_state))
  except Exception:
    cloudlog.exception(f"Error getting logs for panda {serial}")


def flash_panda(panda_serial: str) -> Panda:
  try:
    panda = Panda(panda_serial)
  except PandaProtocolMismatch:
    cloudlog.warning("detected protocol mismatch, reflashing panda")
    HARDWARE.recover_internal_panda()
    raise

  fw_signature = get_expected_signature(panda)
  internal_panda = panda.is_internal()

  panda_version = "bootstub" if panda.bootstub else panda.get_version()
  panda_signature = b"" if panda.bootstub else panda.get_signature()
  cloudlog.warning(f"Panda {panda_serial} connected, version: {panda_version}, signature {panda_signature.hex()[:16]}, expected {fw_signature.hex()[:16]}")

  if panda.bootstub or panda_signature != fw_signature:
    cloudlog.info("Panda firmware out of date, update required")
    panda.flash()
    cloudlog.info("Done flashing")

  if panda.bootstub:
    bootstub_version = panda.get_version()
    cloudlog.info(f"Flashed firmware not booting, flashing development bootloader. {bootstub_version=}, {internal_panda=}")
    if internal_panda:
      HARDWARE.recover_internal_panda()
    panda.recover(reset=(not internal_panda))
    cloudlog.info("Done flashing bootstub")

  if panda.bootstub:
    cloudlog.info("Panda still not booting, exiting")
    raise AssertionError

  panda_signature = panda.get_signature()
  if panda_signature != fw_signature:
    cloudlog.info("Version mismatch after flashing, exiting")
    raise AssertionError

  return panda


def panda_sort_cmp(a: Panda, b: Panda):
  a_type = a.get_type()
  b_type = b.get_type()

  # make sure the internal one is always first
  if a.is_internal() and not b.is_internal():
    return -1
  if not a.is_internal() and b.is_internal():
    return 1

  # sort by hardware type
  if a_type != b_type:
    return a_type < b_type

  # last resort: sort by serial number
  return a.get_usb_serial() < b.get_usb_serial()


def main() -> NoReturn:
  count = 0
  first_run = True
  params = Params()
  no_internal_panda_count = 0

  while True:
    try:
      count += 1
      cloudlog.event("pandad.flash_and_connect", count=count)
      params.remove("PandaSignatures")

      # Handle missing internal panda
      if no_internal_panda_count > 0:
        if no_internal_panda_count == 3:
          cloudlog.info("No pandas found, putting internal panda into DFU")
          HARDWARE.recover_internal_panda()
        else:
          cloudlog.info("No pandas found, resetting internal panda")
          HARDWARE.reset_internal_panda()
        time.sleep(3)  # wait to come back up

      # Flash all Pandas in DFU mode
      dfu_serials = PandaDFU.list()
      if len(dfu_serials) > 0:
        for serial in dfu_serials:
          cloudlog.info(f"Panda in DFU mode found, flashing recovery {serial}")
          PandaDFU(serial).recover()
        time.sleep(1)

      panda_serials = Panda.list()
      if len(panda_serials) == 0:
        no_internal_panda_count += 1
        continue

      cloudlog.info(f"{len(panda_serials)} panda(s) found, connecting - {panda_serials}")

      # Flash pandas
      pandas: List[Panda] = []
      for serial in panda_serials:
        pandas.append(flash_panda(serial))

      # Ensure internal panda is present if expected
      internal_pandas = [panda for panda in pandas if panda.is_internal()]
      if HARDWARE.has_internal_panda() and len(internal_pandas) == 0:
        cloudlog.error("Internal panda is missing, trying again")
        no_internal_panda_count += 1
        continue
      no_internal_panda_count = 0

      # sort pandas to have deterministic order
      pandas.sort(key=cmp_to_key(panda_sort_cmp))
      panda_serials = [p.get_usb_serial() for p in pandas]

      # log panda fw versions
      params.put("PandaSignatures", b','.join(p.get_signature() for p in pandas))

      for panda in pandas:
        # check health for lost heartbeat
        health = panda.health()
        if health["heartbeat_lost"]:
          params.put_bool("PandaHeartbeatLost", True)
          cloudlog.event("heartbeat lost", deviceState=health, serial=panda.get_usb_serial())

        read_panda_logs(panda)

        if first_run:
          if panda.is_internal():
            # update time from RTC
            set_time(cloudlog)

          # reset panda to ensure we're in a good state
          cloudlog.info(f"Resetting panda {panda.get_usb_serial()}")
          if panda.is_internal():
            HARDWARE.reset_internal_panda()
          else:
            panda.reset(reconnect=False)

      for p in pandas:
        p.close()
    # TODO: wrap all panda exceptions in a base panda exception
    except (usb1.USBErrorNoDevice, usb1.USBErrorPipe):
      # a panda was disconnected while setting everything up. let's try again
      cloudlog.exception("Panda USB exception while setting up")
      continue
    except PandaProtocolMismatch:
      cloudlog.exception("pandad.protocol_mismatch")
      continue
    except Exception:
      cloudlog.exception("pandad.uncaught_exception")
      continue

    first_run = False

    # run boardd with all connected serials as arguments
    os.environ['MANAGER_DAEMON'] = 'boardd'
    os.chdir(os.path.join(BASEDIR, "selfdrive/boardd"))
    subprocess.run(["./boardd", *panda_serials], check=True)

if __name__ == "__main__":
  main()
