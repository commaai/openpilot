#!/usr/bin/env python3
# simple boardd wrapper that updates the panda first
import os
import usb1
import time
import subprocess
from typing import NoReturn

from panda import Panda, PandaDFU, PandaProtocolMismatch, FW_PATH
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog


def get_expected_signature(panda: Panda) -> bytes:
  try:
    fn = os.path.join(FW_PATH, panda.get_mcu_type().config.app_fn)
    return Panda.get_signature_from_firmware(fn)
  except Exception:
    cloudlog.exception("Error computing expected signature")
    return b""

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

      # TODO: remove this in the next AGNOS
      # wait until USB is up before counting
      if time.monotonic() < 25.:
        no_internal_panda_count = 0

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
      pandas: list[Panda] = []
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
      # * the internal one is always first
      # * then sort by hardware type
      # * as a last resort, sort by serial number
      pandas.sort(key=lambda x: (not x.is_internal(), x.get_type(), x.get_usb_serial()))
      panda_serials = [p.get_usb_serial() for p in pandas]

      # log panda fw versions
      params.put("PandaSignatures", b','.join(p.get_signature() for p in pandas))

      for panda in pandas:
        # check health for lost heartbeat
        health = panda.health()
        if health["heartbeat_lost"]:
          params.put_bool("PandaHeartbeatLost", True)
          cloudlog.event("heartbeat lost", deviceState=health, serial=panda.get_usb_serial())
        if health["som_reset_triggered"]:
          params.put_bool("PandaSomResetTriggered", True)
          cloudlog.event("panda.som_reset_triggered", health=health, serial=panda.get_usb_serial())

        if first_run:
          # reset panda to ensure we're in a good state
          cloudlog.info(f"Resetting panda {panda.get_usb_serial()}")
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
