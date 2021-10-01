#!/usr/bin/env python3
# simple boardd wrapper that updates the panda first
import os
import time

from typing import List

from panda import BASEDIR as PANDA_BASEDIR, Panda, PandaDFU
from common.basedir import BASEDIR
from common.params import Params
from hardware import TICI
from selfdrive.swaglog import cloudlog

PANDA_FW_FN = os.path.join(PANDA_BASEDIR, "board", "obj", "panda.bin.signed")
PANDA_H7_FW_FN = os.path.join(PANDA_BASEDIR, "board", "obj", "panda_h7.bin.signed")

PERIPHERAL_TYPES = [Panda.HW_TYPE_UNO, Panda.HW_TYPE_DOS]


def get_expected_signature(fn) -> bytes:
  try:
    return Panda.get_signature_from_firmware(fn)
  except Exception:
    cloudlog.exception("Error computing expected signature")
    return b""


def get_fw_fn(panda : Panda) -> str:
  if panda._mcu_type == 2:
    return PANDA_H7_FW_FN
  else:
    return PANDA_FW_FN


def flash_panda(panda_serial : str) -> Panda:
  panda = Panda(panda_serial)

  fw_fn = get_fw_fn(panda)
  fw_signature = get_expected_signature(fw_fn)

  panda_version = "bootstub" if panda.bootstub else panda.get_version()
  panda_signature = b"" if panda.bootstub else panda.get_signature()
  cloudlog.warning(f"Panda %s connected, version: %s, signature %s, expected %s" % (
    panda_serial,
    panda_version,
    panda_signature.hex()[:16],
    fw_signature.hex()[:16],
  ))

  if panda.bootstub or panda_signature != fw_signature:
    cloudlog.info("Panda firmware out of date, update required")
    panda.flash()
    cloudlog.info("Done flashing")

  if panda.bootstub:
    bootstub_version = panda.get_version()
    cloudlog.info(f"Flashed firmware not booting, flashing development bootloader. Bootstub version: {bootstub_version}")
    panda.recover()
    cloudlog.info("Done flashing bootloader")

  if panda.bootstub:
    cloudlog.info("Panda still not booting, exiting")
    raise AssertionError

  panda_signature = panda.get_signature()
  if panda_signature != fw_signature:
    cloudlog.info("Version mismatch after flashing, exiting")
    raise AssertionError

  return panda


def get_pandas() -> List[Panda]:
  panda = None
  panda_dfu = None

  cloudlog.info("Connecting to panda")

  # Flash all Pandas in DFU mode
  for p in PandaDFU.list():
    cloudlog.info(f"Panda in DFU mode found, flashing recovery {p}")
    panda_dfu = PandaDFU(p)
    panda_dfu.recover()
    time.sleep(1)

  # Ensure we have at least one panda
  pandas : List[str] = []
  while not pandas:
    pandas = Panda.list()

    if not pandas:
      time.sleep(1)

  cloudlog.info(f"{len(pandas)} panda(s) found, connecting - {pandas}")

  # Flash pandas
  r = []
  for serial in pandas:
    r.append(flash_panda(serial))

  return r


def main() -> None:
  pandas = get_pandas()

  # check health for lost heartbeat
  for panda in pandas:
    health = panda.health()
    if health["heartbeat_lost"]:
      Params().put_bool("PandaHeartbeatLost", True)
      cloudlog.event("heartbeat lost", deviceState=health, serial=panda._serial)

    cloudlog.info(f"Resetting panda {panda._serial}")
    panda.reset()

  if len(pandas) == 1 or not TICI:
    peripheral_panda = pandas[0]
    panda = pandas[0]
  else:
    peripheral_panda = [p for p in pandas if p.get_type() in PERIPHERAL_TYPES][0] # TODO: add error handling if not found
    panda = [p for p in pandas if p._serial != peripheral_panda._serial][0]

  os.chdir(os.path.join(BASEDIR, "selfdrive/boardd"))
  os.execvp("./boardd", ["./boardd", peripheral_panda._serial, panda._serial])


if __name__ == "__main__":
  main()
