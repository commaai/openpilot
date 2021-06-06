#!/usr/bin/env python3
# simple boardd wrapper that updates the panda first
import os
import time

from panda import BASEDIR as PANDA_BASEDIR, Panda, PandaDFU
from common.basedir import BASEDIR
from selfdrive.swaglog import cloudlog

PANDA_FW_FN = os.path.join(PANDA_BASEDIR, "board", "obj", "panda.bin.signed")


def get_expected_signature() -> bytes:
  try:
    return Panda.get_signature_from_firmware(PANDA_FW_FN)
  except Exception:
    cloudlog.exception("Error computing expected signature")
    return b""


def update_panda() -> Panda:
  panda = None
  panda_dfu = None

  cloudlog.info("Connecting to panda")

  while True:
    # break on normal mode Panda
    panda_list = Panda.list()
    if len(panda_list) > 0:
      cloudlog.info("Panda found, connecting")
      panda = Panda(panda_list[0])
      break

    # flash on DFU mode Panda
    panda_dfu = PandaDFU.list()
    if len(panda_dfu) > 0:
      cloudlog.info("Panda in DFU mode found, flashing recovery")
      panda_dfu = PandaDFU(panda_dfu[0])
      panda_dfu.recover()

    time.sleep(1)

  fw_signature = get_expected_signature()

  try:
    serial = panda.get_serial()[0].decode("utf-8")
  except Exception:
    serial = None

  panda_version = "bootstub" if panda.bootstub else panda.get_version()
  panda_signature = b"" if panda.bootstub else panda.get_signature()
  cloudlog.warning("Panda %s connected, version: %s, signature %s, expected %s" % (
    serial,
    panda_version,
    panda_signature.hex(),
    fw_signature.hex(),
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


def main() -> None:
  panda = update_panda()

  # check health for lost heartbeat
  health = panda.health()
  if health["heartbeat_lost"]:
    cloudlog.event("heartbeat lost", deviceState=health)

  cloudlog.info("Resetting panda")
  panda.reset()

  os.chdir(os.path.join(BASEDIR, "selfdrive/boardd"))
  os.execvp("./boardd", ["./boardd"])


if __name__ == "__main__":
  main()
