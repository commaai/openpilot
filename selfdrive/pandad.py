#!/usr/bin/env python3
# simple boardd wrapper that updates the panda first
import os
import time

from panda import BASEDIR as PANDA_BASEDIR, Panda, PandaDFU, build_st
from common.basedir import BASEDIR
from common.gpio import gpio_init, gpio_set
from selfdrive.hardware import TICI
from selfdrive.hardware.tici.pins import GPIO_HUB_RST_N, GPIO_STM_BOOT0, GPIO_STM_RST_N
from selfdrive.swaglog import cloudlog

def is_legacy_panda_reset():
  return os.path.isfile("/persist/LEGACY_PANDA_RESET")

def set_panda_power(power=True):
  if not TICI:
    return

  gpio_init(GPIO_STM_RST_N, True)
  gpio_init(GPIO_STM_BOOT0, True)

  gpio_set(GPIO_STM_RST_N, False if is_legacy_panda_reset() else True)
  gpio_set(GPIO_HUB_RST_N, True)

  time.sleep(0.1)

  gpio_set(GPIO_STM_RST_N, power if is_legacy_panda_reset() else (not power))


def get_firmware_fn():
  signed_fn = os.path.join(PANDA_BASEDIR, "board", "obj", "panda.bin.signed")
  if os.path.exists(signed_fn):
    cloudlog.info("Using prebuilt signed firmware")
    return signed_fn
  else:
    cloudlog.info("Building panda firmware")
    fn = "obj/panda.bin"
    build_st(fn, clean=False)
    return os.path.join(PANDA_BASEDIR, "board", fn)


def get_expected_signature(fw_fn=None):
  if fw_fn is None:
    fw_fn = get_firmware_fn()

  try:
    return Panda.get_signature_from_firmware(fw_fn)
  except Exception:
    cloudlog.exception("Error computing expected signature")
    return b""


def update_panda():
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

  fw_fn = get_firmware_fn()
  fw_signature = get_expected_signature(fw_fn)

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
    panda.flash(fw_fn)
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


def main():
  set_panda_power()
  update_panda()

  os.chdir(os.path.join(BASEDIR, "selfdrive/boardd"))
  os.execvp("./boardd", ["./boardd"])


if __name__ == "__main__":
  main()
