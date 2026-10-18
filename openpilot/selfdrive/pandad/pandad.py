#!/usr/bin/env python3
# simple pandad wrapper that updates the panda first
import threading
import time

from panda import Panda, PandaDFU
from openpilot.common.params import Params
from openpilot.common.hardware import HARDWARE
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.pandad.runner import PandaRunner


def wait_for_panda(timeout: float) -> str | None:
  """Serial of the internal (SPI) panda, or None if it didn't show up in time. USB pandas aren't supported."""
  t_start = time.monotonic()
  while True:
    serials = Panda.spi_list()
    if serials:
      return serials[0]
    if time.monotonic() - t_start > timeout:
      return None
    time.sleep(0.1)


def connect(serial: str) -> Panda:
  # disable_checks=False: pandad is the heartbeat source, it must never send set_heartbeat_disabled
  p = Panda(serial, cli=False, disable_checks=False)
  assert p.spi, "only the internal SPI panda is supported"
  return p


def panda_ready() -> bool:
  """Is the panda up and running the firmware we ship?"""
  try:
    serial = wait_for_panda(timeout=5)
    if serial is None:
      cloudlog.warning("no panda found")
      return False
    with connect(serial) as p:
      if p.bootstub:
        cloudlog.warning(f"panda stuck in bootstub, version {p.get_version()}")
        return False
      if not p.up_to_date():
        cloudlog.warning(f"panda firmware out of date, version {p.get_version()}")
        return False
      return True
  except Exception:
    # e.g. SPI protocol mismatch with old firmware
    cloudlog.exception("pandad.panda_check_failed")
    return False


def flash_panda() -> None:
  """Reflash bootstub + app from the ST bootloader. Works from any state the panda can be in."""
  cloudlog.warning("flashing panda")
  HARDWARE.recover_internal_panda()
  assert Panda.wait_for_dfu(None, timeout=5), "panda didn't enter DFU"
  with PandaDFU(None) as dfu:
    dfu.recover()

  serial = wait_for_panda(timeout=5)
  assert serial is not None, "panda didn't boot into bootstub"
  with connect(serial) as p:
    p.flash()
  cloudlog.warning("done flashing panda")


def main() -> None:
  # check health for lost heartbeat
  try:
    for s in Panda.spi_list():
      with connect(s) as p:
        health = p.health()
        if health["heartbeat_lost"]:
          Params().put_bool("PandaHeartbeatLost", True, block=True)
          cloudlog.event("heartbeat lost", deviceState=health)
  except Exception:
    cloudlog.exception("pandad.uncaught_exception")

  # signal pandad to close the relay and exit
  evt = threading.Event()
  count = 0
  while True:
    panda: Panda | None = None
    try:
      cloudlog.event("pandad.flash_and_connect", count=count)
      count += 1

      HARDWARE.reset_internal_panda()
      if not panda_ready():
        flash_panda()
        assert panda_ready(), "panda not ready after flashing"

      serial = wait_for_panda(timeout=5)
      assert serial is not None, "panda disappeared"
      panda = connect(serial)
      cloudlog.warning(f"connected to panda {serial}")

      # run real pandad. only the 100 Hz loop runs realtime, flashing and recovery busy-poll SPI for seconds
      config_realtime_process(3, 54)
      evt.clear()
      PandaRunner(panda).run(evt)
    except KeyboardInterrupt:
      cloudlog.info("Caught Ctrl+C, exiting")
      break
    except Exception:
      cloudlog.exception("pandad.uncaught_exception")
    finally:
      if panda is not None:
        try:
          panda.close()
        except Exception:
          cloudlog.exception("Error closing panda connection")


if __name__ == "__main__":
  main()
