#!/usr/bin/env python2
# simple boardd wrapper that updates the panda first
import os
import time

from selfdrive.swaglog import cloudlog
from panda import Panda, PandaDFU, BASEDIR


def update_panda():
  with open(os.path.join(BASEDIR, "VERSION")) as f:
    repo_version = f.read()
  repo_version += "-EON" if os.path.isfile('/EON') else "-DEV"

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

    print "waiting for board..."
    time.sleep(1)

  current_version = "bootstub" if panda.bootstub else str(panda.get_version())
  cloudlog.info("Panda connected, version: %s, expected %s" % (current_version, repo_version))

  if panda.bootstub or not current_version.startswith(repo_version):
    cloudlog.info("Panda firmware out of date, update required")

    signed_fn = os.path.join(BASEDIR, "board", "obj", "panda.bin.signed")
    if os.path.exists(signed_fn):
      cloudlog.info("Flashing signed firmware")
      panda.flash(fn=signed_fn)
    else:
      cloudlog.info("Building and flashing unsigned firmware")
      panda.flash()

    cloudlog.info("Done flashing")

  if panda.bootstub:
    cloudlog.info("Flashed firmware not booting, flashing development bootloader")
    panda.recover()
    cloudlog.info("Done flashing bootloader")

  if panda.bootstub:
    cloudlog.info("Panda still not booting, exiting")
    raise AssertionError

  version = str(panda.get_version())
  if not version.startswith(repo_version):
    cloudlog.info("Version mismatch after flashing, exiting")
    raise AssertionError


def main(gctx=None):
  update_panda()

  os.chdir("boardd")
  os.execvp("./boardd", ["./boardd"])

if __name__ == "__main__":
  main()
