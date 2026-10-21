#!/usr/bin/env python3
import argparse
import os
import subprocess

from panda import Panda

BODY_DIR = os.path.dirname(os.path.realpath(__file__))
BOARD_DIR = os.path.abspath(os.path.join(BODY_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BOARD_DIR, ".."))
DEFAULT_FIRMWARE = os.path.join(BOARD_DIR, "obj", "body_h7.bin.signed")


def build_body() -> None:
  subprocess.check_call(
    f"scons -C {REPO_ROOT} -j$(nproc) board/obj/body_h7.bin.signed",
    shell=True,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("firmware", nargs="?", help="Optional path to firmware binary to flash")
  parser.add_argument("--all", action="store_true", help="Flash all Panda devices")
  parser.add_argument(
    "--wait-usb",
    action="store_true",
    help="Wait for the panda to reconnect over USB after flashing (defaults to skipping reconnect).",
  )
  args = parser.parse_args()

  firmware_path = os.path.abspath(args.firmware) if args.firmware is not None else DEFAULT_FIRMWARE

  build_body()

  if not os.path.isfile(firmware_path):
    parser.error(f"firmware file not found: {firmware_path}")

  if args.all:
    serials = Panda.list()
    print(f"found {len(serials)} panda(s) - {serials}")
  else:
    serials = [None]

  for s in serials:
    with Panda(serial=s) as p:
      print("flashing", p.get_usb_serial())
      p.flash(firmware_path, reconnect=args.wait_usb)
  exit(1 if len(serials) == 0 else 0)
