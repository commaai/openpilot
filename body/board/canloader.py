#!/usr/bin/env python3
import os
import time
import argparse
import _thread
from panda import Panda, MCU_TYPE_F4  # pylint: disable=import-error
from panda.tests.pedal.canhandle import CanHandle  # pylint: disable=import-error


def heartbeat_thread(p):
  while True:
    try:
      p.send_heartbeat()
      time.sleep(0.5)
    except Exception:
      continue

def flush_panda():
  while(1):
    if len(p.can_recv()) == 0:
      break

def flasher(p, addr, file):
  p.can_send(addr, b"\xce\xfa\xad\xde\x1e\x0b\xb0\x0a", 0)
  time.sleep(0.1)
  print("flashing", file)
  flush_panda()
  code = open(file, "rb").read()
  retries = 3 # How many times to retry on timeout error
  while(retries+1>0):
    try:
      Panda.flash_static(CanHandle(p, 0), code, MCU_TYPE_F4)
    except TimeoutError:
      print("Timeout, trying again...")
      retries -= 1
    else:
      print("Successfully flashed")
      break


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Flash body over can')
  parser.add_argument("board", type=str, nargs='?', help="choose base or knee")
  parser.add_argument("fn", type=str, nargs='?', help="flash file")
  args = parser.parse_args()

  assert args.board in ["base", "knee"]
  assert os.path.isfile(args.fn)

  addr = 0x250 if args.board == "base" else 0x350

  p = Panda()
  _thread.start_new_thread(heartbeat_thread, (p,))
  p.set_safety_mode(Panda.SAFETY_BODY)

  print("Flashing motherboard")
  flasher(p, addr, args.fn)

  print("CAN flashing done")
