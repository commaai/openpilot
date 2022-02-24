#!/usr/bin/env python3

import sys
import time
import requests
import json
import io

def flash_release(path=None, st_serial=None):
  from panda import Panda, PandaDFU
  from zipfile import ZipFile

  def status(x):
    print("\033[1;32;40m" + x + "\033[00m")

  if st_serial is not None:
    # look for Panda
    panda_list = Panda.list()
    if len(panda_list) == 0:
      raise Exception("panda not found, make sure it's connected and your user can access it")
    elif len(panda_list) > 1:
      raise Exception("Please only connect one panda")
    st_serial = panda_list[0]
    print("Using panda with serial %s" % st_serial)

  if path is None:
    print("Fetching latest firmware from github.com/commaai/panda-artifacts")
    r = requests.get("https://raw.githubusercontent.com/commaai/panda-artifacts/master/latest.json")
    url = json.loads(r.text)['url']
    r = requests.get(url)
    print("Fetching firmware from %s" % url)
    path = io.BytesIO(r.content)

  zf = ZipFile(path)
  zf.printdir()

  version = zf.read("version").decode().strip()
  status("0. Preparing to flash " + str(version))

  code_bootstub = zf.read("bootstub.panda.bin")
  code_panda = zf.read("panda.bin")

  # enter DFU mode
  status("1. Entering DFU mode")
  panda = Panda(st_serial)
  panda.reset(enter_bootstub=True)
  panda.reset(enter_bootloader=True)
  time.sleep(1)

  # program bootstub
  status("2. Programming bootstub")
  dfu = PandaDFU(PandaDFU.st_serial_to_dfu_serial(st_serial))
  dfu.program_bootstub(code_bootstub)
  time.sleep(1)

  # flash main code
  status("3. Flashing main code")
  panda = Panda(st_serial)
  panda.flash(code=code_panda)
  panda.close()

  # check for connection
  status("4. Verifying version")
  panda = Panda(st_serial)
  my_version = panda.get_version()
  print("dongle id: %s" % panda.get_serial()[0])
  print(my_version, "should be", version)
  assert(str(version) == str(my_version))

  # done!
  status("6. Success!")

if __name__ == "__main__":
  flash_release(*sys.argv[1:])
