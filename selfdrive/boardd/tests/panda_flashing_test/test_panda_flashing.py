#!/usr/bin/env python3
from panda import Panda, PandaDFU, build_st
import unittest
import time
import os
import requests
import zipfile
from io import BytesIO
import subprocess

from panda import BASEDIR as PANDA_BASEDIR

SIGNED_FW_FN = os.path.join(os.path.abspath(PANDA_BASEDIR), "board", "obj", "panda.bin.signed")
SIGNED_FIRMWARE_URL = "https://github.com/commaai/openpilot/blob/release2/panda/board/obj/panda.bin.signed?raw=true"


def build_dev_fw():
  fn = "obj/panda.bin"
  build_st(fn, clean=False)
  return os.path.abspath(os.path.join(PANDA_BASEDIR, "board", fn))


def get_expected_signature(fn):
  try:
    return Panda.get_signature_from_firmware(fn)
  except Exception:
    return b""


def download_file(url):
  r = requests.get(url, allow_redirects=True)
  return r.content


class TestPandaFlashing(unittest.TestCase):
  def ensure_dfu(self):
    """Ensures the connected panda is running in DFU mode"""
    if len(PandaDFU.list()) == 1:
      return

    pandas = Panda.list()
    self.assertEqual(len(pandas), 1)

    # Move to DFU mode
    panda1 = Panda(pandas[0])
    panda1.reset(enter_bootstub=True)
    panda1.reset(enter_bootloader=True)
    panda1.close()

    # TODO: check faster, but still have max timeout
    time.sleep(5)

    # Ensure no normal pandas and one DFU panda
    self.assertEqual(len(PandaDFU.list()), 1)
    self.assertEqual(len(Panda.list()), 0)

  def check_panda_running(self, expected_signature=None):
    self.assertEqual(len(PandaDFU.list()), 0)
    self.assertEqual(len(Panda.list()), 1)

    panda = Panda(Panda.list()[0])
    self.assertFalse(panda.bootstub)

    # TODO: check signature
    # self.assertNotEqual(panda.get_signature(), comma_sig)
    panda.close()

  def flash_release_bootloader_and_fw(self):
    self.ensure_dfu()

    fp = BytesIO(download_file("https://github.com/commaai/panda-artifacts/blob/master/panda-v1.7.3-DEV-d034f3e9-RELEASE.zip?raw=true"))

    with zipfile.ZipFile(fp) as zip_file:
      # Flash bootstub
      bootstub_code = zip_file.open('bootstub.panda.bin').read()
      PandaDFU(None).program_bootstub(bootstub_code)

      # Wait for panda to come back online
      while not Panda.list():
        print("Waiting for Panda")
        time.sleep(0.5)

      # Flash firmware
      firmware_code = zip_file.open('panda.bin').read()
      Panda().flash(code=firmware_code)

  def run_flasher(self):
    subprocess.check_call("./flash_panda")

  def setUp(self):
    try:
      os.unlink(SIGNED_FW_FN)
    except FileNotFoundError:
      pass

    self.flash_release_bootloader_and_fw()

  def test_flash_from_dfu(self):
    self.ensure_dfu()

    self.run_flasher()
    self.check_panda_running()

  def test_dev_firmware(self):
    self.run_flasher()

    # TODO: check for development signature
    self.check_panda_running()

  def test_signed_firmware(self):
    with open(SIGNED_FW_FN, 'wb') as f:
      f.write(download_file(SIGNED_FIRMWARE_URL))

    try:
      os.system("./flash_panda")
    finally:
      os.unlink(SIGNED_FW_FN)

    # TODO: check for signed signature
    self.check_panda_running()


if __name__ == '__main__':
    unittest.main()
