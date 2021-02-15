#!/usr/bin/env python3
import selfdrive.boardd.boardd as boardd
from panda import *
import unittest
import time
import os

class TestPandaFlashing(unittest.TestCase):
  def test_panda_flashing(self):
    # Ensure we have one normal panda running
    pandas = Panda.list()
    self.assertEqual(len(pandas), 1)
    # Ensure it is running the firmware
    panda1 = Panda(pandas[0])
    self.assertFalse(panda1.bootstub)
    # Move the panda into the DFU mode
    panda1.reset(enter_bootloader=True)
    time.sleep(1)
    # Ensure no normal pandas and one DFU panda
    self.assertEqual(len(PandaDFU.list()), 1)
    self.assertEqual(len(Panda.list()), 0)
    # Run the C++ panda flasher, takes a bit of time
    os.system("export BASEDIR=\"/home/batman/openpilot/\"; ./fix_panda")
    # Now we should have one panda and 0 DFU pandas
    self.assertEqual(len(Panda.list()), 1)
    self.assertEqual(len(PandaDFU.list()), 0)
    print("Test 1 finished")

  def test_release_bootloader(self):
    # Ensure we have one normal panda running
    pandas = Panda.list()
    self.assertEqual(len(pandas), 1)
    # Flash release bootloader and firmware
    os.system("git clone https://github.com/commaai/panda-artifacts.git; cd panda-artifacts; python flash.py")
    # Now we should flash the development firmware, find it doesn't run due to the signature and reflash the dev bootloader and firmware
    os.system("export BASEDIR=\"/home/batman/openpilot/\"; ./fix_panda")
    # Cleanup
    os.system("rm -rf panda-artifacts")
    # In the end we want no DFU pandas and one running panda
    self.assertEqual(len(PandaDFU.list()), 0)
    self.assertEqual(len(Panda.list()), 1)
    panda1 = Panda(pandas[0])
    self.assertFalse(panda1.bootstub)
    print("Test 2 finished")

  def test_signed_firmware(self):
    # We make the same preparation as on previous test
    pandas = Panda.list()
    self.assertEqual(len(pandas), 1)
    os.system("git clone https://github.com/commaai/panda-artifacts.git; cd panda-artifacts; python flash.py")
    os.system("ls; rm -rf panda-artifacts")
    # We load the signed firmware
    os.system("cd ../../../panda/board/obj/; wget \"https://github.com/commaai/openpilot/blob/release2/panda/board/obj/panda.bin.signed?raw=true\" -O panda.bin.signed")
    # Run the panda flasher. It should install the signed firmware
    os.system("export BASEDIR=\"/home/batman/openpilot/\"; ./fix_panda")
    # In the end we want no DFU pandas and one running panda
    self.assertEqual(len(PandaDFU.list()), 0)
    self.assertEqual(len(Panda.list()), 1)
    panda1 = Panda(pandas[0])
    self.assertFalse(panda1.bootstub)
    print("Test 3 finished")

if __name__ == '__main__':
    unittest.main()
