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
    command = "export BASEDIR=\"/home/batman/openpilot/\"; ./fix_panda"
    os.system(command)
    # Now we should have one panda and 0 DFU pandas
    self.assertEqual(len(Panda.list()), 1)
    self.assertEqual(len(PandaDFU.list()), 0)


    print("Done testing")
if __name__ == '__main__':
    unittest.main()
