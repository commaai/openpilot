import os
import time
import subprocess
import unittest
from panda import Panda, BASEDIR
from panda_jungle import PandaJungle  # pylint: disable=import-error
from panda.tests.pedal.canhandle import CanHandle


JUNGLE_SERIAL = os.getenv("PEDAL_JUNGLE")
PEDAL_SERIAL = 'none'
PEDAL_BUS = 1

class TestPedal(unittest.TestCase):
  
  def setUp(self):
    self.jungle = PandaJungle(JUNGLE_SERIAL)
    self.jungle.set_panda_power(True)
    self.jungle.set_ignition(False)

  def tearDown(self):
    self.jungle.close()

  def _flash_over_can(self, bus, fw_file):
    print(f"Flashing {fw_file}")
    while len(self.jungle.can_recv()) != 0:
      continue
    self.jungle.can_send(0x200, b"\xce\xfa\xad\xde\x1e\x0b\xb0\x0a", bus)

    time.sleep(0.1)
    with open(fw_file, "rb") as code:
      PandaJungle.flash_static(CanHandle(self.jungle, bus), code.read())

  def _listen_can_frames(self):
    self.jungle.can_clear(0xFFFF)
    msgs = 0
    for _ in range(10):
      incoming = self.jungle.can_recv()
      for message in incoming:
        address, _, _, bus = message
        if address == 0x201 and bus == PEDAL_BUS:
          msgs += 1
      time.sleep(0.1)
    return msgs

  def test_usb_fw(self):
    subprocess.check_output(f"cd {BASEDIR} && PEDAL=1 PEDAL_USB=1 scons", shell=True)
    self._flash_over_can(PEDAL_BUS, f"{BASEDIR}board/obj/pedal_usb.bin.signed")
    time.sleep(2)
    p = Panda(PEDAL_SERIAL)
    self.assertTrue(p.is_pedal())
    p.close()
    self.assertTrue(self._listen_can_frames() > 40)

  def test_nonusb_fw(self):
    subprocess.check_output(f"cd {BASEDIR} && PEDAL=1 scons", shell=True)
    self._flash_over_can(PEDAL_BUS, f"{BASEDIR}board/obj/pedal.bin.signed")
    time.sleep(2)
    self.assertTrue(PEDAL_SERIAL not in Panda.list())
    self.assertTrue(self._listen_can_frames() > 40)


if __name__ == '__main__':
  unittest.main()
