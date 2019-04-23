import os
from panda import Panda
from helpers import panda_color_to_serial, test_white_and_grey

@test_white_and_grey
@panda_color_to_serial
def test_recover(serial=None):
  p = Panda(serial=serial)
  assert p.recover(timeout=30)

@test_white_and_grey
@panda_color_to_serial
def test_flash(serial=None):
  p = Panda(serial=serial)
  p.flash()
