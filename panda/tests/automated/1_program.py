import os
from panda import Panda
from .helpers import panda_type_to_serial, test_white_and_grey, test_all_pandas, panda_connect_and_init

@test_all_pandas
@panda_connect_and_init
def test_recover(p):
  assert p.recover(timeout=30)

@test_all_pandas
@panda_connect_and_init
def test_flash(p):
  p.flash()
