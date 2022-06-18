from nose.tools import assert_equal

from panda import Panda, DEFAULT_FW_FN, DEFAULT_H7_FW_FN, MCU_TYPE_H7
from .helpers import reset_pandas, test_all_pandas, panda_connect_and_init


# Reset the pandas before flashing them
def aaaa_reset_before_tests():
  reset_pandas()


@test_all_pandas
@panda_connect_and_init(clear_can=False)
def test_a_recover(p):
  assert p.recover(timeout=30)


@test_all_pandas
@panda_connect_and_init(clear_can=False)
def test_b_flash(p):
  p.flash()


@test_all_pandas
@panda_connect_and_init(clear_can=False)
def test_get_signature(p):
  fn = DEFAULT_H7_FW_FN if p.get_mcu_type() == MCU_TYPE_H7  else DEFAULT_FW_FN

  firmware_sig = Panda.get_signature_from_firmware(fn)
  panda_sig = p.get_signature()

  assert_equal(panda_sig, firmware_sig)
