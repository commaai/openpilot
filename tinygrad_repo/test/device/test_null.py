import unittest
from tinygrad import dtypes, Device
from tinygrad.device import is_dtype_supported

@unittest.skipUnless(Device.DEFAULT=="NULL", "Don't run when testing non-NULL backends")
class TestNULLSupportsDTypes(unittest.TestCase):
  def test_null_supports_ints_floats_bool(self):
    dts = dtypes.ints + dtypes.floats + (dtypes.bool,)
    not_supported = [dt for dt in dts if not is_dtype_supported(dt, "NULL")]
    self.assertFalse(not_supported, msg=f"expected these dtypes to be supported by NULL: {not_supported}")

if __name__ == "__main__":
  unittest.main()
