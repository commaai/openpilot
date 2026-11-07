import unittest
from tinygrad import dtypes, Tensor, Device
from tinygrad.llm.gguf import ggml_data_to_tensor
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.half in supported_dtypes, "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

if __name__ == '__main__':
  unittest.main()
