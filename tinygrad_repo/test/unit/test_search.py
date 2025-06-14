import unittest
from tinygrad.engine.search import get_test_global_size

class TestSearchUtil(unittest.TestCase):
  def test_get_test_global_size(self):
    self.assertEqual(get_test_global_size([256, 256, 256], 65536, {}), ([256, 16, 16], 256.0))
    self.assertEqual(get_test_global_size([65536, 1, 1], 256, {}), ([256, 1, 1], 256.0))
    self.assertEqual(get_test_global_size([77, 1, 1], 16, {}), ([9, 1, 1], 77/9))

if __name__ == "__main__":
  unittest.main()