import unittest

from extra.optimization.helpers import load_worlds

class TestKernelDataset(unittest.TestCase):
  def test_load_worlds_filters(self):
    all_kernels = load_worlds(filter_reduce=False, filter_noimage=False, filter_novariable=False)

    reduce_kernels = load_worlds(filter_reduce=True, filter_noimage=False, filter_novariable=False)
    self.assertGreater(len(all_kernels), len(reduce_kernels))

    image_kernels = load_worlds(filter_reduce=False, filter_noimage=True, filter_novariable=False)
    self.assertGreater(len(all_kernels), len(image_kernels))

    variable_kernels = load_worlds(filter_reduce=False, filter_noimage=False, filter_novariable=True)
    self.assertGreater(len(all_kernels), len(variable_kernels))

if __name__ == '__main__':
  unittest.main()