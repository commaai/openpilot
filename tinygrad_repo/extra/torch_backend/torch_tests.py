import sys
import unittest
import torch
import extra.torch_backend.backend

from torch.testing._internal.common_utils import TestCase, is_privateuse1_backend_available
assert is_privateuse1_backend_available() and torch._C._get_privateuse1_backend_name() == "tiny"
from torch.testing._internal.common_device_type import ops, onlyOn, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import unary_ufuncs, binary_ufuncs, reduction_ops, shape_funcs

def to_cpu(arg): return arg.to(device="cpu") if isinstance(arg, torch.Tensor) else arg
def filter_funcs(ufuncs): return [x for x in ufuncs if not x.name.startswith("_refs") and not x.name.startswith("special")]

class TestTinyBackend(TestCase):
  def _test(self, device, dtype, op):
    samples = op.sample_inputs(device, dtype)
    for sample in samples:
      tiny_results = op(sample.input, *sample.args, **sample.kwargs)
      tiny_results = sample.output_process_fn_grad(tiny_results)

      cpu_sample = sample.transform(to_cpu)
      cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)
      cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

      self.assertEqual(tiny_results, cpu_results, atol=1e-3, rtol=1e-3)

  @ops(filter_funcs(unary_ufuncs), allowed_dtypes=[torch.float])
  def test_unary(self, device, dtype, op): self._test(device, dtype, op)

  @ops(filter_funcs(binary_ufuncs), allowed_dtypes=[torch.float])
  def test_binary(self, device, dtype, op): self._test(device, dtype, op)

  @ops(filter_funcs(reduction_ops), allowed_dtypes=[torch.float])
  def test_reduction(self, device, dtype, op): self._test(device, dtype, op)

  # none of these pass
  #@ops(shape_funcs)
  #def test_shape(self, device, dtype, op): self._test(device, dtype, op)

instantiate_device_type_tests(TestTinyBackend, globals(), only_for=["tiny"])

if __name__ == "__main__":
  unittest.main(verbosity=2)
