import unittest
from typing import List, Callable
import numpy as np
import torch
from tinygrad import Tensor, dtypes
from test.helpers import TensorTestCase

class TestOps(TensorTestCase):
  def test_negative_dims(self):
    creation_methods: List[Callable[..., Tensor]] = [
      Tensor.empty,
      Tensor.rand,
      Tensor.zeros,
      Tensor.ones,
      Tensor.randn,
      Tensor.randint,
      Tensor.normal,
      Tensor.uniform,
      Tensor.scaled_uniform,
      Tensor.glorot_uniform
    ]

    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 2)
      with self.assertRaises(ValueError): method((2, -3))
      with self.assertRaises(ValueError): method((2, -3, 0))

  def test_negative_dims_full(self):
    with self.assertRaises(ValueError): Tensor.full((-3,), 2)
    with self.assertRaises(ValueError): Tensor.full((2, -3), 4)
    with self.assertRaises(ValueError): Tensor.full((2, -3, 0), 4)

  def test_negative_dims_eye(self):
    with self.assertRaises(ValueError): Tensor.eye(-3, 3)
    with self.assertRaises(ValueError): Tensor.eye(3, -3)
    with self.assertRaises(ValueError): Tensor.eye(-3, -3)

  def test_negative_dims_kaiming(self):
    creation_methods = [Tensor.kaiming_uniform, Tensor.kaiming_normal]
    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 3)
      with self.assertRaises(ValueError): method((-3, 3), 3)
      with self.assertRaises(ValueError): method((-3, -3), 3)

  def test_prod_dtype_arg(self):
    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).prod(dtype="")

  def test_slice_errors(self):
    a = Tensor.ones(4, 3)
    b = Tensor(2)
    with self.assertRaisesRegex(IndexError, "too many"): a[1, 77, 77, 77] # IndexError: (finds too many indices before the out of bounds)
    with self.assertRaisesRegex(IndexError, "out of bounds"): a[1, 3] # IndexError: (out of bounds).
    with self.assertRaisesRegex(IndexError, "out of bounds"): a[1, -4]
    with self.assertRaisesRegex(IndexError, "single ellipsis"): a[..., ...] # IndexError: only single ellipsis
    with self.assertRaises(ValueError): a[::0, 1] # no 0 strides
    with self.assertRaises(TypeError): a[:Tensor([3]), 1] # Tensor can't be used as a slice parameter
    with self.assertRaises(IndexError): b[:] # slice cannot be applied to a 0-dim tensor

  def test_einsum_shape_check(self):
    self.helper_test_exception([(3,8,10,5), (11,5,13,16,8)], lambda a, b: torch.einsum('pqrs,tuqvr->pstuv', [a, b]),
                lambda a, b: Tensor.einsum('pqrs,tuqvr->pstuv', [a, b]), expected=RuntimeError)
    # repeated letter with different sizes
    self.helper_test_exception([(3,4)], lambda a: torch.einsum('ii->i', a), lambda a: Tensor.einsum('ii->i', a), expected=RuntimeError)
    # number of letters doesn't match ndim
    self.helper_test_exception([(3,4,5)], lambda a: torch.einsum('ij->ij', a), lambda a: Tensor.einsum('ij->ij', a),
                expected=(ValueError, RuntimeError))
    # output letter not in the inputs
    self.helper_test_exception([(3,4)], lambda a: torch.einsum('ij->ik', a), lambda a: Tensor.einsum('ij->ik', a),
                expected=(ValueError, RuntimeError))

  def test_einsum_arity_check1(self):
    self.helper_test_exception([(10,15), (15,20), (20,10)], lambda a, b, c: torch.einsum('ij,jk->ij', [a, b, c]),
                lambda a, b, c: Tensor.einsum('ij,jk->ij', [a, b, c]), expected=(ValueError, RuntimeError))

  def test_einsum_arity_check2(self):
    self.helper_test_exception([(10,10)], lambda a: torch.einsum('ij,jk->ij', a),
                lambda a: Tensor.einsum('ij,jk->ij', a), expected=(ValueError, RuntimeError))

  def test_conv2d_errors(self):
    # kernel size cannot be larger than input size
    self.helper_test_exception([(1,1,6,7), (6,1,3,3)],
                               lambda x,w:torch.nn.functional.conv2d(x,w,dilation=3),
                               lambda x,w: Tensor.conv2d(x,w,dilation=3), expected=(RuntimeError, AssertionError))
    # regression test for https://github.com/tinygrad/tinygrad/pull/7549/
    self.helper_test_exception([(2,16,2,2), (32,16,3,3)], lambda x,w:torch.nn.functional.conv2d(x,w), lambda x,w: Tensor.conv2d(x,w),
                               expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,16,2,2), (32,16,3,3)], lambda x,w:torch.nn.functional.conv2d(x,w,padding=(1,1,1)),
                               lambda x,w: Tensor.conv2d(x,w,padding=(1,1,1)), expected=(RuntimeError, ValueError))

  def test_scatter_reduce_errors(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32)
    # invalid reduce arg
    self.helper_test_exception([(4,5,6), (4,5,6)],
      lambda x,src: x.scatter_reduce(dim=0, index=b, src=src, reduce="INVALID"),
      lambda x,src: x.scatter_reduce(dim=0, index=a, src=src, reduce="INVALID"),
      RuntimeError)
    # dtype mismatch
    self.helper_test_exception([(4,5,6), (4,5,6)],
      lambda x,src: x.half().scatter_reduce(dim=0, index=b, src=src, reduce="sum"),
      lambda x,src: x.half().scatter_reduce(dim=0, index=a, src=src, reduce="sum"),
      RuntimeError)

  def test_scaled_dot_product_attention_gqa_errors(self):
    self.helper_test_exception([(32,31,16,64), (32,8,16,64), (32,8,16,64)],
      lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z),
      lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z,enable_gqa=True),
      expected=(AssertionError, RuntimeError, ValueError, IndexError))

  def test_scatter_no_reduce_tensor_src(self):
    with self.assertRaises(TypeError):
      Tensor.ones(4).scatter(dim=1, index=Tensor([0]), src=Tensor.ones(4), reduce="add")

if __name__ == '__main__':
  unittest.main()
