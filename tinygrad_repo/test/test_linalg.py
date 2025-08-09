import numpy as np
import unittest
from tinygrad import Tensor
from typing import List
import functools

def orthogonality_helper(A:Tensor,tolerance=1.0e-5):
  b_shape,m = A.shape[0:-2],A.shape[-2]  #outer dimension should be the dim along orthogonality
  A_identity = (Tensor.eye(m).reshape((1,) * len(b_shape)+(m,m)).expand(b_shape+(m,m)))
  np.testing.assert_allclose((A @ A.transpose(-2,-1)).numpy(),A_identity.numpy(),atol=tolerance,rtol=tolerance)

def reconstruction_helper(A:List[Tensor],B:Tensor, tolerance=1.0e-5):
  reconstructed_tensor = functools.reduce(Tensor.matmul, A)
  np.testing.assert_allclose(reconstructed_tensor.numpy(),B.numpy(),atol=tolerance,rtol=tolerance)

class TestLinAlg(unittest.TestCase):

  def test_svd_general(self):
    sizes = [(2,2),(5,3),(3,5),(3,4,4),(2,2,2,2,3)]
    for size in sizes:
      a = Tensor.randn(size).realize()
      U,S,V = Tensor.svd(a)
      b_shape,m,n = size[0:-2],size[-2],size[-1]
      k = min(m,n)
      s_diag = (S.unsqueeze(-2) * Tensor.eye(k).reshape((1,) * len(b_shape) + (k,k)))
      s_diag = s_diag.expand(b_shape + (k,k)).pad(tuple([(0,0) for _ in range(len(size)-2)] + [(0,m-k), (0,n-k)]))
      orthogonality_helper(U)
      orthogonality_helper(V)
      reconstruction_helper([U,s_diag,V],a)

  def test_svd_nonfull(self):
    sizes = [(2,2),(5,3),(3,5),(2,2,2,2,3)]
    for size in sizes:
      a = Tensor.randn(size).realize()
      U,S,V = Tensor.svd(a,full_matrices=False)
      b_shape,m,n = size[0:-2],size[-2],size[-1]
      k = min(m,n)
      s_diag = (S.unsqueeze(-2) * Tensor.eye(k).reshape((1,) * len(b_shape) + (k,k)).expand(b_shape + (k,k)))
      #reduced U,V is only orthogonal along smaller dim
      if (m < n): orthogonality_helper(U),orthogonality_helper(V)
      else: orthogonality_helper(U.transpose(-2,-1)),orthogonality_helper(V.transpose(-2,-1))
      reconstruction_helper([U,s_diag,V],a)

  @unittest.skip("very big. recommend wrapping with TinyJit around inner function")
  def test_svd_large(self):
    size = (1024,1024)
    a = Tensor.randn(size).realize()
    U,S,V = Tensor.svd(a)
    b_shape,m,n = size[0:-2],size[-2],size[-1]
    k = min(m,n)
    s_diag = (S.unsqueeze(-2) * Tensor.eye(k).reshape((1,) * len(b_shape) + (k,k)))
    s_diag = s_diag.expand(b_shape + (k,k)).pad(tuple([(0,0) for _ in range(len(size)-2)] + [(0,m-k), (0,n-k)]))
    orthogonality_helper(U,tolerance=1.0e-3)
    orthogonality_helper(V,tolerance=1.0e-3)
    reconstruction_helper([U,s_diag,V],a,tolerance=1.0e-3)

  def test_qr_general(self):
    sizes = [(3,3),(3,6),(6,3),(2,2,2,2,2)]
    for size in sizes:
      a = Tensor.randn(size).realize()
      Q,R = Tensor.qr(a)
      orthogonality_helper(Q)
      reconstruction_helper([Q,R],a)

if __name__ == "__main__":
  unittest.main()