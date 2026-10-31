import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.kernels.amd import gated_delta_prefill, amd_custom_kernels_supported

class TestGatedDeltaNetBlock(unittest.TestCase):
  def test_gated_delta_rectangular_state_and_column_decay(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q, k = (rng.normal(size=(1, 1, 3, 32)).astype(np.float32) for _ in range(2))
    v, beta = rng.normal(size=(1, 1, 3, 4)).astype(np.float32), rng.uniform(size=(1, 1, 3)).astype(np.float32)
    alpha, initial = rng.uniform(0.8, 1, size=(1, 1, 3, 32)).astype(np.float32), rng.normal(size=(1, 1, 4, 32)).astype(np.float32)
    expected_state, expected_out = initial.copy(), np.empty_like(v)
    for t in range(3):
      previous, av = expected_state.copy(), alpha[:, :, t, None, :]
      delta = (v[:, :, t] - (previous*av*k[:, :, t, None, :]).sum(-1)) * beta[:, :, t, None]
      expected_state = previous*av + delta[..., None]*k[:, :, t, None, :]
      expected_out[:, :, t] = (previous*av*q[:, :, t, None, :]).sum(-1) + delta*(q[:, :, t]*k[:, :, t]).sum(-1)
    state = Tensor(initial).contiguous().realize()
    out = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state).realize()
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-4)

if __name__ == "__main__": unittest.main()
