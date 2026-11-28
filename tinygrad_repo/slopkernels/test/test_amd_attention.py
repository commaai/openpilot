import unittest

import numpy as np
from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.helpers import unwrap
from tinygrad.renderer.cstyle import HIPRenderer

import slopkernels


def _amd_supported(dev:str|tuple[str, ...]|None) -> bool:
  d = dev[0] if isinstance(dev, tuple) else dev
  if d is None or d.split(":")[0] != "AMD": return False
  t = getattr(Device[d], "target", None)
  return t is not None and t[0] in (9, 11, 12) and isinstance(Device[d].renderer, HIPRenderer)


def _gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor) -> Tensor:
  B, H, T, K, V = int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3]), int(v.shape[3])
  ak = K if alpha.ndim == 4 else 1
  d = q.device[0] if isinstance(q.device, tuple) else q.device
  assert d is not None
  t = getattr(Device[d], "target", (11, 0, 0))
  kern = unwrap(slopkernels.get("amd_gated_delta")).specialize(
    batch=B, heads=H, tokens=T, v_tiles=V // 4, k_blks=K // 32, alpha_k=ak, arch=t[0])
  out = UOp.alloc((B, H, T, V), dtypes.float32, device=q.device)
  return Tensor(out.after(kern.call(out, q.contiguous().uop, k.contiguous().uop, v.contiguous().uop, beta.contiguous().uop,
                                    alpha.contiguous().reshape(B, H, T, ak).uop, state.uop, UOp.const(1))))


class TestGatedDeltaNetBlock(unittest.TestCase):
  def test_gated_delta_rectangular_state_and_column_decay(self):
    if not _amd_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q, k = (rng.normal(size=(1, 1, 3, 32)).astype(np.float32) for _ in range(2))
    v, beta = rng.normal(size=(1, 1, 3, 4)).astype(np.float32), rng.uniform(size=(1, 1, 3)).astype(np.float32)
    alpha, initial = rng.uniform(0.8, 1, size=(1, 1, 3, 32)).astype(np.float32), rng.normal(size=(1, 1, 4, 32)).astype(np.float32)
    expected_state, expected_out = initial.copy(), np.empty_like(v)
    for t in range(3):
      previous, av = expected_state.copy(), alpha[:, :, t, None, :]
      delta = (v[:, :, t] - (previous * av * k[:, :, t, None, :]).sum(-1)) * beta[:, :, t, None]
      expected_state = previous * av + delta[..., None] * k[:, :, t, None, :]
      expected_out[:, :, t] = (previous * av * q[:, :, t, None, :]).sum(-1) + delta * (q[:, :, t] * k[:, :, t]).sum(-1)
    state = Tensor(initial).contiguous().realize()
    out = _gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state).realize()
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-4)

  def test_gated_delta_subnormal_alpha_no_nan(self):
    if not _amd_supported(Tensor.empty(1).device): self.skipTest("AMD required")
    rng = np.random.default_rng(99)
    B, H, T, D = 1, 2, 32, 128
    q, k = (rng.normal(0, 0.05, (B, H, T, D)).astype(np.float32) for _ in range(2))
    v = rng.normal(0, 0.2, (B, H, T, D)).astype(np.float32)
    beta = rng.uniform(0.05, 0.3, (B, H, T)).astype(np.float32)
    alpha = np.full((B, H, T), 1e-40, dtype=np.float32)  # subnormal alpha (flushed to 0 by AMD FTZ v_log_f32!)
    alpha[:, 0] = rng.uniform(0.5, 0.99, (B, T)).astype(np.float32)
    st_ref, out_ref = np.zeros((B, H, D, D), dtype=np.float32), np.empty((B, H, T, D), dtype=np.float32)
    for t in range(T):
      st_ref *= alpha[:, :, t, None, None]
      delta = (v[:, :, t] - (st_ref * k[:, :, t, None, :]).sum(-1)) * beta[:, :, t, None]
      st_ref += delta[..., None] * k[:, :, t, None, :]
      out_ref[:, :, t] = (st_ref * q[:, :, t, None, :]).sum(-1)
    state = Tensor.zeros(B, H, D, D, dtype=dtypes.float32).contiguous().realize()
    out = _gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state).realize()
    self.assertTrue(np.isfinite(out.numpy()).all() and np.isfinite(state.numpy()).all())
    np.testing.assert_allclose(out.numpy(), out_ref, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(state.numpy(), st_ref, rtol=2e-3, atol=2e-4)


if __name__ == "__main__":
  unittest.main()
