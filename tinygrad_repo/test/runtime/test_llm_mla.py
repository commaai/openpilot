import unittest
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.llm.model import TransformerConfig, apply_rope, MLATransformerBlock, precompute_freqs_cis
from tinygrad.llm.hf import e4m3fn_to_fnuz
from test.runtime.test_llm_quantized import fp8_reference

class TestMLA(unittest.TestCase):
  def _make_config(self, **kwargs):
    return TransformerConfig(**{
      "num_blocks": 1, "dim": 64, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 1,
      "norm_eps": 1e-5, "vocab_size": 100, "head_dim": 16, "rope_theta": 10000.0, "rope_dim": 8, "max_context": 32,
      "kv_lora_rank": 16, "v_head_dim": 8,
    } | kwargs)

  def test_mla_attention_matches_naive(self):
    for interleave in (False, True):
      with self.subTest(interleave=interleave): self._test_mla_attention(interleave)

  def _test_mla_attention(self, interleave, fnuz=None):
    config = self._make_config(max_context=16, rope_interleave=interleave, rope_dim=4)

    block = MLATransformerBlock(config)
    c = config
    B, T = 1, 4
    q_nope_head_dim = c.head_dim - c.rope_dim
    block.attn_k_b['weight'] = Tensor.randn(c.n_heads, c.kv_lora_rank, q_nope_head_dim) * .1
    block.attn_v_b['weight'] = Tensor.randn(c.n_heads, c.v_head_dim, c.kv_lora_rank) * .1
    fp8_weights = []
    if fnuz is not None:
      block.kv_b_block_size = (4, 8)
      for projection, scale_shape in ((block.attn_k_b, (4, 2, 3)), (block.attn_v_b, (4, 2, 2))):
        bits = Tensor.randint(projection['weight'].shape, high=73, dtype=dtypes.uint8)
        bits |= Tensor.randint(bits.shape, high=2, dtype=dtypes.uint8) << 7
        w, s = bits.bitcast(dtypes.fp8e4m3), Tensor.rand(scale_shape, dtype=dtypes.float32) * .49 + .01
        fp8_weights.append((bits.numpy(), s.numpy()))
        if fnuz: w, s = e4m3fn_to_fnuz(w, s)
        projection['weight'].replace(w)
        projection['weight_scale_inv'] = s

    x = Tensor.randn(B, T, c.dim)
    x_norm = block.attn_norm(x)

    # --- Our absorbed implementation ---
    q = block.attn_q(x_norm).reshape(B, T, c.n_heads, c.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    freqs = precompute_freqs_cis(c.rope_dim, 16, c.rope_theta)
    q_rope = apply_rope(q_rope, freqs[0:T], interleave)

    kv_a = block.attn_kv_a_mqa(x_norm)
    c_kv = block.attn_kv_a_norm(kv_a[..., :c.kv_lora_rank])
    k_rope = kv_a[..., c.kv_lora_rank:].reshape(B, T, 1, c.rope_dim).transpose(1, 2)
    k_rope = apply_rope(k_rope, freqs[0:T], interleave)

    scale = 1.0 / c.head_dim ** 0.5
    mask = Tensor.full((1, 1, T, T), float("-inf")).triu(1)
    if fnuz is None:
      # --- Naive (non-absorbed): expand K and V, do standard attention ---
      k_nope_naive = c_kv.unsqueeze(1) @ block.attn_k_b["weight"]  # (B, H, T, nope)
      k_naive = k_nope_naive.cat(k_rope.expand(-1, c.n_heads, -1, -1), dim=-1)  # (B, H, T, nope+rope)
      v_naive = c_kv.unsqueeze(1) @ block.attn_v_b["weight"].transpose(-1, -2)  # (B, H, T, v_dim)

      q_naive = q_nope.cat(q_rope, dim=-1)
      scores_naive = (q_naive @ k_naive.transpose(-1, -2)) * scale
      attn_naive = (scores_naive + mask).softmax(-1) @ v_naive  # (B, H, T, v_dim)
      out_naive = block.attn_output(attn_naive.transpose(1, 2).reshape(B, T, -1))

    # --- Absorbed: q_nope @ wk_b^T, then dot with compressed kv ---
    q_nope_abs = (q_nope @ block.attn_k_b["weight"].transpose(-1, -2) if fnuz is None else
                 Tensor(fp8_reference(q_nope.numpy(), *fp8_weights[0], (8, 4), fnuz)))  # (B, H, T, lora)
    q_abs = q_nope_abs.cat(q_rope, dim=-1)  # (B, H, T, lora+rope)
    k_abs = c_kv.reshape(B, 1, T, c.kv_lora_rank).cat(k_rope.reshape(B, 1, T, c.rope_dim), dim=-1)
    scores_abs = (q_abs @ k_abs.transpose(-1, -2)) * scale
    attn_abs = (scores_abs + mask).softmax(-1)
    # attn @ v_compressed @ wv_b
    v_compressed = c_kv.reshape(B, 1, T, c.kv_lora_rank)
    attn_compressed = attn_abs @ v_compressed
    attn_abs_out = (attn_compressed @ block.attn_v_b["weight"].transpose(-1, -2) if fnuz is None else
                   Tensor(fp8_reference(attn_compressed.numpy(), *fp8_weights[1], (4, 8), fnuz)))
    out_abs = block.attn_output(attn_abs_out.transpose(1, 2).reshape(B, T, -1))

    # Compare
    abs_np = out_abs.realize().numpy()
    if fnuz is None:
      np.testing.assert_allclose(out_naive.numpy(), abs_np, atol=1e-4, rtol=1e-4,
        err_msg="Absorbed MLA should match naive MLA")
    block._init_state(x_norm)
    prefill = block._attention(x_norm[:, :3], 0).realize()
    decode = block._attention(x_norm[:, 3:], 3).realize()
    np.testing.assert_allclose(prefill.cat(decode, dim=1).numpy(), abs_np, atol=2e-5, rtol=2e-5)

  def test_fp8_projections(self):
    formats = [fnuz for fnuz, dt in ((False, dtypes.fp8e4m3), (True, dtypes.fp8e4m3fnuz))
               if dt in Device[Device.DEFAULT].renderer.supported_dtypes()]
    if not formats: self.skipTest('requires native FP8')
    for fnuz in formats:
      with self.subTest(fnuz=fnuz): self._test_mla_attention(True, fnuz)
