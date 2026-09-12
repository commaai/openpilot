import unittest
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.model import (
  GatedDeltaNetBlock, SSMConfig, TransformerBlock, TransformerConfig,
  apply_rope as apply_rope_new, precompute_freqs_cis, pairwise_topk,
)
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, amd_custom_kernels_supported
from tinygrad.llm.gguf import ggml_data_to_tensor

def apply_rope(x:Tensor, start_pos:int):
  B, H, T, Hd = x.shape
  precompute_freqs_cis.cache_clear()
  freqs_cis = precompute_freqs_cis(Hd, start_pos+T)[start_pos:start_pos+T]
  return apply_rope_new(x, freqs_cis)

class TestLinear(unittest.TestCase):
  def test_recovers_packed_ggml_weight(self):
    for ggml_type,packed_size,words in ((13, 176, 44), (14, 210, 53), (23, 136, 34)):
      packed = Tensor.empty(packed_size+4, dtype=dtypes.uint8, device="CPU")[4:]
      decoded = ggml_data_to_tensor(packed, 256, ggml_type).reshape(1, 256)
      linear = Linear(256, 1, bias=False)
      linear.set_quantized(decoded)
      self.assertEqual((linear.ggml_type, linear.weight.numel()), (ggml_type, words))

class TestAttention(unittest.TestCase):
  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

  def test_partial_rope_in_attention(self):
    dim, rope_dim, seqlen = 8, 4, 3
    config = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=16, n_heads=1, n_kv_heads=1,
                               norm_eps=1e-5, vocab_size=32, head_dim=dim, rope_theta=10000.0,
                               rope_dim=rope_dim, v_head_dim=dim, max_context=8)
    block = TransformerBlock(config)

    x = Tensor.randn(1, seqlen, dim, dtype=dtypes.float32)
    x_norm = block.attn_norm(x)
    k = block.attn_k(x_norm).reshape(1, seqlen, 1, dim).transpose(1, 2)

    precompute_freqs_cis.cache_clear()
    block.cache_kv = Tensor.empty(2, 1, 1, config.max_context, max(dim, config.v_head_dim), device=x.device)
    block.freqs_cis = precompute_freqs_cis(rope_dim, config.max_context, config.rope_theta)
    block._attention(x_norm, 0).realize()

    expected = apply_rope_new(k[..., :rope_dim], block.freqs_cis[:seqlen]).cat(k[..., rope_dim:], dim=-1)
    np.testing.assert_allclose(block.cache_kv[0, :, :, :seqlen, :].numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)

class TestGatedDeltaNetBlock(unittest.TestCase):
  def test_gated_delta_rectangular_state_and_row_decay(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q, k = (rng.normal(size=(1, 1, 3, 32)).astype(np.float32) for _ in range(2))
    v, beta = rng.normal(size=(1, 1, 3, 4)).astype(np.float32), rng.uniform(size=(1, 1, 3)).astype(np.float32)
    alpha, initial = rng.uniform(0.8, 1, size=(1, 1, 3, 4)).astype(np.float32), rng.normal(size=(1, 1, 4, 32)).astype(np.float32)
    expected_state, expected_out = initial.copy(), np.empty_like(v)
    for t in range(3):
      previous, av = expected_state.copy(), alpha[:, :, t, :, None]
      delta = (v[:, :, t] - (previous*k[:, :, t, None]).sum(-1)*alpha[:, :, t]) * beta[:, :, t, None]
      expected_state = previous*av + delta[..., None]*k[:, :, t, None, :]
      expected_out[:, :, t] = (previous*q[:, :, t, None]).sum(-1)*alpha[:, :, t] + delta*(q[:, :, t]*k[:, :, t]).sum(-1)
    state = Tensor(initial).contiguous().realize()
    out = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state).realize()
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-4)

  def _tensor_linspace(self, start:float, stop:float, shape:tuple[int, ...]) -> Tensor:
    return Tensor.linspace(start, stop, int(np.prod(shape)), dtype=dtypes.float32).reshape(*shape)

  def _make_config(self, **kwargs):
    return TransformerConfig(**({"num_blocks":1, "dim":8, "hidden_dim":16, "n_heads":1, "n_kv_heads":1,
                                 "norm_eps":1e-5, "vocab_size":32, "head_dim":8, "rope_theta":10000.0,
                                 "rope_dim":8, "v_head_dim":8, "max_context":4, "ssm_layers":(True,),
                                 "ssm":SSMConfig(conv_kernel=2, state_size=4, group_count=1, time_step_rank=1, inner_size=4)} | kwargs))

  def _make_block(self, config:TransformerConfig) -> GatedDeltaNetBlock:
    block = GatedDeltaNetBlock(config, config.ssm)
    block.attn_norm.weight = self._tensor_linspace(0.8, 1.2, (config.dim,))
    block.attn_qkv.weight = self._tensor_linspace(-0.15, 0.2, (block.conv_channels, config.dim))
    block.attn_gate.weight = self._tensor_linspace(-0.1, 0.15, (config.ssm.inner_size, config.dim))
    block.ssm_alpha.weight = self._tensor_linspace(-0.08, 0.12, (block.num_v_heads, config.dim))
    block.ssm_beta.weight = self._tensor_linspace(-0.12, 0.07, (block.num_v_heads, config.dim))
    block.ssm_conv1d["weight"] = self._tensor_linspace(-0.05, 0.05, (block.conv_channels, block.ssm_conv_kernel))
    block.ssm_dt["bias"] = self._tensor_linspace(-0.1, 0.1, (block.num_v_heads,))
    block.ssm_a = self._tensor_linspace(-0.1, -0.05, (block.num_v_heads,))
    block.ssm_norm.weight = self._tensor_linspace(0.9, 1.1, (block.head_v_dim,))
    block.ssm_out.weight = self._tensor_linspace(-0.2, 0.18, (config.dim, config.ssm.inner_size))
    return block

  def _run_attention(self, block:GatedDeltaNetBlock, x:Tensor, start_pos:int):
    x_norm = block.attn_norm(x)
    block._init_state(x_norm)
    return block._attention(x_norm, start_pos).realize().numpy()

  def _cache_views(self, block:GatedDeltaNetBlock) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(block, 'conv_state'):
      return block.conv_state.numpy(), block.recurrent_state.numpy()
    else:
      conv_flat = (block.ssm_conv_kernel - 1) * block.conv_channels
      cache = block.delta_cache.numpy()
      conv_state = cache[:, :conv_flat].reshape(cache.shape[0], block.ssm_conv_kernel - 1, block.conv_channels)
      recurrent_state = cache[:, conv_flat:].reshape(cache.shape[0], block.num_v_heads, block.head_v_dim, block.head_v_dim)
      return conv_state, recurrent_state

  def _reset_state(self, block:GatedDeltaNetBlock):
    Tensor.realize(block.conv_state.assign(block.conv_state.const_like(0)),
                   block.recurrent_state.assign(block.recurrent_state.const_like(0)))

  def _linear_np(self, x:np.ndarray, weight:np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ weight.T.astype(np.float32)

  def _rms_norm_np(self, x:np.ndarray, weight:np.ndarray, eps:float) -> np.ndarray:
    x_float = x.astype(np.float32)
    return (x_float / np.sqrt((x_float * x_float).mean(axis=-1, keepdims=True) + eps)) * weight.astype(np.float32)

  def _normalize_np(self, x:np.ndarray, eps:float=1e-6) -> np.ndarray:
    return x / np.maximum(np.sqrt((x * x).sum(axis=-1, keepdims=True)), eps)

  def _softplus_np(self, x:np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

  def _silu_np(self, x:np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))

  def _naive_attention(self, block:GatedDeltaNetBlock, x:Tensor):
    x_np = x.numpy().astype(np.float32)
    B, T, _ = x_np.shape
    conv_state = np.zeros((B, block.ssm_conv_kernel - 1, block.conv_channels), dtype=np.float32)
    recurrent_state = np.zeros((B, block.num_v_heads, block.head_v_dim, block.head_v_dim), dtype=np.float32)
    conv_weight = block.ssm_conv1d["weight"].numpy().astype(np.float32).T[None, :, :]
    qkv_weight = block.attn_qkv.weight.numpy().astype(np.float32)
    gate_weight = block.attn_gate.weight.numpy().astype(np.float32)
    alpha_weight = block.ssm_alpha.weight.numpy().astype(np.float32)
    beta_weight = block.ssm_beta.weight.numpy().astype(np.float32)
    out_weight = block.ssm_out.weight.numpy().astype(np.float32)
    dt_bias = block.ssm_dt["bias"].numpy().astype(np.float32)
    ssm_a = block.ssm_a.numpy().astype(np.float32)
    attn_norm_weight = block.attn_norm.weight.numpy().astype(np.float32)
    ssm_norm_weight = block.ssm_norm.weight.numpy().astype(np.float32)
    outputs, conv_states, recurrent_states = [], [], []

    for t in range(T):
      x_norm = self._rms_norm_np(x_np[:, t:t+1, :], attn_norm_weight, block.attn_norm.eps)
      x_half = x_norm.astype(np.float16)
      out_gate = self._linear_np(x_half, gate_weight).reshape(B, 1, block.num_v_heads, block.head_v_dim)
      beta = 1.0 / (1.0 + np.exp(-self._linear_np(x_half, beta_weight))).reshape(B, block.num_v_heads, 1, 1)
      alpha = np.exp((self._softplus_np(self._linear_np(x_half, alpha_weight) + dt_bias)).reshape(B, block.num_v_heads, 1, 1) *
                     ssm_a.reshape(1, block.num_v_heads, 1, 1))
      conv_window = np.concatenate([conv_state, self._linear_np(x_half, qkv_weight)], axis=1)
      conv_out = self._silu_np((conv_window * conv_weight).sum(axis=1))
      q, k, v = np.split(conv_out, [block.q_dim, 2 * block.q_dim], axis=-1)
      q = self._normalize_np(q.reshape(B, block.num_k_heads, block.head_k_dim))
      k = self._normalize_np(k.reshape(B, block.num_k_heads, block.head_k_dim))
      v = v.reshape(B, block.num_v_heads, block.head_v_dim)
      if block.num_v_heads != block.num_k_heads:
        k_repeat = block.num_v_heads // block.num_k_heads
        q = np.repeat(q[:, None, :, :], k_repeat, axis=1).reshape(B, block.num_v_heads, block.head_k_dim)
        k = np.repeat(k[:, None, :, :], k_repeat, axis=1).reshape(B, block.num_v_heads, block.head_k_dim)
      q, k, v = (q * (block.head_k_dim ** -0.5))[..., None], k[..., None], v[..., None]
      recurrent_state = recurrent_state * alpha
      recurrent_state = recurrent_state + np.matmul((v - np.matmul(recurrent_state, k)) * beta, np.swapaxes(k, -1, -2))
      core_attn_out = np.matmul(recurrent_state, q).squeeze(-1).reshape(B, 1, block.num_v_heads, block.head_v_dim)
      core_attn_out = self._rms_norm_np(core_attn_out, ssm_norm_weight, block.ssm_norm.eps)
      out = self._linear_np((core_attn_out * self._silu_np(out_gate)).reshape(B, 1, -1).astype(np.float16), out_weight)
      conv_state = conv_window[:, 1:, :]
      outputs.append(out)
      conv_states.append(conv_state.copy())
      recurrent_states.append(recurrent_state.copy())

    return outputs, conv_states, recurrent_states

  def test_gatedeltanet_reference_and_reset(self):
    config = self._make_config(max_context=3)
    block = self._make_block(config)
    x = Tensor.linspace(-1.0, 1.0, 3 * config.dim, dtype=dtypes.float32).reshape(1, 3, config.dim)

    expected_outs, expected_conv, expected_recurrent = self._naive_attention(block, x)
    out = self._run_attention(block, x, 0)
    conv_state, recurrent_state = self._cache_views(block)
    np.testing.assert_allclose(out, np.concatenate(expected_outs, axis=1), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(conv_state, expected_conv[-1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(recurrent_state, expected_recurrent[-1], rtol=1e-3, atol=1e-3)
    self._reset_state(block)

    for step in range(x.shape[1]):
      out = self._run_attention(block, x[:, step:step+1], step)
      conv_state, recurrent_state = self._cache_views(block)
      np.testing.assert_allclose(out, expected_outs[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet output mismatch at step {step}")
      np.testing.assert_allclose(conv_state, expected_conv[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet conv cache mismatch at step {step}")
      np.testing.assert_allclose(recurrent_state, expected_recurrent[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet recurrent cache mismatch at step {step}")

    warmup = Tensor.linspace(-0.5, 0.5, 2 * config.dim, dtype=dtypes.float32).reshape(1, 2, config.dim)
    prompt = Tensor.linspace(0.75, -0.75, 2 * config.dim, dtype=dtypes.float32).reshape(1, 2, config.dim)

    for i in range(warmup.shape[1]): self._run_attention(block, warmup[:, i:i+1], i)
    self._reset_state(block)
    expected_outs, expected_conv, expected_recurrent = self._naive_attention(block, prompt)

    for step in range(prompt.shape[1]):
      out = self._run_attention(block, prompt[:, step:step+1], step)
      conv_state, recurrent_state = self._cache_views(block)
      np.testing.assert_allclose(out, expected_outs[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset output mismatch at step {step}")
      np.testing.assert_allclose(conv_state, expected_conv[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset conv cache mismatch at step {step}")
      np.testing.assert_allclose(recurrent_state, expected_recurrent[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset recurrent cache mismatch at step {step}")

  def test_kda_channel_decay(self):
    config = self._make_config(dim=4, hidden_dim=8, n_heads=2, head_dim=4, rope_dim=4, v_head_dim=4,
      ssm=SSMConfig(conv_kernel=2, state_size=2, group_count=2, time_step_rank=2, inner_size=4, kda=True))
    block, x = GatedDeltaNetBlock(config, config.ssm), Tensor([[[1., 2., 0., 0.], [2., 1., 0., 0.]]])
    block.ssm_f_a.weight = Tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    block.ssm_f_b.weight = Tensor([[1., 0.], [0., 1.], [1., 1.], [2., 1.]])
    block._init_state(x)
    initial_state = Tensor.arange(8, dtype=dtypes.float32).reshape(1, 2, 2, 2)
    block.recurrent_state.assign(initial_state).realize()
    block.ssm_a = Tensor([[-1.], [-1.]])
    block._attention(x, x.shape[1]).realize()
    alpha = np.exp(-self._softplus_np(np.array([[1, 2, 3, 4], [2, 1, 3, 5]])).reshape(2, 2, 2)).prod(0)
    np.testing.assert_allclose(block.recurrent_state.numpy(), initial_state.numpy() * alpha[..., None], rtol=1e-5, atol=1e-5)

  def test_kda_prefill_matches_decode(self):
    config = self._make_config(ssm=SSMConfig(conv_kernel=2, state_size=4, group_count=1, time_step_rank=1, inner_size=4, kda=True))
    block = GatedDeltaNetBlock(config, config.ssm)
    for p in nn.state.get_parameters(block):
      p.replace(self._tensor_linspace(-0.05, 0.05, p.shape) if len(p.shape) > 1 else self._tensor_linspace(0.05, 0.1, p.shape))
    x = self._tensor_linspace(-0.5, 0.5, (1, 3, config.dim))
    prefill = self._run_attention(block, x, 0)
    prefill_conv, prefill_recurrent = self._cache_views(block)
    self._reset_state(block)
    decode = np.concatenate([self._run_attention(block, x[:, i:i+1], i) for i in range(3)], axis=1)
    decode_conv, decode_recurrent = self._cache_views(block)
    np.testing.assert_allclose(prefill, decode, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(prefill_conv, decode_conv, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(prefill_recurrent, decode_recurrent, rtol=1e-3, atol=1e-3)

  def test_varied_chunk_sizes_match_decode(self):
    for kda in (False, True):
      ssm = SSMConfig(conv_kernel=2, state_size=4, group_count=1, time_step_rank=1, inner_size=4, kda=kda)
      config = self._make_config(ssm=ssm)
      if kda:
        block = GatedDeltaNetBlock(config, config.ssm)
        for p in nn.state.get_parameters(block):
          p.replace(self._tensor_linspace(-0.05, 0.05, p.shape) if len(p.shape) > 1 else self._tensor_linspace(0.05, 0.1, p.shape))
      else: block = self._make_block(config)
      x = self._tensor_linspace(-0.5, 0.5, (1, 4, config.dim))
      decode = np.concatenate([self._run_attention(block, x[:, i:i+1], i) for i in range(4)], axis=1)
      decode_conv, decode_recurrent = self._cache_views(block)
      for chunking in ([4], [2, 2], [1, 3], [3, 1], [2, 1, 1]):
        self._reset_state(block)
        outs, start = [], 0
        for size in chunking:
          outs.append(self._run_attention(block, x[:, start:start+size], start))
          start += size
        chunked_conv, chunked_recurrent = self._cache_views(block)
        np.testing.assert_allclose(np.concatenate(outs, axis=1), decode, rtol=1e-3, atol=1e-3, err_msg=f"{kda=} {chunking=}")
        np.testing.assert_allclose(chunked_conv, decode_conv, rtol=1e-3, atol=1e-3, err_msg=f"{kda=} {chunking=}")
        np.testing.assert_allclose(chunked_recurrent, decode_recurrent, rtol=1e-3, atol=1e-3, err_msg=f"{kda=} {chunking=}")

  def test_start_zero_resets_realized_state(self):
    config = self._make_config(max_context=3)
    x = self._tensor_linspace(-1, 1, (1, 3, config.dim))
    block = self._make_block(config)
    self._run_attention(block, x, 0)
    restarted = self._run_attention(block, x[:, :2], 0)
    fresh = self._run_attention(self._make_block(config), x[:, :2], 0)
    np.testing.assert_allclose(restarted, fresh, rtol=1e-3, atol=1e-3)

class TestPairwiseTopk(unittest.TestCase):
  def test_basic_topk(self):
    x = Tensor([[[1.0, 3.0, 2.0, 5.0, 4.0]]])
    vals, sel = pairwise_topk(x, 3)
    np.testing.assert_allclose(vals.numpy(), [[[3.0, 4.0, 5.0]]])
    np.testing.assert_equal(sel.numpy(), [[[1, 4, 3]]])

  def test_duplicates(self):
    x = Tensor([[[5.0, 5.0, 3.0, 5.0]]])
    vals, sel = pairwise_topk(x, 2)
    np.testing.assert_allclose(vals.numpy(), [[[5.0, 5.0]]])
    np.testing.assert_equal(sel.numpy(), [[[1, 0]]])

  def test_matches_numpy(self):
    np.random.seed(42)
    data = np.random.randn(4, 2, 16).astype(np.float32)
    vals, sel = pairwise_topk(Tensor(data), 5)
    for b in range(4):
      for t in range(2):
        expected = set(np.argsort(-data[b, t])[:5].tolist())
        self.assertEqual(set(sel.numpy()[b, t].tolist()), expected)
        np.testing.assert_allclose(vals.numpy()[b, t], data[b, t][sel.numpy()[b, t]])

if __name__ == '__main__':
  unittest.main()
