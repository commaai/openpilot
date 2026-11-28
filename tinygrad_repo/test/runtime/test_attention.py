import unittest, math
import numpy as np
from tinygrad import Tensor, dtypes, nn, UOp, TinyJit
from tinygrad.llm.model import (
  GatedDeltaNetBlock, SSMConfig, TransformerBlock, TransformerConfig, MLATransformerBlock, Indexer, IndexerConfig,
  HyperConnection, HCConfig, AttentionResidual, GatedResidual, PLELayer, PLEConfig, Compressor,
  apply_rope as apply_rope_new, precompute_freqs_cis, pairwise_topk, short_conv, RelAttnConfig, RelAttentionBlock, sink_attention,
)
from tinygrad.llm.gguf import GGUFPackedTable

def apply_rope(x:Tensor, start_pos:int):
  B, H, T, Hd = x.shape
  precompute_freqs_cis.cache_clear()
  freqs_cis = precompute_freqs_cis(Hd, start_pos+T)[start_pos:start_pos+T]
  return apply_rope_new(x, freqs_cis)

class TestAttention(unittest.TestCase):
  def _make_config(self, **kwargs):
    return TransformerConfig(**({"num_blocks":1, "dim":4, "hidden_dim":4, "n_heads":2, "n_kv_heads":1,
                                 "norm_eps":1e-5, "vocab_size":32, "head_dim":2, "rope_theta":10000.0,
                                 "rope_dim":2, "v_head_dim":2, "max_context":4} | kwargs))

  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

  def test_rope_per_row_positions(self):
    freqs = Tensor([[[1., 0.]], [[0., 1.]]])
    x = Tensor([[[[1., 0.]]], [[[1., 0.]]]])
    out = apply_rope_new(x, freqs)
    np.testing.assert_array_equal(out.numpy(), [[[[1., 0.]]], [[[0., 1.]]]])

  def test_rope_layouts(self):
    # 4 of 6 dims rotated, the first frequency by 90 degrees and the second by 0: split pairs (i, i+2) or adjacent pairs, leading or trailing dims
    x, freqs = Tensor([[[[1., 2., 3., 4., 5., 6.]]]]), Tensor([[0., 1., 1., 0.]])
    out = Tensor.stack(*(apply_rope_new(x, freqs, interleaved, last) for interleaved in (False, True) for last in (False, True))).numpy()
    np.testing.assert_array_equal(out[:, 0, 0, 0], [[-3, 2, 1, 4, 5, 6], [1, 2, -5, 4, 3, 6], [-2, 1, 3, 4, 5, 6], [1, 2, -4, 3, 5, 6]])

  def test_independent_positions_jit(self):
    block = TransformerBlock(self._make_config(dim=2, hidden_dim=2, n_heads=1))
    block.attn_q.weight = block.attn_k.weight = Tensor.zeros(2, 2)
    block.attn_v.weight = block.attn_output.weight = Tensor.eye(2)
    prefix = Tensor([[[1., 1.], [3., 3.], [5., 5.]], [[2., 2.], [4., 4.], [6., 6.]]])
    block._init_state(prefix)
    block.cache_kv.assign(float('nan')).realize()
    block._attention(prefix, (0,)).realize()
    p0, p1 = UOp.variable('pos0', 0, 3), UOp.variable('pos1', 0, 3)
    @TinyJit
    def forward(x, pos0, pos1): return block._attention(x, (pos0, pos1))
    # Zero queries/keys make attention the mean of each row's valid values.
    for positions, (a, b), expected in [((3, 1), (7., 8.), [[4., 4.], [5., 5.]]),
                                        ((0, 2), (9., 10.), [[9., 9.], [20/3, 20/3]]),
                                        ((1, 0), (11., 12.), [[10., 10.], [12., 12.]])]:
      out = forward(Tensor([[[a, a]], [[b, b]]]), p0.bind(positions[0]), p1.bind(positions[1]))
      np.testing.assert_allclose(out.numpy()[:, 0], expected, atol=1e-6, rtol=1e-6)
    self.assertEqual(forward.cnt, 3)
    self.assertIsNotNone(forward.captured)

  def test_attention_sink(self):
    x = Tensor([[[2., 4.]]])
    def attention(attn_sinks:bool):
      block = TransformerBlock(self._make_config(dim=2, hidden_dim=2, n_heads=1, attn_sinks=attn_sinks))
      block.attn_q.weight, block.attn_k.weight = Tensor.zeros(2, 2), Tensor.zeros(2, 2)
      block.attn_v.weight, block.attn_output.weight = Tensor.eye(2), Tensor.eye(2)
      block._init_state(x)
      return block._attention(x, (0,))
    self.assertTrue(attention(False).allclose(x, rtol=0, atol=1e-6).item())
    # the token and zero-value sink have equal weights: ([2,4] + [0,0]) / 2
    self.assertTrue(attention(True).allclose(x / 2, rtol=0, atol=1e-6).item())

  def test_sink_attention_mask(self):
    # q is zero, so the weights are exp(mask) and exp(sink) per row, the sink has a zero value: 2/(1+1), (2 + 4*2)/(1+2+5)
    mask, sink = Tensor([[[[0., -math.inf], [0., math.log(2)]]]]), Tensor([[[[0.], [math.log(5)]]]])
    out = sink_attention(Tensor.zeros(1, 1, 2, 1), Tensor.zeros(1, 1, 2, 1), Tensor([[[[2.], [4.]]]]), mask, sink)
    np.testing.assert_allclose(out.numpy().flatten(), [1., 1.25], rtol=1e-6)

  def test_sliding_window(self):
    x = Tensor([[[1., 1.], [3., 3.], [5., 5.]]])
    block = TransformerBlock(self._make_config(dim=2, hidden_dim=2, n_heads=1, sliding_window=2))
    block.attn_q.weight, block.attn_k.weight = Tensor.zeros(2, 2), Tensor.zeros(2, 2)
    block.attn_v.weight, block.attn_output.weight = Tensor.eye(2), Tensor.eye(2)
    block._init_state(x)
    expected = Tensor([[[1., 1.],  # only itself
                        [2., 2.],  # ([1,1] + [3,3]) / 2
                        [4., 4.]]]) # ([3,3] + [5,5]) / 2
    self.assertTrue(block._attention(x, (0,)).allclose(expected, rtol=0, atol=1e-6).item())

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
    block._attention(x_norm, (0,)).realize()

    expected = apply_rope_new(k[..., :rope_dim], block.freqs_cis[:seqlen]).cat(k[..., rope_dim:], dim=-1)
    np.testing.assert_allclose(block.cache_kv[0, :, :, :seqlen, :].numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)

  def test_hyperconnection(self):
    hc = HyperConnection(self._make_config(dim=1, hc=HCConfig(count=2, sinkhorn_iters=1, eps=0.0)))
    hc.fn.weight = Tensor.zeros(8, 2)
    hc.base["weight"] = Tensor([0., 0., 0., 0., float(np.log(3)), 0., 0., 0.])
    # pre averages streams [1, 3] to 2, the sublayer squares it to 4, and post adds 4 to both streams
    # mixing rows [3/4, 1/4], [1/2, 1/2] normalize to [3/5, 1/3], [2/5, 2/3]
    expected = [[[[4 + (3/5 + 3*2/5)], [4 + (1/3 + 3*2/3)]]]]
    for mask in (None, Tensor([0., -np.inf])):
      with self.subTest(mask=mask is not None):
        actual, returned_mask, pre = hc(Tensor([[[[1.], [3.]]]]), lambda x: (x.square(), mask))
        self.assertIs(returned_mask, mask)
        self.assertIsNone(pre)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-7)

  def test_hyperconnection_delayed(self):
    hc = HyperConnection(self._make_config(dim=1, hc=HCConfig(count=2, sinkhorn_iters=1, eps=0.0, pre_delayed=True)))
    hc.fn.weight = Tensor.zeros(8, 2)
    hc.base["weight"] = Tensor([0., 0., 0., 0., float(np.log(3)), 0., 0., 0.])
    # the sublayer sees streams [1, 3] weighed by the previous pre, stream 0 when there is none
    for pre, seen in ((None, 1.), (Tensor([[[0., 1.]]]), 3.), (Tensor([[[.5, .5]]]), 2.)):
      with self.subTest(seen=seen):
        actual, _, own = hc(Tensor([[[[1.], [3.]]]]), lambda x: (x.square(), None), pre)
        np.testing.assert_allclose(actual.numpy(), [[[[seen**2 + (3/5 + 3*2/5)], [seen**2 + (1/3 + 3*2/3)]]]], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(own.numpy(), [[[.5, .5]]], rtol=1e-6)

  def test_hyperconnection_head(self):
    # the output head only mixes: pre weights [1/2, 3/4] sum the streams [1, 3] to 11/4
    hc = HyperConnection(self._make_config(dim=1, hc=HCConfig(count=2, sinkhorn_iters=1, eps=0.0)), head=True)
    self.assertEqual((hc.fn.weight.shape, hc.base["weight"].shape, hc.scale["weight"].shape), ((2, 2), (2,), (1,)))
    hc.fn.weight, hc.base["weight"] = Tensor.zeros(2, 2), Tensor([0., float(np.log(3))])
    np.testing.assert_allclose(hc.mix(Tensor([[[[1.], [3.]]]]))[0].numpy().flatten(), [11/4], rtol=1e-6)

  def test_attention_residual(self):
    # rmsnorm makes the streams [1, -3] into scores [log 3, -log 3], softmax [9/10, 1/10] mixes them to 0.6, the sublayer squares it
    for new_block, expected in ((True, [1, -3, 0.36]), (False, [1, -3 + 0.36])):
      res = AttentionResidual(self._make_config(dim=1), new_block)
      res.weight = Tensor([np.log(3)])
      actual, _, _ = res(Tensor([[[[1.], [-3.]]]]), lambda x: (x.square(), None))
      np.testing.assert_allclose(actual.numpy().flatten(), expected, rtol=1e-5)

  def test_gated_residual(self):
    res = GatedResidual(self._make_config(dim=1, hc=HCConfig(count=2, sinkhorn_iters=0, eps=0.0, low_rank=1)))
    res.norm.weight, res.down.weight, res.up.weight = Tensor([2., 4.]), Tensor([[np.log(3)/3, np.log(3)/3]]), Tensor([[0.], [4/3]])
    res.inject.weight = Tensor([[0., 0.], [0., np.log(3)/2]])
    # norm [2, 4], gate [1/2, 3/4] mixes to 2, squared to 4, inject gates [1, 3/2] add [4, 6]
    actual, _, _ = res(Tensor([[[[1.], [1.]]]]), lambda x: (x.square(), None))
    np.testing.assert_allclose(actual.numpy().flatten(), [5, 7], rtol=1e-4)

  def test_indexer(self):
    indexer = Indexer(self._make_config(dim=2, q_lora_rank=2, indexer=IndexerConfig(n_heads=1, head_dim=2, top_k=2, kpool=2, norm_eps=1e-6)))
    indexer.attn_q_b.weight, indexer.attn_k.weight = Tensor.eye(2), Tensor.eye(2)
    indexer.proj.weight, indexer.compressor_gate.weight = Tensor.ones(1, 2), Tensor.zeros(2, 2)
    x, q = Tensor([[[2., 0.], [2., 0.], [0., 2.], [0., 2.]]]), Tensor([[[1., -1.], [1., -1.], [1., -1.], [-1., 1.]]])
    indexer._init_state(x)
    # the first three queries prefer the first pool, the final query selects the completed second pool
    expected = np.where([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]], 0., -np.inf)
    np.testing.assert_array_equal(indexer(x, q, (0,)).numpy(), expected[None, None])

  def test_indexer_no_kpool(self):
    # glm-dsa: every key is scored on its own (no k-pools) and the first rope_dim dims are rotated, interleaved
    idx = IndexerConfig(n_heads=1, head_dim=2, top_k=2, kpool=1, norm_eps=1e-6, rope_dim=2)
    indexer = Indexer(self._make_config(dim=2, q_lora_rank=2, indexer=idx))
    indexer.attn_q_b.weight, indexer.attn_k.weight = Tensor.eye(2), Tensor.eye(2)
    indexer.proj.weight = Tensor.ones(1, 2)
    x, q = Tensor([[[2., 0.], [2., 0.], [0., 2.], [0., 2.]]]), Tensor([[[1., -1.], [1., -1.], [1., -1.], [-1., 1.]]])
    indexer._init_state(x)
    # reference: interleaved rope on q/k, relu scores, top-2 of the causal positions
    pos = np.arange(4)
    cos, sin = np.cos(pos), np.sin(pos)  # rope_dim=2 -> one frequency, theta=10000**0
    def irope(v):
      return np.concatenate([v[:, 0::2]*cos[:, None] - v[:, 1::2]*sin[:, None], v[:, 0::2]*sin[:, None] + v[:, 1::2]*cos[:, None]], axis=-1)
    scores = np.maximum(irope(q.numpy()[0]) @ irope(x.numpy()[0]).T, 0) / 2**0.5
    expected = np.full((4, 4), -np.inf)
    for t in range(4): expected[t, np.argsort(scores[t, :t+1])[-2:]] = 0.0
    np.testing.assert_allclose(indexer(x, q, (0,), precompute_freqs_cis(2, 4, 10000.0)).numpy(), expected[None, None])

  def test_indexer_msa(self):
    # minimax-m3: each head selects the pools with its best max(q·k) and always its own pool
    idx = IndexerConfig(n_heads=2, head_dim=2, top_k=4, kpool=2, norm_eps=1e-6, msa=True)
    indexer = Indexer(self._make_config(dim=2, max_context=6, indexer=idx))
    indexer.attn_q_b.weight, indexer.attn_k.weight = Tensor([[1., 0.], [0., 0.], [0., 0.], [0., 1.]]), Tensor.eye(2)
    x = Tensor([[[1., 0.], [-1., 0.], [1., 1.], [1., 1.], [-1., -1.], [-1., -1.]]])
    indexer._init_state(x)
    # head 0 picks the first pool by its best key (the second has the higher sum), head 1 the second, the last query keeps its own low pool
    allowed = np.tril(np.ones((2, 6, 6)))
    allowed[0, 4:, 2:4] = allowed[1, 4:, 0:2] = 0
    np.testing.assert_array_equal(indexer(x, Tensor.ones(1, 6, 2), (0,)).numpy(), np.where(allowed, 0., -np.inf)[None])

  def test_indexer_qsa(self):
    # a pool's key is its normed mean key, roped at the pool start
    idx = IndexerConfig(n_heads=1, head_dim=2, top_k=2, kpool=2, norm_eps=1e-6, rope_dim=2, qsa=True)
    indexer = Indexer(self._make_config(dim=2, indexer=idx))
    indexer.attn_q_b.weight, indexer.attn_k.weight = Tensor.eye(2), Tensor.eye(2)
    x, q = Tensor([[[-6., 8.], [0., 0.], [0., 2.], [2., -2.]]]), Tensor([[[1., 0.], [1., 0.], [1., 0.], [0., -1.]]])
    indexer._init_state(x)
    # a quarter turn per position: query 3 faces 180 degrees like pool 1 (mean (1, 0) roped at 2), not the larger pool 0 at 127 degrees
    freqs = Tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    allowed = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]]
    np.testing.assert_array_equal(indexer(x, q, (0,), freqs).numpy(), np.where(allowed, 0., -np.inf)[None, None])

  def test_indexer_candidates(self):
    # deepseek-v4.1: later layers pick only from the candidates, not the best pool 0
    indexer = Indexer(self._make_config(dim=2, max_context=8, compress_ratios=(1,), indexer=IndexerConfig(n_heads=1, head_dim=1, top_k=1)))
    indexer.attn_q_b.weight = indexer.proj.weight = Tensor.ones(1, 2)
    x, keys, candidates = Tensor.ones(1, 1, 2), Tensor([[[6.], [5.]]]), Tensor([[[[False, True]]]])
    np.testing.assert_array_equal(indexer(x, x, (1,), None, keys, candidates.cat(candidates, dim=1)).numpy(), [[[[0, 1]], [[0, 1]]]])

  def test_compressor(self):
    # ratio 4 (overlap): pool j weighs the first half of pool j-1's keys and the second half of its own, tokens 0-1 come from the rolling state
    comp, rng = Compressor(self._make_config(dim=2, max_context=12), 4, 2), np.random.default_rng(0)
    wkv, wg, ape, x = (rng.standard_normal(s).astype(np.float32) for s in ((4, 2), (4, 2), (4, 4), (1, 12, 2)))
    comp.kv.weight, comp.gate.weight, comp.ape["weight"] = Tensor(wkv), Tensor(wg), Tensor(ape)
    kvg = np.stack([x[0] @ wkv.T, x[0] @ wg.T], 1)  # (12, 2, 4)
    comp._init_state(xt:=Tensor(x))
    comp.cache_kg = Tensor(np.concatenate([np.zeros((6, 2, 4), np.float32), kvg[:2]])[None]).realize()
    out = comp(xt[:, 2:], (2,), Tensor([[1., 0.]] * 16)).numpy()[0, :3]  # identity rope
    np.testing.assert_allclose(comp.cache_kg.numpy()[0], kvg[4:], rtol=1e-6)  # the next state holds the last 8 tokens
    kv, g = kvg[:, 0].reshape(3, 4, 4), kvg[:, 1].reshape(3, 4, 4) + ape
    prev_kv, prev_g = np.concatenate([np.zeros((1, 4, 2)), kv[:-1, :, :2]]), np.concatenate([np.full((1, 4, 2), -np.inf), g[:-1, :, :2]])
    kv, g = np.concatenate([prev_kv, kv[:, :, 2:]], 1), np.concatenate([prev_g, g[:, :, 2:]], 1)
    pools = (np.exp(g) / np.exp(g).sum(1, keepdims=True) * kv).sum(1)
    np.testing.assert_allclose(out, pools / np.sqrt((pools ** 2).mean(-1, keepdims=True) + 1e-5), rtol=2e-3, atol=2e-3)

  def test_mla_indexer_mask(self):
    idx = IndexerConfig(n_heads=1, head_dim=2, top_k=2, kpool=2, norm_eps=1e-6)
    block = MLATransformerBlock(self._make_config(dim=2, kv_lora_rank=2, q_lora_rank=2, max_context=5, indexer=idx))
    block.attn_q_a.weight = Tensor([[1., 1.], [-1., -1.]])
    block.indexer.attn_q_b.weight, block.indexer.attn_k.weight = Tensor.eye(2), Tensor.eye(2)
    block.indexer.proj.weight, block.indexer.compressor_gate.weight = Tensor.ones(1, 2), Tensor.zeros(2, 2)
    x = Tensor([[[2., 0.], [2., 0.], [0., 2.], [0., 2.], [0., 2.]]])
    block._init_state(x)
    # every query prefers the first pool, and incomplete pools contribute only their visible tail
    expected = np.where([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 1]], 0., -np.inf)
    np.testing.assert_array_equal(block._indexer_mask(x, (0,), None).numpy(), expected[None, None])

  def test_ple_rows(self):
    # a bigram and a trigram head, EOS (7) before the start or an EOS
    ple = PLELayer(self._make_config(hc=HCConfig(2, 0, 0.0), ple=PLEConfig((0,), 32, (1, 2, 4), (13, 17), 7, 2)))
    table = np.random.default_rng(0).integers(0, 256, (30, 18), dtype=np.uint8)
    ple.table = GGUFPackedTable((Tensor(table.flatten()).pad((7, 11)).contiguous().realize()[7:-11], (30, 32), 20))
    for start_pos in (0, 2): ple.write_rows([3, 7, 5, 9], start_pos, 2, 2)
    # (t0 ^ 2*t1) % 13, (t0 ^ 2*t1 ^ 4*t2) % 17 + 13
    np.testing.assert_array_equal(ple.table.rows.numpy()[:4], table[[[0, 13], [1, 25], [11, 19], [3, 27]]])

  def test_rel_conv_taps(self):
    # each head adds its taps to the conv weight, out = (x + conv) / sqrt(5)
    ra = RelAttnConfig(length=4, extent=8, qk_clip=5.0, rel_clip=5.0, log_scale_floor=2, log_scale_alpha=0.5, conv_kernel=2)
    block, conv = RelAttentionBlock(self._make_config(n_kv_heads=2, rel_attn=ra)), {"weight": Tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])}
    block.conv_states, block.initial = {conv["weight"]: Tensor([[[1., 2., 3., 4.]]]).contiguous().realize()}, Tensor(False)
    out = block._conv(conv, Tensor([[[1., 0., 2., 1.], [0., 3., 1., 2.]]]), Tensor([[[1., 0., 0., 2.], [0., 3., 1., 0.]]]))
    np.testing.assert_allclose((out * 5 ** 0.5).numpy(), [[[3., 0., 13., 9.], [4., 6., 5., 4.]]], rtol=1e-6)

class TestGatedDeltaNetBlock(unittest.TestCase):
  def _tensor_linspace(self, start:float, stop:float, shape:tuple[int, ...]) -> Tensor:
    return Tensor.linspace(start, stop, int(np.prod(shape)), dtype=dtypes.float32).reshape(*shape)

  def _make_config(self, **kwargs):
    return TransformerConfig(**({"num_blocks":1, "dim":8, "hidden_dim":16, "n_heads":1, "n_kv_heads":1,
                                 "norm_eps":1e-5, "vocab_size":32, "head_dim":8, "rope_theta":10000.0,
                                 "rope_dim":8, "v_head_dim":8, "max_context":4, "ssm_layers":(True,),
                                 "ssm":SSMConfig(conv_kernel=2, state_size=4, group_count=1, time_step_rank=1, inner_size=4,
                                                 silu_gate=True)} | kwargs))

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
    return block._attention(x_norm, (start_pos,)).realize().numpy()

  def _cache_views(self, block:GatedDeltaNetBlock) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(block, 'conv_state'):
      return block.conv_state.numpy(), block.recurrent_state.numpy()
    elif hasattr(block, 'conv_states'):
      return np.concatenate([s.numpy() for s in block.conv_states], -1), block.recurrent_state.numpy()
    else:
      conv_flat = (block.ssm_conv_kernel - 1) * block.conv_channels
      cache = block.delta_cache.numpy()
      conv_state = cache[:, :conv_flat].reshape(cache.shape[0], block.ssm_conv_kernel - 1, block.conv_channels)
      recurrent_state = cache[:, conv_flat:].reshape(cache.shape[0], block.num_v_heads, block.head_v_dim, block.head_v_dim)
      return conv_state, recurrent_state

  def _reset_state(self, block:GatedDeltaNetBlock):
    Tensor.realize(*[s.assign(s.const_like(0)) for s in (block.conv_states if hasattr(block, 'conv_states') else [block.conv_state])],
                   block.recurrent_state.assign(block.recurrent_state.const_like(0)))

  def _linear_np(self, x:np.ndarray, weight:np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ weight.T.astype(np.float32)

  def _rms_norm_np(self, x:np.ndarray, weight:np.ndarray, eps:float) -> np.ndarray:
    x_float = x.astype(np.float32)
    return (x_float / np.sqrt((x_float * x_float).mean(axis=-1, keepdims=True) + eps)) * weight.astype(np.float32)

  def _l2norm_np(self, x:np.ndarray, eps:float=1e-6) -> np.ndarray:  # as in the FLA kernels
    return x / np.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)

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
      q = self._l2norm_np(q.reshape(B, block.num_k_heads, block.head_k_dim))
      k = self._l2norm_np(k.reshape(B, block.num_k_heads, block.head_k_dim))
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

  def test_kda_qk_l2norm(self):
    # q/k are normalized as x/sqrt(sum(x²)+1e-6) like FLA: tiny vectors aren't scaled up to unit length
    ssm = SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=2, inner_size=16, kda=True)
    block = GatedDeltaNetBlock(self._make_config(dim=16, n_heads=2, ssm=ssm), ssm)
    block.attn_q.weight = block.attn_k.weight = Tensor.full((16, 16), 1e-4/16)  # x of ones: q = k = 1e-4, v = 1
    block.attn_v.weight, block.ssm_beta.weight = Tensor.full((16, 16), 1/16), Tensor.zeros(2, 16)
    block.ssm_conv1d_q["weight"] = block.ssm_conv1d_k["weight"] = block.ssm_conv1d_v["weight"] = Tensor.ones(16, 4)
    x = Tensor.ones(1, 1, 16)
    block._init_state(x)
    block._attention(x, (0,)).realize()
    # the first token's state is beta * v (x) l2norm(k), with beta = sigmoid(0) and v, k through the conv and silu
    def silu(z): return z / (1 + np.exp(-z))
    np.testing.assert_allclose(block.recurrent_state.numpy(), 0.5 * silu(1.0) * silu(1e-4) / np.sqrt(8 * silu(1e-4)**2 + 1e-6), rtol=1e-3)

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
    block._attention(x, (x.shape[1],)).realize()
    alpha = np.exp(-self._softplus_np(np.array([[1, 2, 3, 4], [2, 1, 3, 5]])).reshape(2, 2, 2)).prod(0)
    np.testing.assert_allclose(block.recurrent_state.numpy(), initial_state.numpy() * alpha[:, None, :], rtol=1e-5, atol=1e-5)

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

class TestShortConv(unittest.TestCase):
  def test_short_conv(self):
    # the window is [state 1, 2 | x 3, 4 | pad 0], tap i of output t reads row t + i*dilation, x becomes the state
    for weight, dilation, initial, expected in (([[1., 0., 10.]], 1, False, [31, 42]), ([[1., 0., 10.]], 1, True, [30, 40]),
                                                ([[1., 10.]], 2, False, [31, 42])):
      with self.subTest(dilation=dilation, initial=initial):
        state = Tensor([[[1.], [2.]]]).contiguous().realize()
        x = Tensor([[[3.], [4.], [0.]]])[:, :UOp.variable("T", 1, 3).bind(2)]
        out, store = short_conv(state, x, Tensor(weight), Tensor(initial), dilation)
        np.testing.assert_allclose(out.numpy().flatten()[:2], expected, rtol=1e-6)
        Tensor(state.uop.after(store)).realize()
        np.testing.assert_array_equal(state.numpy().flatten(), [3, 4])

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
