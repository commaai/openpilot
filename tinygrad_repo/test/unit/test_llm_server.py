import unittest
import numpy as np
from unittest.mock import patch
from tinygrad import Tensor, UOp
from tinygrad.nn.state import get_state_dict
from tinygrad.schedule import schedule_cache
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.serve import StreamRouter

TEST_CONFIG = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                           norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)
V_START_POS = UOp.variable("start_pos", 0, TEST_CONFIG.max_context-1)
V_TOKS = UOp.variable("toks", 1, 32)  # 32 is the default chunk_size in generate

class TestTransformerGenerate(unittest.TestCase):
  def test_warmup(self):
    model, calls = Transformer(TEST_CONFIG), []
    def generate(tokens, **kwargs):
      calls.append(tokens)
      yield from (1, 2)
    with patch.object(model, "generate", generate): model.warmup()
    self.assertEqual(calls, [[0], [0]])

  def test_warmup_then_generate_with_default_chunk(self):
    # warmup must not capture JIT graphs that generate()'s default chunk_size then rejects
    model = Transformer(TEST_CONFIG)
    model.warmup()
    self.assertIsInstance(next(model.generate([5, 6, 7, 8])), int)

  def test_first_recurrent_generate_before_state_init(self):
    model = Transformer(TEST_CONFIG)
    model.has_recurrent_block = True
    with patch.object(Transformer, '__call__', return_value=Tensor([[42]])):
      self.assertEqual(next(model.generate([0])), 42)

  def test_recurrent_live_state_reuse(self):
    model = Transformer(TEST_CONFIG)
    model.has_recurrent_block = True
    model._cached_tokens = [1, 2, 3, 4, 5]
    self.assertEqual(model.get_start_pos([1, 2, 3, 4, 5, 42, 10]), 5)
    calls = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      calls.append((tokens.shape, start_pos))
      return Tensor([[42]])
    with patch.object(Transformer, '__call__', mock_call):
      next(model.generate([1, 2, 3, 4, 5, 42, 10]))
    # resumes from the reused state at position 5 and consumes the 2 new tokens (one chunk or two decode steps)
    self.assertEqual(calls[0][1], V_START_POS.bind(5))
    def ntok(shape): return shape[1] if isinstance(shape[1], int) else shape[1].unbind()[1]
    self.assertEqual(sum(ntok(c[0]) for c in calls), 2)

  def test_recurrent_divergent_prompt_restarts(self):
    model, calls = Transformer(TEST_CONFIG), []
    model.has_recurrent_block, model._cached_tokens = True, [1, 2, 9]
    def mock_call(self, tokens, start_pos, temperature):
      calls.append(start_pos)
      return Tensor([[42]])
    with patch.object(Transformer, '__call__', mock_call): next(model.generate([1, 2, 10, 11]))
    self.assertEqual(calls[0], V_START_POS.bind(0))

  def test_template_starts_reasoning(self):
    router = StreamRouter(reasoning=True)
    self.assertEqual(list(router.route("reasoning</think>answer")),
                     [("reasoning_content", "reasoning"), ("content", "answer")])

  def test_kv_cache_reuse(self):
    """Test that generate reuses the KV cache when tokens extend the cached prefix."""
    model = Transformer(TEST_CONFIG)

    captured_inputs = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      captured_inputs.append((tokens.shape, start_pos))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation: prefill 5 tokens + 1 decode
      tokens = [1, 2, 3, 4, 5]
      gen = model.generate(tokens)
      next(gen)  # prefill
      next(gen)  # decode

      # second call extends the conversation — cached prefix should be reused
      captured_inputs.clear()
      tokens = [1, 2, 3, 4, 5, 42, 42, 10, 11, 12]
      gen = model.generate(tokens)
      next(gen)

    # should process tokens[6:] = [42, 10, 11, 12] since first 6 have cached k/v
    self.assertEqual(captured_inputs, [((1, V_TOKS.bind(4)), V_START_POS.bind(6))])

  def test_kv_cache_invalidation(self):
    """Test that generate invalidates the KV cache when tokens diverge from the cached prefix."""
    model = Transformer(TEST_CONFIG)

    captured_inputs = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      captured_inputs.append((tokens.shape, start_pos))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation
      gen = model.generate([1, 2, 3, 4, 5])
      next(gen)

      # completely different prompt — KV cache should be invalidated
      captured_inputs.clear()
      gen = model.generate([10, 20, 30])
      next(gen)

    # should process all 3 tokens from start
    self.assertEqual(captured_inputs, [((1, V_TOKS.bind(3)), V_START_POS.bind(0))])

  def test_two_prompts_schedule_cache(self):
    """Third prompt should hit the schedule cache, not miss (first two warm up both jits: prefill + decode)."""
    from dataclasses import replace
    model = Transformer(replace(TEST_CONFIG, max_context=64))

    # first two prompts warm up both jits (prefill + decode)
    ids = list(range(1, 6))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    ids += list(range(10, 15))
    gen = model.generate(ids)
    for _ in range(3): next(gen)
    cache_size_after_warmup = len(schedule_cache)

    # third prompt should reuse the same schedule cache entries, not create new ones
    ids += list(range(20, 25))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    self.assertEqual(cache_size_after_warmup, len(schedule_cache),
      f"third prompt added {len(schedule_cache) - cache_size_after_warmup} new schedule cache entries (expected 0)")

  def test_chunked_prefill(self):
    """When prompt > chunk_size, all chunks should be prefill"""
    from tinygrad.uop.ops import resolve
    from dataclasses import replace
    model = Transformer(replace(TEST_CONFIG, max_context=64))

    def get_prefill_flags(tokens, chunk_size):
      is_prefill = []
      def mock_call(self, tokens, start_pos, temperature, **kwargs):
        is_prefill.append(resolve(tokens.shape[1] != 1))
        return Tensor([[42]])
      with patch.object(Transformer, '__call__', mock_call):
        gen = model.generate(tokens, chunk_size=chunk_size)
        for _ in range(3): next(gen)
      model._cached_tokens = []
      return is_prefill

    # 8 tokens, chunk_size=4 -> 2 prefill chunks
    self.assertEqual(get_prefill_flags(list(range(8)), 4), [True, True, False, False])
    # 9 tokens, chunk_size=4 -> 3 prefill chunks (4+4+1)
    self.assertEqual(get_prefill_flags(list(range(9)), 4), [True, True, True, False, False])
    # 4 tokens, chunk_size=4 -> 1 prefill chunk
    self.assertEqual(get_prefill_flags(list(range(4)), 4), [True, False, False])

  def test_chunked_prefill_kv_cache_matches_single_chunk(self):
    config = TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=1000000, rope_dim=4, qk_norm=4, v_head_dim=4, max_context=16)
    def model():
      m = Transformer(config)
      rng = np.random.RandomState(1234)
      for t in get_state_dict(m).values():
        t.assign(Tensor(rng.uniform(-1, 1, t.shape).astype(np.float32))).realize()
      return m
    def prefill(m, chunk_size):
      gen = m.generate(list(range(1, 9)), chunk_size=chunk_size, temperature=0.0)
      next(gen)
      return [b.cache_kv.numpy() for b in m.blk]
    for g, r in zip(prefill(model(), 4), prefill(model(), 8)):
      np.testing.assert_allclose(g[:, :, :, :8, :], r[:, :, :, :8, :], atol=1e-5)

  def test_kv_cache_resume_matches_fresh(self):
    model = Transformer(TEST_CONFIG)

    # generate 2 tokens, then abandon
    prompt = list(range(1, 6))
    gen = model.generate(list(prompt))
    out1, out2 = next(gen), next(gen)

    # resume with conversation history + new user tokens appended
    extended = prompt + [out1, out2, 10, 11, 12]
    gen = model.generate(list(extended))
    resumed_out = [next(gen) for _ in range(3)]

    # compare against fresh generation (no cache) of the same prompt
    model._cached_tokens = []
    gen = model.generate(list(extended))
    fresh_out = [next(gen) for _ in range(3)]

    self.assertEqual(fresh_out, resumed_out)

  def test_temperature_zero_is_greedy(self):
    """Temperature 0 (or near 0) should produce deterministic output."""
    model = Transformer(TEST_CONFIG)
    tokens = list(range(1, 6))
    results = [list(zip(range(5), model.generate(list(tokens)))) for _ in range(3)]
    # all runs should produce the same tokens
    self.assertEqual(results[0], results[1])
    self.assertEqual(results[1], results[2])

  def test_temperature_high_produces_variety(self):
    """High temperature should produce different outputs across runs."""
    model = Transformer(TEST_CONFIG)
    tokens = list(range(1, 6))
    runs = set()
    for _ in range(5):
      gen = model.generate(list(tokens), temperature=2.0)
      out = tuple(next(gen) for _ in range(10))
      runs.add(out)
    # with temperature=2.0, we should see at least 2 distinct outputs across 5 runs
    self.assertGreater(len(runs), 1, "high temperature should produce varied outputs")

  def test_recurrent_temperature_high_produces_variety(self):
    model = Transformer(TEST_CONFIG)
    model.has_recurrent_block = True
    outputs = {model.forward(Tensor([[1]]), 0, Tensor([2.0])).item() for _ in range(5)}
    self.assertGreater(len(outputs), 1)

  def test_temperature_passed_to_forward(self):
    """Temperature from generate should be passed through to __call__."""
    model = Transformer(TEST_CONFIG)
    captured_temps = []
    def mock_call(self, tokens, start_pos, temperature, **kwargs):
      captured_temps.append(float(temperature.item()))
      return Tensor([[42]])
    with patch.object(Transformer, '__call__', mock_call):
      gen = model.generate([1, 2, 3], temperature=0.6)
      next(gen)
    self.assertAlmostEqual(captured_temps[-1], 0.6, places=5)

if __name__ == '__main__':
  unittest.main()
