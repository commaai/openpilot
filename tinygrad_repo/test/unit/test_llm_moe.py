import unittest
import numpy as np
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.llm.model import TransformerBlock, TransformerConfig

def _moe_config(dim=8, hidden=16, n_heads=2, num_experts=4, num_experts_per_tok=2):
  return TransformerConfig(
    num_blocks=1, dim=dim, hidden_dim=hidden, n_heads=n_heads, n_kv_heads=n_heads,
    norm_eps=1e-5, vocab_size=100, head_dim=dim//n_heads, rope_theta=10000,
    rope_dim=dim//n_heads, v_head_dim=dim//n_heads, max_context=16,
    num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

class TestMoEFeedForward(unittest.TestCase):
  def test_moe_feed_forward(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(_moe_config(dim, hidden, n_heads, num_experts, k))

    # set up weights: gate scales by (expert_id+1), up/down are identity-ish, router picks experts 0,2
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T  # router strongly prefers experts 0 and 2
    block.ffn_norm.weight = Tensor.ones(dim)  # identity norm

    # input of ones -> after norm still ~ones -> experts 0,2 selected -> weighted sum of silu outputs
    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    # expected moe_output ≈ avg(silu(1), silu(3))
    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_moe_feed_forward_batched(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(_moe_config(dim, hidden, n_heads, num_experts, k))

    # same setup as BS=1 test
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T
    block.ffn_norm.weight = Tensor.ones(dim)

    # test with BS=2, T=3
    h = Tensor.ones(2, 3, dim)
    out = block._feed_forward(block.ffn_norm(h))

    # all outputs should match the BS=1 expected value
    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-2)

  def test_moe_feed_forward_norm_topk_prob(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(replace(_moe_config(dim, hidden, n_heads, num_experts, k), norm_topk_prob=True))

    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[0.1, 0, 0.1, 0]] * dim).T  # equal top-2 experts, but only ~69% mass before renorm
    block.ffn_norm.weight = Tensor.ones(dim)

    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_moe_feed_forward_shared_expert(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(replace(_moe_config(dim, hidden, n_heads, num_experts, k), shared_expert_dim=dim))

    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T
    block.ffn_gate_shexp.weight = Tensor.eye(dim) * 2
    block.ffn_up_shexp.weight = Tensor.eye(dim)
    block.ffn_down_shexp.weight = Tensor.eye(dim)
    block.ffn_gate_inp_shexp["weight"] = Tensor.zeros(dim)
    block.ffn_norm.weight = Tensor.ones(dim)

    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    moe_expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    shared_expected = Tensor([2.0]).silu().item() * 0.5
    expected = moe_expected + shared_expected
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-2)

if __name__ == '__main__':
  unittest.main()
