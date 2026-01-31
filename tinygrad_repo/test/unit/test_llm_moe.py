import unittest
import numpy as np
from tinygrad import Tensor

class TestMoEFeedForward(unittest.TestCase):
  def test_moe_feed_forward(self):
    from tinygrad.apps.llm import TransformerBlock
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(dim, hidden, n_heads, n_heads, norm_eps=1e-5, head_dim=dim//n_heads,
                             rope_theta=10000, max_context=16, num_experts=num_experts, num_experts_per_tok=k)

    # set up weights: gate scales by (expert_id+1), up/down are identity-ish, router picks experts 0,2
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T  # router strongly prefers experts 0 and 2
    block.ffn_norm.weight = Tensor.ones(dim)  # identity norm

    # input of ones -> after norm still ~ones -> experts 0,2 selected -> weighted sum of silu outputs
    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(h)

    # expected: residual + moe_output ≈ 1 + avg(silu(1), silu(3))
    expected = 1 + (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_moe_feed_forward_batched(self):
    from tinygrad.apps.llm import TransformerBlock
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(dim, hidden, n_heads, n_heads, norm_eps=1e-5, head_dim=dim//n_heads,
                             rope_theta=10000, max_context=16, num_experts=num_experts, num_experts_per_tok=k)

    # same setup as BS=1 test
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T
    block.ffn_norm.weight = Tensor.ones(dim)

    # test with BS=2, T=3
    h = Tensor.ones(2, 3, dim)
    out = block._feed_forward(h)

    # all outputs should match the BS=1 expected value
    expected = 1 + (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-2)

if __name__ == '__main__':
  unittest.main()
