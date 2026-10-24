import unittest
from tinygrad import Tensor
from tinygrad.llm.model import Transformer, TransformerConfig

class TestMLA(unittest.TestCase):
  def _make_config(self, **kwargs):
    return TransformerConfig(**{
      "num_blocks": 1, "dim": 64, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 1,
      "norm_eps": 1e-5, "vocab_size": 100, "head_dim": 16, "rope_theta": 10000.0, "rope_dim": 8, "max_context": 32,
      "kv_lora_rank": 16, "v_head_dim": 8,
    } | kwargs)

  def test_shared_expert_gate_optional(self):
    from tinygrad import nn
    model = Transformer(self._make_config(num_experts=4, num_experts_per_tok=2, shared_expert_dim=32, shared_expert_gate=False))
    self.assertNotIn('blk.0.ffn_gate_inp_shexp.weight', nn.state.get_state_dict(model))
    out = model.blk[0]._feed_forward(Tensor.randn(1, 4, model.blk[0].config.dim))
    self.assertEqual(out.shape, (1, 4, model.blk[0].config.dim))

if __name__ == '__main__':
  unittest.main()
