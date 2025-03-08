#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor
import torch

def get_question_samp(bsz, seq_len, vocab_size, seed):
  np.random.seed(seed)
  in_ids= np.random.randint(vocab_size, size=(bsz, seq_len))
  mask = np.random.choice([True, False], size=(bsz, seq_len))
  seg_ids = np.random.randint(1, size=(bsz, seq_len))
  return in_ids, mask, seg_ids

def set_equal_weights(mdl, torch_mdl):
  from tinygrad.nn.state import get_state_dict
  state, torch_state = get_state_dict(mdl), torch_mdl.state_dict()
  assert len(state) == len(torch_state)
  for k, v in state.items():
    assert k in torch_state
    torch_state[k].copy_(torch.from_numpy(v.numpy()))
  torch_mdl.eval()

class TestBert(unittest.TestCase):
  def test_questions(self):
    from extra.models.bert import BertForQuestionAnswering
    from transformers import BertForQuestionAnswering as TorchBertForQuestionAnswering
    from transformers import BertConfig

    # small
    config = {
      'vocab_size':24, 'hidden_size':2, 'num_hidden_layers':2, 'num_attention_heads':2,
      'intermediate_size':32, 'hidden_dropout_prob':0.1, 'attention_probs_dropout_prob':0.1,
      'max_position_embeddings':512, 'type_vocab_size':2
      }

    # Create in tinygrad
    Tensor.manual_seed(1337)
    mdl = BertForQuestionAnswering(**config)

    # Create in torch
    with torch.no_grad():
      torch_mdl = TorchBertForQuestionAnswering(BertConfig(**config))

    set_equal_weights(mdl, torch_mdl)

    seeds = (1337, 3141)
    bsz, seq_len = 1, 16
    for _, seed in enumerate(seeds):
      in_ids, mask, seg_ids = get_question_samp(bsz, seq_len, config['vocab_size'], seed)
      out = mdl(Tensor(in_ids), Tensor(mask), Tensor(seg_ids))
      torch_out = torch_mdl.forward(torch.from_numpy(in_ids).long(), torch.from_numpy(mask), torch.from_numpy(seg_ids).long())[:2]
      torch_out = torch.cat(torch_out).unsqueeze(2)
      np.testing.assert_allclose(out.numpy(), torch_out.detach().numpy(), atol=5e-4, rtol=5e-4)

if __name__ == '__main__':
  unittest.main()
