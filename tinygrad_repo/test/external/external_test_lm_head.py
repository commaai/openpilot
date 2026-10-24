from tinygrad import Tensor, nn

if __name__ == "__main__":
  vocab_size = 50257
  n_embd = 768
  lm_head = nn.Linear(n_embd, vocab_size, bias=False)
  bs = 4
  seq_len = 1024
  x = Tensor.rand(bs, seq_len, n_embd)
  ret = lm_head(x).realize()
