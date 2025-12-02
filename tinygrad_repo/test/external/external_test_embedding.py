from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding

if __name__ == "__main__":
  vocab_size = 50257
  dim = 128
  test = Embedding(vocab_size, dim)
  ret = test(Tensor([[1,2,3]])).numpy()
