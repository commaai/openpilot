#!/usr/bin/env python3
from tinygrad import Tensor, TinyJit, nn
from extra.models.llama import FeedForward

if __name__ == "__main__":
  model = FeedForward(4096, 14336)
  for x in nn.state.get_parameters(model): x.replace(x.half()).realize()
  jrun = TinyJit(model)
  for i in range(5):
    print(f"*** run {i}")
    jrun(Tensor.rand(1, 4096))

