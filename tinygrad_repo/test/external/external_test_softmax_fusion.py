from tinygrad import Tensor, GlobalCounters, Context

if __name__ == "__main__":
  with Context(TRACK_MATCH_STATS=0): test = Tensor.ones(32, 10).contiguous().realize()
  GlobalCounters.reset()

  # this is the softmax from scaled_dot_product_attention
  # it becomes 3 kernels
  print("*** softmax ***")
  with Context(NOOPT=1):
    out = test.softmax(-1)
    out.realize()
