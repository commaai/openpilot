#!/usr/bin/env python3
from tinygrad import Tensor, TinyJit, nn, dtypes
from tinygrad.helpers import getenv
from extra.models.llama import TransformerBlock, precompute_freqs_cis

BS = getenv("BS", 1)
SEQLEN = getenv("SEQLEN", 128)

# DEFAULT_FLOAT=bfloat16 SEQLEN=8192 ASM_GEMM=1 HK_FLASH_ATTENTION=1 DEV=NULL:HIP:gfx950 DEBUG=2 VIZ=1 PYTHONPATH="."
# python test/external/external_test_llama3_layer.py

if __name__ == "__main__":
  dim, hidden_dim, n_heads, n_kv_heads, norm_eps = 4096, 14336, 32, 8, 1e-5
  layer = TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context=0)
  for x in nn.state.get_parameters(layer): x.replace(x.cast(dtypes.default_float)).realize()

  freqs_cis = precompute_freqs_cis(dim // n_heads, SEQLEN, theta=500000.0).contiguous().realize()

  @TinyJit
  def run(t): return layer(t, 0, freqs_cis, None)

  for i in range(5):
    print(f"*** run {i}")
    run(Tensor.rand(BS, SEQLEN, dim, dtype=dtypes.default_float).realize())
