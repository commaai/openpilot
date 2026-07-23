import math, os
if __name__ == "__main__":
  os.environ["DEFAULT_FLOAT"] = "bfloat16"
  os.environ["OPTIM_DTYPE"] = "bfloat16"
  if "DEV" not in os.environ: os.environ["DEV"] = "NULL::gfx950"
  # CDNA
  os.environ["DEVICE_IN_FUNCTION_BUG"] = "1"
  os.environ["ALL2ALL"] = "1"
  os.environ["USE_ATOMICS"] = "1"
  if "HK_FLASH_ATTENTION" not in os.environ:
    os.environ["HK_FLASH_ATTENTION"] = "1"
    if "ASM_GEMM" not in os.environ:
      os.environ["ASM_GEMM"] = "1"
from tinygrad import Tensor, nn, function, getenv, dtypes, TinyJit
from tinygrad.helpers import Timing, colored, GlobalCounters, profile_marker, round_up
from tinygrad.uop.ops import Ops, UOp
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis
from extra.llama_kernels.rmsnorm import rmsnorm
from extra.llama_kernels import FP8_MAX, local_abs_max

ASM_GEMM = getenv("ASM_GEMM", 0)
FUSED_INPUT_QUANTIZE = getenv("FUSED_INPUT_QUANTIZE", 0)
FUSED_ADD_NORM_MUL_QUANTIZE = getenv("FUSED_ADD_NORM_MUL_QUANTIZE", 0)
FUSED_SILU_W13 = getenv("FUSED_SILU_W13", 0)
SPLIT_W13 = getenv("SPLIT_W13", 0)
COLUMNWISE_WEIGHT_SCALE = getenv("COLUMNWISE_WEIGHT_SCALE", 0)
MXFP8 = getenv("MXFP8", 0)

FP8_DTYPE = dtypes.fp8e4m3
FP8_GRAD_DTYPE = dtypes.fp8e5m2

def quantize_fp8(x:Tensor, amax_state:Tensor|None=None):
  new_amax = (local_abs_max(x) if isinstance(x.device, tuple) else x.abs().max()).detach().cast(dtypes.float32)
  scale = FP8_MAX / ((amax_state if amax_state is not None else new_amax) + 1e-8)
  x_scaled = x * scale
  x_clamped = x_scaled + (x_scaled.detach().clamp(-FP8_MAX, FP8_MAX) - x_scaled.detach())  # STE
  return x_clamped.cast(FP8_DTYPE), scale.float().reciprocal(), new_amax

def matmul(x:Tensor, w:Tensor, fp8:bool=True, amax_x:Tensor|None=None, w_inv_scale:Tensor|None=None,
           x_fp8:Tensor|None=None, grad_amax_state:Tensor|None=None, next_grad_amax_state:Tensor|None=None, x_prequant_mx:tuple|None=None,
           next_amax_x:Tensor|None=None) -> tuple[Tensor,...]:
  if not fp8:
    if ASM_GEMM:
      from extra.gemm.cdna_asm_gemm import can_use_asm_gemm, asm_gemm
      if can_use_asm_gemm(x, w.T): return (asm_gemm(x, w.T),)
    return (x @ w.T,)
  assert w_inv_scale is not None, "fp8 matmul requires w_inv_scale (weights must be stored in fp8 with per-tensor scale)"
  if MXFP8:
    from extra.gemm.cdna_asm_gemm import asm_gemm, quantize_mxfp8, mx_pack, can_use_asm_gemm, _mx_block_scale
    if x_prequant_mx is not None: x_q, x_e8, x_si = x_prequant_mx       # fused producer already quantized (2d)
    else: x_q, x_e8, x_si = quantize_mxfp8(x.reshape(-1, x.shape[-1]))
    l_shape = x.shape[:-1] if x is not None else x_q.shape[:-1]
    if can_use_asm_gemm(x_q, w.T):
      out = asm_gemm(x_q, w.T, mx=True, mx_scales=(x_si, x_e8, mx_pack(w_inv_scale), w_inv_scale),
                     mx_w_stored=True).reshape(*l_shape, w.shape[0])
    else:
      x_phys = (x_q.cast(dtypes.bfloat16) * _mx_block_scale(x_e8)).reshape(*l_shape, x_q.shape[-1])
      out = x_phys @ (w.cast(dtypes.bfloat16) * _mx_block_scale(w_inv_scale)).T
    return out, x_q
  if x_fp8 is None:
    if FUSED_INPUT_QUANTIZE:
      from extra.llama_kernels.quantize_fp8_delayed import quantize_fp8_delayed
      x_fp8, _ = quantize_fp8_delayed(x, amax_x, next_amax_x, FP8_DTYPE)
    else:
      x_fp8, _, new_amax_x = quantize_fp8(x, amax_state=amax_x)
      next_amax_x.assign(new_amax_x)
  if ASM_GEMM:
    from extra.gemm.cdna_asm_gemm import can_use_asm_gemm, asm_gemm
    if can_use_asm_gemm(x_fp8, w.T):
      assert amax_x is not None
      if COLUMNWISE_WEIGHT_SCALE:
        out = asm_gemm(x_fp8, w.T, x_scale=amax_x, grad_amax_state=grad_amax_state,
                       next_grad_amax_state=next_grad_amax_state, w_post_scale=w_inv_scale)
      else:
        out = asm_gemm(x_fp8, w.T, x_scale=amax_x, w_scale=w_inv_scale, grad_amax_state=grad_amax_state,
                       next_grad_amax_state=next_grad_amax_state)
      return out, x_fp8
  return (x_fp8.dot(w.T, dtype=dtypes.float) * ((amax_x.float() + 1e-8) / FP8_MAX) * w_inv_scale).cast(dtypes.bfloat16), x_fp8

def norm_quantize_matmul(x:Tensor, norm:Tensor, w:Tensor, w_inv_scale:Tensor, eps:float, amax_x:Tensor,
                         next_amax_x:Tensor, grad_amax_state:Tensor, next_grad_amax_state:Tensor):
  if FUSED_ADD_NORM_MUL_QUANTIZE:
    from extra.llama_kernels.fused_rmsnorm_mul_quantize_fp8 import fused_rmsnorm_mul_quantize_fp8
    x_fp8, x_normed, rrms = fused_rmsnorm_mul_quantize_fp8(x, norm, amax_x, eps, FP8_DTYPE, next_amax_x)
    out, *ret = matmul(None, w, w_inv_scale=w_inv_scale, x_fp8=x_fp8, amax_x=amax_x,
                       grad_amax_state=grad_amax_state, next_grad_amax_state=next_grad_amax_state)
    return out, x_normed, rrms, ret
  x_normed, rrms = rmsnorm(x, eps)
  out, *ret = matmul(x_normed * norm, w, amax_x=amax_x, w_inv_scale=w_inv_scale, grad_amax_state=grad_amax_state,
                     next_grad_amax_state=next_grad_amax_state, next_amax_x=next_amax_x)
  return out, x_normed, rrms, ret

def add_norm_quantize_matmul(x:Tensor, residual:Tensor, norm:Tensor, w:Tensor, w_inv_scale:Tensor, eps:float, amax_x:Tensor,
                             next_amax_x:Tensor, grad_amax_state:Tensor|None=None, next_grad_amax_state:Tensor|None=None):
  if FUSED_ADD_NORM_MUL_QUANTIZE:
    from extra.llama_kernels.fused_rmsnorm_mul_quantize_fp8 import fused_add_rmsnorm_mul_quantize_fp8
    x_fp8, h, x_normed, rrms = fused_add_rmsnorm_mul_quantize_fp8(x, residual, norm, amax_x, eps, FP8_DTYPE, next_amax_x)
    out, *ret = matmul(None, w, w_inv_scale=w_inv_scale, x_fp8=x_fp8, amax_x=amax_x,
                       grad_amax_state=grad_amax_state, next_grad_amax_state=next_grad_amax_state)
    return out, h, x_normed, rrms, ret
  h = x + residual
  x_normed, rrms = rmsnorm(h, eps)
  out, *ret = matmul(x_normed * norm, w, amax_x=amax_x, w_inv_scale=w_inv_scale, grad_amax_state=grad_amax_state,
                     next_grad_amax_state=next_grad_amax_state, next_amax_x=next_amax_x)
  return out, h, x_normed, rrms, ret

def silu_w13_quantize_matmul(x_w13:Tensor, w2:Tensor, s_2:Tensor,
                             amax_x2:Tensor, next_amax_x2:Tensor,
                             grad_amax_xw13:Tensor, next_grad_amax_xw13:Tensor,
                             grad_amax_xout:Tensor, next_grad_amax_xout:Tensor):
  if FUSED_SILU_W13:
    from extra.llama_kernels.cast_amax import fused_quantize_fp8_w13
    x2_fp8 = fused_quantize_fp8_w13(x_w13, amax_x2, FP8_DTYPE, grad_amax_state=grad_amax_xw13,
                                                 next_grad_amax_state=next_grad_amax_xw13, amax_out=next_amax_x2)
    out, *ret = matmul(None, w2, w_inv_scale=s_2, x_fp8=x2_fp8, amax_x=amax_x2,
                       grad_amax_state=grad_amax_xout, next_grad_amax_state=next_grad_amax_xout)
    return out, ret
  hidden = x_w13.shape[-1] // 2
  x_w1, x_w3 = x_w13[..., :hidden], x_w13[..., hidden:]
  out, *ret = matmul(x_w1.silu() * x_w3, w2, amax_x=amax_x2, w_inv_scale=s_2, grad_amax_state=grad_amax_xout,
                     next_grad_amax_state=next_grad_amax_xout, next_amax_x=next_amax_x2)
  return out, ret

class FlatTransformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size:int, n_kv_heads:int|None=None,
               rope_theta:int=10000, max_context:int=1024):
    self.vocab_size = vocab_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.hidden_dim = hidden_dim

    scaled_std = 0.02 / math.sqrt(2 * n_layers)

    # Attention
    self.wqkv, s_qkv = self.lin_per_layer(dim, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2)
    self.wo, s_o = self.lin_per_layer(self.n_heads * self.head_dim, dim, std=scaled_std)

    # FeedForward
    if SPLIT_W13:
      self.w1, s_1 = self.lin_per_layer(dim, hidden_dim)
      self.w3, s_3 = self.lin_per_layer(dim, hidden_dim)
    else:
      self.w13, s_13 = self.lin_per_layer(dim, hidden_dim * 2)
    self.w2, s_2 = self.lin_per_layer(hidden_dim, dim, std=scaled_std)

    self.norm_eps = norm_eps
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.tok_embeddings.weight = Tensor.normal(vocab_size, dim, mean=0.0, std=0.02, dtype=dtypes.bfloat16)
    self.output = Tensor.normal(1, vocab_size, dim, mean=0.0, std=0.02, dtype=dtypes.bfloat16)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).clone().is_param_(False)

    def _amax(): return Tensor.full((), FP8_MAX, dtype=dtypes.float32).contiguous().is_param_(False)
    names = ["xqkv", "xo", "x2"]
    names += ["x1", "x3"] if SPLIT_W13 else ["x13"]
    self._fp8_amax = {name: [_amax() for _ in range(n_layers)] for name in names}
    self._fp8_next_amax = {name: [_amax() for _ in range(n_layers)] for name in names}
    grad_names = ["xqkv", "xo", "xout"]
    grad_names += ["xw1", "xw3"] if SPLIT_W13 else ["xw13"]
    self._fp8_grad_amax = {name: [_amax() for _ in range(n_layers)] for name in grad_names}
    self._fp8_next_grad_amax = {name: [_amax() for _ in range(n_layers)] for name in grad_names}
    w_scales = [("wqkv", s_qkv), ("wo", s_o), ("w2", s_2)]
    w_scales += [("w1", s_1), ("w3", s_3)] if SPLIT_W13 else [("w13", s_13)]
    self._fp8_inv_scale = {name: (s if MXFP8 else s.float()).contiguous().is_param_(False) for name, s in w_scales}
    self._fp8_next_inv_scale = {name: (s if MXFP8 else s.float()).contiguous().is_param_(False) for name, s in w_scales}

  def lin_per_layer(self, in_features:int, out_features:int, std:float=0.02, w:Tensor|None=None):
    if w is None:
      if getenv("ZEROS"): w = Tensor.zeros(self.n_layers, out_features, in_features)
      else: w = Tensor.normal(self.n_layers, out_features, in_features, mean=0.0, std=std)
    if MXFP8:
      from extra.gemm.cdna_asm_gemm import quantize_mxfp8
      w_q, w_e8, _ = quantize_mxfp8(w.reshape(self.n_layers * out_features, in_features))
      return w_q.reshape(self.n_layers, out_features, in_features), w_e8.reshape(self.n_layers, out_features, in_features // 32)
    amax = (w.abs().max(axis=2) if COLUMNWISE_WEIGHT_SCALE else w.abs().flatten(1).max(1)).detach()
    scale = FP8_MAX / (amax + 1e-8)
    inv_scale = (amax + 1e-8) / FP8_MAX
    scale_b = scale.reshape(self.n_layers, out_features, 1) if COLUMNWISE_WEIGHT_SCALE else scale.reshape(-1, 1, 1)
    return (w * scale_b).clamp(-FP8_MAX, FP8_MAX).cast(FP8_DTYPE), inv_scale

  def attention(self, x:Tensor, freqs_cis:Tensor, *, attention_norm:Tensor, wqkv:Tensor, wo:Tensor,
                amax_xqkv:Tensor, amax_xo:Tensor, s_qkv:Tensor, s_o:Tensor,
                next_amax_xqkv:Tensor, next_amax_xo:Tensor,
                grad_amax_xqkv:Tensor, grad_amax_xo:Tensor, next_grad_amax_xqkv:Tensor, next_grad_amax_xo:Tensor):
    bsz, seqlen, _ = x.shape
    saves = []

    xqkv, x_normed, rrms, s = norm_quantize_matmul(x, attention_norm, wqkv, s_qkv, self.norm_eps,
                                                                  amax_x=amax_xqkv, grad_amax_state=grad_amax_xqkv,
                                                                  next_grad_amax_state=next_grad_amax_xqkv, next_amax_x=next_amax_xqkv)
    saves.extend([x_normed, rrms, *s, xqkv])
    if getenv("HK_FLASH_ATTENTION"):
      from extra.thunder.amd.fa import flash_attention, fused_qkv_rope
      xq, xk, xv = fused_qkv_rope(xqkv, freqs_cis, self.n_heads, self.n_kv_heads, self.head_dim)
      attn, *save = flash_attention(xq, xk, xv, is_causal=True, write_flat=True)
      saves.extend(save)
    else:
      xqkv = xqkv.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep + 2, self.head_dim)
      xq = xqkv[:, :, :, :self.n_rep].reshape(bsz, seqlen, self.n_heads, self.head_dim)
      xk = xqkv[:, :, :, self.n_rep].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
      xv = xqkv[:, :, :, self.n_rep+1].reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
      xq, xk, xv = xq.cast(dtypes.bfloat16), xk.cast(dtypes.bfloat16), xv.cast(dtypes.bfloat16)
      xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
      attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True, enable_gqa=True).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)

    out, *s = matmul(attn, wo, amax_x=amax_xo, w_inv_scale=s_o, grad_amax_state=grad_amax_xo,
                               next_grad_amax_state=next_grad_amax_xo, next_amax_x=next_amax_xo)
    saves.extend([*s, out])
    return out, saves

  def feed_forward(self, x:Tensor, residual:Tensor, **kwargs):
    saves = []

    if SPLIT_W13:
      h = x + residual
      x_normed, rrms = rmsnorm(h, self.norm_eps)
      saves.extend([x_normed, rrms])
      inp = x_normed * kwargs["ffn_norm"]
      x_w1, *s = matmul(inp, kwargs["w1"], amax_x=kwargs["amax_x1"], w_inv_scale=kwargs["s_1"],
                                  grad_amax_state=kwargs["grad_amax_xw1"], next_grad_amax_state=kwargs["next_grad_amax_xw1"],
                                  next_amax_x=kwargs["next_amax_x1"])
      saves.extend([*s, x_w1])
      x_w3, *s = matmul(inp, kwargs["w3"], amax_x=kwargs["amax_x3"], w_inv_scale=kwargs["s_3"],
                                  grad_amax_state=kwargs["grad_amax_xw3"], next_grad_amax_state=kwargs["next_grad_amax_xw3"],
                                  next_amax_x=kwargs["next_amax_x3"])
      saves.extend([*s, x_w3])
      if FUSED_SILU_W13 and MXFP8:
        from extra.llama_kernels.fused_silu_mul_quantize_mxfp8 import fused_silu_mul_quantize_mxfp8
        aq, ae8, asi = fused_silu_mul_quantize_mxfp8(x_w1.reshape(-1, x_w1.shape[-1]), x_w3.reshape(-1, x_w3.shape[-1]))
        out, *s = matmul(None, kwargs["w2"], x_prequant_mx=(aq, ae8, asi), amax_x=kwargs["amax_x2"],
                         w_inv_scale=kwargs["s_2"], grad_amax_state=kwargs["grad_amax_xout"],
                         next_grad_amax_state=kwargs["next_grad_amax_xout"], next_amax_x=kwargs["next_amax_x2"])
        out = out.reshape(*x_w1.shape[:-1], kwargs["w2"].shape[0])
      else:
        out, *s = matmul(x_w1.silu() * x_w3, kwargs["w2"], amax_x=kwargs["amax_x2"], w_inv_scale=kwargs["s_2"],
                         grad_amax_state=kwargs["grad_amax_xout"], next_grad_amax_state=kwargs["next_grad_amax_xout"],
                         next_amax_x=kwargs["next_amax_x2"])
      saves.extend([*s, out])
    else:
      x_w13, h, x_normed, rrms, s = add_norm_quantize_matmul(x, residual, kwargs["ffn_norm"], kwargs["w13"], kwargs["s_13"],
                                                                          self.norm_eps, amax_x=kwargs["amax_x13"],
                                                                          next_amax_x=kwargs["next_amax_x13"],
                                                                          grad_amax_state=kwargs["grad_amax_xw13"],
                                                                          next_grad_amax_state=kwargs["next_grad_amax_xw13"])
      saves.extend([x_normed, rrms, *s, x_w13])
      out, s = silu_w13_quantize_matmul(x_w13, kwargs["w2"], kwargs["s_2"], amax_x2=kwargs["amax_x2"],
                                                     next_amax_x2=kwargs["next_amax_x2"],
                                                     grad_amax_xw13=kwargs["grad_amax_xw13"],
                                                     next_grad_amax_xw13=kwargs["next_grad_amax_xw13"],
                                                     grad_amax_xout=kwargs["grad_amax_xout"],
                                                     next_grad_amax_xout=kwargs["next_grad_amax_xout"])
      saves.extend([*s, out])
    return out, h, saves

  @function(precompile=True, precompile_backward=True)
  def run_layer(self, x:Tensor, freqs_cis:Tensor, attn_kwargs:dict, ffn_kwargs:dict, save:bool=True):
    attn, attn_saves = self.attention(x, freqs_cis, **attn_kwargs)
    ffn, h, ffn_saves = self.feed_forward(x, attn, **ffn_kwargs)
    h = h + ffn
    if save: return (h, *attn_saves, *ffn_saves)
    else: return (h,)

  def shard(self, device:tuple[str, ...], mp:bool=False):
    from tinygrad.nn.state import get_parameters
    if not mp:
      for v in get_parameters(self): v.shard_(device, axis=None)
    else:
      # flat per-layer weights: axis 0 is n_layers, so shard axes are +1 vs per-layer Transformer
      def _shard_fp8(name:str, axis:int, std:float=0.02):
        w = getattr(self, name)
        if MXFP8:
          from extra.gemm.cdna_asm_gemm import quantize_mxfp8
          w_bf16 = Tensor.empty(self.n_layers, w.shape[1], w.shape[2], dtype=dtypes.bfloat16).shard(device, axis=axis).randn_like() * std
          w_q, w_e8, _ = quantize_mxfp8(w_bf16)
          w.replace(w_q)
          self._fp8_inv_scale[name].replace(w_e8.contiguous()).is_param_(False)
          self._fp8_next_inv_scale[name].replace(w_e8.contiguous()).is_param_(False)
        else:
          w.shard_(device, axis=axis)
          scale_axis = (1 if axis == 1 else None) if COLUMNWISE_WEIGHT_SCALE else None
          self._fp8_inv_scale[name] = self._fp8_inv_scale[name].shard(device, axis=scale_axis).contiguous().is_param_(False)
          self._fp8_next_inv_scale[name] = self._fp8_next_inv_scale[name].shard(device, axis=scale_axis).contiguous().is_param_(False)
        Tensor.realize(w, self._fp8_inv_scale[name], self._fp8_next_inv_scale[name])
      sstd = 0.02 / math.sqrt(2 * self.n_layers)
      _shard_fp8("wqkv", 1)          # (n_layers, out, dim) shard out
      _shard_fp8("wo", 2, sstd)      # (n_layers, dim, in) shard in
      if SPLIT_W13:
        _shard_fp8("w1", 1)
        _shard_fp8("w3", 1)
      else:
        _shard_fp8("w13", 1)         # (n_layers, hidden*2, dim) shard out
      _shard_fp8("w2", 2, sstd)      # (n_layers, dim, hidden) shard in
      self.attention_norm.shard_(device, axis=None).realize()
      self.ffn_norm.shard_(device, axis=None).realize()
      self.norm.weight.shard_(device, axis=None).realize()
      self.tok_embeddings.weight.shard_(device, axis=0).realize()
      self.output.shard_(device, axis=1).realize()
      self.freqs_cis.shard_(device, axis=None).realize()
      for amax_dict in (self._fp8_amax, self._fp8_next_amax, self._fp8_grad_amax, self._fp8_next_grad_amax):
        for name in amax_dict:
          for i in range(len(amax_dict[name])):
            amax_dict[name][i] = amax_dict[name][i].to(device).contiguous().is_param_(False)

  def __call__(self, tokens:Tensor, save:bool=True):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)
    if not getenv("HK_FLASH_ATTENTION"): freqs_cis = freqs_cis[:, :tokens.shape[1], :, :, :]
    a, na, ga, nga, s = self._fp8_amax, self._fp8_next_amax, self._fp8_grad_amax, self._fp8_next_grad_amax, self._fp8_inv_scale
    for i in range(self.n_layers):
      attn_kwargs = dict(attention_norm=self.attention_norm[i], wqkv=self.wqkv[i], wo=self.wo[i],
                         amax_xqkv=a["xqkv"][i], amax_xo=a["xo"][i], s_qkv=s["wqkv"][i], s_o=s["wo"][i],
                         next_amax_xqkv=na["xqkv"][i], next_amax_xo=na["xo"][i],
                         grad_amax_xqkv=ga["xqkv"][i], grad_amax_xo=ga["xo"][i],
                         next_grad_amax_xqkv=nga["xqkv"][i], next_grad_amax_xo=nga["xo"][i])
      ffn_kwargs = dict(ffn_norm=self.ffn_norm[i], w2=self.w2[i],
                        amax_x2=a["x2"][i], s_2=s["w2"][i], grad_amax_xout=ga["xout"][i], next_grad_amax_xout=nga["xout"][i],
                        next_amax_x2=na["x2"][i])
      if SPLIT_W13:
        ffn_kwargs.update(w1=self.w1[i], w3=self.w3[i], amax_x1=a["x1"][i], amax_x3=a["x3"][i],
                          next_amax_x1=na["x1"][i], next_amax_x3=na["x3"][i],
                          s_1=s["w1"][i], s_3=s["w3"][i], grad_amax_xw1=ga["xw1"][i], grad_amax_xw3=ga["xw3"][i],
                          next_grad_amax_xw1=nga["xw1"][i], next_grad_amax_xw3=nga["xw3"][i])
      else:
        ffn_kwargs.update(w13=self.w13[i], amax_x13=a["x13"][i], s_13=s["w13"][i], grad_amax_xw13=ga["xw13"][i],
                          next_grad_amax_xw13=nga["xw13"][i], next_amax_x13=na["x13"][i])
      h, *_ = self.run_layer(h, freqs_cis, attn_kwargs, ffn_kwargs, save=save)

    logits = matmul(self.norm(h), self.output[0], fp8=False)[0]
    return logits

def _get_pads(uop:UOp) -> list[UOp]:
  if uop.op == Ops.ADD: return _get_pads(uop.src[0]) + _get_pads(uop.src[1])
  return [uop]

def apply_grad(grad_buf:Tensor, new_grad:UOp):
  pads = _get_pads(new_grad)
  if len(pads) <= 1:
    new_grad = new_grad.cast(grad_buf.dtype)
    grad_buf.uop = grad_buf.uop.after(grad_buf.uop.store(grad_buf.uop + new_grad))
    return
  cur = grad_buf.uop
  for pad in sorted(pads, key=lambda p: p.marg[0][0] if p.op == Ops.PAD else 0, reverse=True):
    if pad.op == Ops.PAD:
      grad_shrink = tuple([(p[0], s+p[0]) for s,p in zip(pad.src[0].shape, pad.marg)])
      buf_slice = cur.shrink(grad_shrink)
      cur = cur.after(buf_slice.store(buf_slice + pad.src[0].cast(cur.dtype)))
    else:
      cur = cur.after(cur.store(cur + pad.cast(cur.dtype)))
  grad_buf.uop = cur

if __name__ == "__main__":
  config = {}
  BS                 = config["BS"]                     = getenv("BS", 16)
  SEQLEN             = config["SEQLEN"]                 = getenv("SEQLEN", 8192)
  SMALL              = config["SMALL"]                  = getenv("SMALL", 0)

  from examples.llama3 import MODEL_PARAMS
  model_params = MODEL_PARAMS[llama_size:=getenv("LLAMA3_SIZE", "8B")]["args"]
  # vocab_size from mixtral tokenizer
  if not SMALL: model_params |= {"vocab_size": 32000}
  real_vocab_size = model_params['vocab_size']
  if (llama_layers:=getenv("LLAMA_LAYERS")) != 0: model_params["n_layers"] = llama_layers

  # pad vocab
  if (MP := getenv("MP", 1)) > 1: model_params["vocab_size"] = round_up(model_params["vocab_size"], 256 * MP)
  vocab_mask:Tensor = Tensor.arange(model_params["vocab_size"]).reshape(1, 1, -1) >= real_vocab_size

  model = FlatTransformer(**model_params, max_context=SEQLEN)

  state = nn.state.get_state_dict(model)
  print("tensor count:", len(state))

  # shard the model
  from tinygrad import Device
  is_dp = (DP := getenv("DP", 1)) > 1
  is_mp = (MP := getenv("MP", 1)) > 1
  is_sharding = is_dp or is_mp
  device_count = max(DP, MP)
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(device_count))

  model.shard(device, is_mp)

  if is_dp: vocab_mask.shard_(device, axis=None).realize()
  if is_mp: vocab_mask.shard_(device, axis=2).realize()

  # preallocate all the grad buffers and zero them out
  grad_dtype = lambda x: dtypes.bfloat16 if x.dtype in dtypes.fp8s else x.dtype
  grads = {x:x.zeros_like(dtype=grad_dtype(x)).contiguous() for x in state.values() if x.is_param}

  fp8_amax = [t for ts in model._fp8_amax.values() for t in ts]
  fp8_grad_amax = [t for ts in model._fp8_grad_amax.values() for t in ts]

  # print model size
  sz = 0
  for k,v in state.items():
    print(f"{colored(k, 'green' if v in grads else 'white'):30s} {str(v.shape):30s} {str(v.dtype):20s} {v.device}  {v.nbytes()/1e9:.2f} GB")
    sz += v.nbytes()
  print(f"total sz: {sz/1e9:.2f} GB")

  with Timing("fake data: "): tokens = Tensor.randint(BS, SEQLEN+1, low=0, high=real_vocab_size, dtype=dtypes.int)
  with Timing("realize weights/grads/data: "): Tensor.realize(*state.values(), *grads.values(), tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
  if DP > 1: tokens = tokens.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(DP)), axis=0)
  if MP > 1: tokens = tokens.shard(tuple(f"{Device.DEFAULT}:{i}" for i in range(MP)))

  @TinyJit
  def fwd_bwd(tokens:Tensor):
    with Timing("python forward: "):
      for amax_dict in (model._fp8_next_amax, model._fp8_next_grad_amax):
        for ts in amax_dict.values():
          for nxt in ts: nxt.assign(0)
      logits = model(tokens[:, :-1], save=llama_size=="8B")
      loss = vocab_mask.where(-1e9, logits).sparse_categorical_crossentropy(tokens[:, 1:])
    with Timing("python backward: "):
      for t,g in zip(grads, loss.gradient(*grads)):
        apply_grad(grads[t], g.uop)
    with Timing("run fwd_bwd: "): loss.realize(*grads.values(), *fp8_amax, *fp8_grad_amax)

  @TinyJit
  def optim_step():
    for g in grads.values(): g.assign(g.zeros_like())
    Tensor.realize(*grads.values())

  for i in range(6):
    GlobalCounters.reset()
    profile_marker(f"step {i}")
    with Timing(colored(f"*** step {i}: ", "red")):
      fwd_bwd(tokens)
      optim_step()
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
