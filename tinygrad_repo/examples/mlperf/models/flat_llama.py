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

ASM_GEMM = getenv("ASM_GEMM", 0)
FUSED_SILU_W13 = getenv("FUSED_SILU_W13", 0)
SPLIT_W13 = getenv("SPLIT_W13", 0)
MXFP4 = getenv("MXFP4", 0)

def matmul(x:Tensor, w:Tensor, mxfp4:bool=bool(MXFP4), mxfp4_w:tuple[Tensor, Tensor, Tensor, Tensor]|None=None,
           x_prequant_mxfp4:tuple[Tensor|None, Tensor|None, Tensor|None, Tensor|None]|None=None) -> tuple[Tensor,...]:
  if mxfp4 or ASM_GEMM:
    from extra.gemm.cdna_asm_gemm import asm_gemm, can_use_asm_gemm
    if can_use_asm_gemm(x, w.T): return (asm_gemm(x, w.T, mxfp4=mxfp4, mxfp4_w=mxfp4_w, mxfp4_x=x_prequant_mxfp4),)
  return (x @ w.T,)

def norm_quantize_matmul(x:Tensor, norm:Tensor, w:Tensor, eps:float, mxfp4_w=None):
  x_normed, rrms = rmsnorm(x, eps)
  out, *ret = matmul(x_normed * norm, w, mxfp4_w=mxfp4_w)
  return out, x_normed, rrms, ret

def add_norm_quantize_matmul(x:Tensor, residual:Tensor, norm:Tensor, w:Tensor, eps:float, mxfp4_w=None):
  h = x + residual
  x_normed, rrms = rmsnorm(h, eps)
  out, *ret = matmul(x_normed * norm, w, mxfp4_w=mxfp4_w)
  return out, h, x_normed, rrms, ret

def silu_w13_quantize_matmul(x_w13:Tensor, w2:Tensor, mxfp4_w=None):
  if FUSED_SILU_W13 and MXFP4:
    from extra.llama_kernels.swiglu import swiglu_mxfp4
    x2, x2_mxfp4 = swiglu_mxfp4(x_w13)
    out, *ret = matmul(x2, w2, mxfp4_w=mxfp4_w, x_prequant_mxfp4=x2_mxfp4)
    return out, ret
  hidden = x_w13.shape[-1] // 2
  x_w1, x_w3 = x_w13[..., :hidden], x_w13[..., hidden:]
  out, *ret = matmul(x_w1.silu() * x_w3, w2, mxfp4_w=mxfp4_w)
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
    self.wqkv = self.lin_per_layer(dim, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2)
    self.wo = self.lin_per_layer(self.n_heads * self.head_dim, dim, std=scaled_std)

    # FeedForward
    if SPLIT_W13:
      self.w1 = self.lin_per_layer(dim, hidden_dim)
      self.w3 = self.lin_per_layer(dim, hidden_dim)
    else:
      self.w13 = self.lin_per_layer(dim, hidden_dim * 2)
    self.w2 = self.lin_per_layer(hidden_dim, dim, std=scaled_std)

    self.norm_eps = norm_eps
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.tok_embeddings.weight = Tensor.normal(vocab_size, dim, mean=0.0, std=0.02, dtype=dtypes.bfloat16)
    self.output = Tensor.normal(1, vocab_size, dim, mean=0.0, std=0.02, dtype=dtypes.bfloat16)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).clone().is_param_(False)

  def lin_per_layer(self, in_features:int, out_features:int, std:float=0.02, w:Tensor|None=None):
    if w is None:
      if getenv("ZEROS"): w = Tensor.zeros(self.n_layers, out_features, in_features)
      else: w = Tensor.normal(self.n_layers, out_features, in_features, mean=0.0, std=std)
    return w.cast(dtypes.bfloat16)

  def attention(self, x:Tensor, freqs_cis:Tensor, *, attention_norm:Tensor, wqkv:Tensor, wo:Tensor, mxfp4_wqkv=None, mxfp4_wo=None):
    bsz, seqlen, _ = x.shape
    saves = []

    xqkv, x_normed, rrms, s = norm_quantize_matmul(x, attention_norm, wqkv, self.norm_eps, mxfp4_w=mxfp4_wqkv)
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

    out, *s = matmul(attn, wo, mxfp4_w=mxfp4_wo)
    saves.extend([*s, out])
    return out, saves

  def feed_forward(self, x:Tensor, residual:Tensor, **kwargs):
    saves = []

    if SPLIT_W13:
      h = x + residual
      x_normed, rrms = rmsnorm(h, self.norm_eps)
      saves.extend([x_normed, rrms])
      inp = x_normed * kwargs["ffn_norm"]
      x_w1, *s = matmul(inp, kwargs["w1"], mxfp4_w=kwargs.get("mxfp4_w1"))
      saves.extend([*s, x_w1])
      x_w3, *s = matmul(inp, kwargs["w3"], mxfp4_w=kwargs.get("mxfp4_w3"))
      saves.extend([*s, x_w3])
      out, *s = matmul(x_w1.silu() * x_w3, kwargs["w2"], mxfp4_w=kwargs.get("mxfp4_w2"))
      saves.extend([*s, out])
    else:
      x_w13, h, x_normed, rrms, s = add_norm_quantize_matmul(x, residual, kwargs["ffn_norm"], kwargs["w13"],
                                                          self.norm_eps, mxfp4_w=kwargs.get("mxfp4_w13"))
      saves.extend([x_normed, rrms, *s, x_w13])
      out, s = silu_w13_quantize_matmul(x_w13, kwargs["w2"], mxfp4_w=kwargs.get("mxfp4_w2"))
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
      self.wqkv.shard_(device, axis=1).realize()   # (n_layers, out, dim) shard out
      self.wo.shard_(device, axis=2).realize()     # (n_layers, dim, in) shard in
      if SPLIT_W13:
        self.w1.shard_(device, axis=1).realize()
        self.w3.shard_(device, axis=1).realize()
      else:
        self.w13.shard_(device, axis=1).realize()  # (n_layers, hidden*2, dim) shard out
      self.w2.shard_(device, axis=2).realize()     # (n_layers, dim, hidden) shard in
      self.attention_norm.shard_(device, axis=None).realize()
      self.ffn_norm.shard_(device, axis=None).realize()
      self.norm.weight.shard_(device, axis=None).realize()
      self.tok_embeddings.weight.shard_(device, axis=0).realize()
      self.output.shard_(device, axis=1).realize()
      self.freqs_cis.shard_(device, axis=None).realize()

  def create_mxfp4_weight_cache(self) -> dict[str, list[tuple[Tensor, Tensor, Tensor, Tensor]]]:
    assert MXFP4
    from extra.llama_kernels.quantize_mxfp4 import quantize_mxfp4
    names = ("wqkv", "wo", "w1", "w3", "w2") if SPLIT_W13 else ("wqkv", "wo", "w13", "w2")
    return {name:[quantize_mxfp4(w, shuffle_row=True, shuffle_col=True) for w in getattr(self, name)] for name in names}

  def update_mxfp4_weight_cache(self, cache:dict[str, list[tuple[Tensor, Tensor, Tensor, Tensor]]]) -> list[Tensor]:
    from extra.llama_kernels.quantize_mxfp4 import quantize_mxfp4
    return [x for name, layers in cache.items() for weight, outputs in zip(getattr(self, name), layers)
            for x in quantize_mxfp4(weight, shuffle_row=True, shuffle_col=True, out=outputs)]

  def __call__(self, tokens:Tensor, save:bool=True,
               mxfp4_weights:dict[str, list[tuple[Tensor, Tensor, Tensor, Tensor]]]|None=None):
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)
    if not getenv("HK_FLASH_ATTENTION"): freqs_cis = freqs_cis[:, :tokens.shape[1], :, :, :]
    for i in range(self.n_layers):
      attn_kwargs = dict(attention_norm=self.attention_norm[i], wqkv=self.wqkv[i], wo=self.wo[i])
      ffn_kwargs = dict(ffn_norm=self.ffn_norm[i], w2=self.w2[i])
      if mxfp4_weights is not None:
        attn_kwargs.update(mxfp4_wqkv=mxfp4_weights["wqkv"][i], mxfp4_wo=mxfp4_weights["wo"][i])
        ffn_kwargs.update(mxfp4_w2=mxfp4_weights["w2"][i])
      if SPLIT_W13:
        ffn_kwargs.update(w1=self.w1[i], w3=self.w3[i])
        if mxfp4_weights is not None: ffn_kwargs.update(mxfp4_w1=mxfp4_weights["w1"][i], mxfp4_w3=mxfp4_weights["w3"][i])
      else:
        ffn_kwargs.update(w13=self.w13[i])
        if mxfp4_weights is not None: ffn_kwargs.update(mxfp4_w13=mxfp4_weights["w13"][i])
      h, *_ = self.run_layer(h, freqs_cis, attn_kwargs, ffn_kwargs, save=save)

    logits = matmul(self.norm(h), self.output[0], mxfp4=False)[0]
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
  grads = {x:x.zeros_like().contiguous() for x in state.values() if x.is_param}

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
      logits = model(tokens[:, :-1], save=llama_size=="8B")
      loss = vocab_mask.where(-1e9, logits).sparse_categorical_crossentropy(tokens[:, 1:])
    with Timing("python backward: "):
      for t,g in zip(grads, loss.gradient(*grads)):
        apply_grad(grads[t], g.uop)
    with Timing("run fwd_bwd: "): loss.realize(*grads.values())

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
