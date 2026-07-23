import math, os, functools
if __name__ == "__main__":
  os.environ["DEFAULT_FLOAT"] = "bfloat16"
  os.environ["OPTIM_DTYPE"] = "bfloat16"
  if "DEV" not in os.environ: os.environ["DEV"] = "NULL::gfx950"
  # CDNA
  os.environ["DEVICE_IN_FUNCTION_BUG"] = "1"
  os.environ["ALL2ALL"] = "1"
  os.environ["USE_ATOMICS"] = "1"
from tinygrad import Tensor, nn, function, getenv, dtypes, TinyJit
from tinygrad.helpers import Timing, colored, GlobalCounters, profile_marker
from tinygrad.uop.ops import Ops, UOp
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis
from extra.llama_kernels.rmsnorm import rmsnorm
from extra.gemm.cdna_asm_gemm import _mx_block_scale, _mx_block_scale_3d, quantize_mxfp8

FP8_DTYPE = dtypes.fp8e4m3
FP8_MAX = 448.0
INIT_STD = 0.008

def _quant_dequant_fwd(x:Tensor) -> Tensor:
  # x (2d bf16) -> bf16 value after an mxfp8 round-trip (1x32 block scaling on the last axis)
  M, K = x.shape
  scale_K = K // 32
  amax = x.float().reshape(M, scale_K, 32).abs().max(axis=-1)
  e8 = (amax.maximum(1e-38).log2().floor() + 127).clamp(0, 254).cast(dtypes.uint8)
  qscale = (127.0 - e8.cast(dtypes.float32)).exp2().reshape(M, scale_K, 1).expand(M, scale_K, 32).reshape(M, K)
  x_fp8 = (x.float() * qscale).clamp(-FP8_MAX, FP8_MAX).cast(FP8_DTYPE).cast(dtypes.float32)
  return (x_fp8 * _mx_block_scale(e8)).cast(dtypes.bfloat16)

@functools.cache
def _quant_dequant_fwd_fxn(x_p, device):
  return _quant_dequant_fwd(Tensor(x_p, device=device))

def _quant_dequant_bwd(grad:UOp, call:UOp) -> tuple:
  return (Tensor(grad).cast(dtypes.bfloat16).uop,)

def quant_dequant_mx(x:Tensor) -> Tensor:
  fxn = _quant_dequant_fwd_fxn(x.as_param(0).uop, x.device)
  return Tensor(UOp.maketuple(fxn.uop).call(x.uop, grad_fxn=_quant_dequant_bwd).gettuple(0))

def _mx_scale(e8:Tensor) -> Tensor:
  return _mx_block_scale(e8) if e8.ndim == 2 else _mx_block_scale_3d(e8)

def _dequant_fwd(w_q:Tensor, w_scale:Tensor) -> Tensor:
  return w_q.cast(dtypes.bfloat16) * _mx_scale(w_scale)

@functools.cache
def _dequant_fwd_fxn(wq_p, ws_p, device):
  return _dequant_fwd(Tensor(wq_p, device=device), Tensor(ws_p, device=device))

def _dequant_bwd(grad:UOp, call:UOp) -> tuple:
  w_scale = Tensor(call.src[2])
  return ((Tensor(grad).cast(dtypes.bfloat16) * _mx_scale(w_scale).cast(dtypes.bfloat16)).uop, None)

def dequant_weight(w_q:Tensor, w_scale:Tensor) -> Tensor:
  fxn = _dequant_fwd_fxn(w_q.as_param(0).uop, w_scale.as_param(1).uop, w_q.device)
  call = UOp.maketuple(fxn.uop).call(w_q.uop, w_scale.uop, grad_fxn=_dequant_bwd)
  return Tensor(call.gettuple(0))

def matmul_mx(x:Tensor, w_q:Tensor, w_scale:Tensor) -> Tensor:
  l_shape = x.shape[:-1]
  x_phys = quant_dequant_mx(x.reshape(-1, x.shape[-1])).reshape(*l_shape, x.shape[-1])
  w_phys = dequant_weight(w_q, w_scale)
  return (x_phys @ w_phys.T).cast(dtypes.bfloat16)

def swiglu(x:Tensor, limit:float=7.0, alpha:float=1.702) -> Tensor:
  x_glu, x_linear = x[..., ::2], x[..., 1::2]
  x_glu = x_glu.clamp(max_=limit)
  x_linear = x_linear.clamp(-limit, limit)
  return (x_glu * (alpha * x_glu).sigmoid()) * (x_linear + 1)

class GPTOSS:
  def __init__(self, dim:int, n_layers:int, n_heads:int, n_kv_heads:int, head_dim:int, n_experts:int, experts_per_tok:int,
               intermediate_size:int, vocab_size:int, norm_eps:float=1e-5, rope_theta:int=150000, sliding_window:int=128,
               swiglu_limit:float=7.0, max_context:int=8192):
    self.dim, self.n_layers, self.n_heads, self.n_kv_heads, self.head_dim = dim, n_layers, n_heads, n_kv_heads, head_dim
    self.n_rep = n_heads // n_kv_heads
    self.n_experts, self.experts_per_tok, self.intermediate_size = n_experts, experts_per_tok, intermediate_size
    self.vocab_size, self.norm_eps, self.sliding_window, self.swiglu_limit = vocab_size, norm_eps, sliding_window, swiglu_limit
    self.sm_scale = 1.0 / math.sqrt(head_dim)

    scaled_std = INIT_STD / math.sqrt(2 * n_layers)
    q_dim, qkv_dim = n_heads * head_dim, head_dim * (n_heads + 2 * n_kv_heads)

    # attn
    self.wqkv, self.wqkv_scale = self._quant_weight(n_layers, qkv_dim, dim)
    self.wqkv_bias = Tensor.zeros(n_layers, qkv_dim, dtype=dtypes.bfloat16).contiguous()
    self.wo, self.wo_scale = self._quant_weight(n_layers, dim, q_dim, std=scaled_std)
    self.wo_bias = Tensor.zeros(n_layers, dim, dtype=dtypes.bfloat16).contiguous()
    self.sinks = Tensor.zeros(n_layers, n_heads, dtype=dtypes.bfloat16).contiguous()
    self.attention_norm = Tensor.ones(n_layers, dim).contiguous()

    # moe ffn
    self.ffn_norm = Tensor.ones(n_layers, dim).contiguous()
    self.gate = Tensor.normal(n_layers, n_experts, dim, mean=0.0, std=INIT_STD, dtype=dtypes.bfloat16)
    self.gate_bias = Tensor.zeros(n_layers, n_experts, dtype=dtypes.bfloat16).contiguous()
    self.w_gate_up, self.w_gate_up_scale = self._quant_weight(n_layers, n_experts, intermediate_size * 2, dim)
    self.w_gate_up_bias = Tensor.zeros(n_layers, n_experts, intermediate_size * 2, dtype=dtypes.bfloat16).contiguous()
    self.w_down, self.w_down_scale = self._quant_weight(n_layers, n_experts, dim, intermediate_size, std=scaled_std)
    self.w_down_bias = Tensor.zeros(n_layers, n_experts, dim, dtype=dtypes.bfloat16).contiguous()

    # output
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.tok_embeddings.weight = Tensor.normal(vocab_size, dim, mean=0.0, std=INIT_STD, dtype=dtypes.bfloat16)
    self.output = Tensor.normal(vocab_size, dim, mean=0.0, std=INIT_STD, dtype=dtypes.bfloat16)
    self.freqs_cis = precompute_freqs_cis(head_dim, max_context * 2, rope_theta).contiguous().is_param_(False)

  def _quant_weight(self, *shape:int, std:float=INIT_STD):
    w = Tensor.zeros(*shape) if getenv("ZEROS") else Tensor.normal(*shape, mean=0.0, std=std)
    w_q, w_e8, _ = quantize_mxfp8(w)
    return w_q, w_e8.is_param_(False)

  def _attn_mask(self, seqlen:int, sliding:bool, dtype) -> Tensor:
    i, j = Tensor.arange(seqlen).reshape(seqlen, 1), Tensor.arange(seqlen).reshape(1, seqlen)
    allowed = j <= i
    if sliding: allowed = allowed & (i - j < self.sliding_window)
    return allowed.where(0.0, -1e30).cast(dtype).contiguous()

  def attention(self, x:Tensor, freqs_cis:Tensor, mask:Tensor, *, attention_norm:Tensor, wqkv:Tensor, wqkv_scale:Tensor,
                wqkv_bias:Tensor, wo:Tensor, wo_scale:Tensor, wo_bias:Tensor, sinks:Tensor):
    bsz, seqlen, _ = x.shape
    x_normed, rrms = rmsnorm(x, self.norm_eps)
    qkv = matmul_mx(x_normed * attention_norm, wqkv, wqkv_scale) + wqkv_bias
    qkv = qkv.reshape(bsz, seqlen, self.n_kv_heads, self.n_rep + 2, self.head_dim)
    xq = qkv[:, :, :, :self.n_rep].reshape(bsz, seqlen, self.n_heads, self.head_dim)
    xk, xv = qkv[:, :, :, self.n_rep], qkv[:, :, :, self.n_rep + 1]
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    xq = xq.cast(dtypes.bfloat16).reshape(bsz, seqlen, self.n_kv_heads, self.n_rep, self.head_dim).permute(0, 2, 3, 1, 4)
    xk = xk.cast(dtypes.bfloat16).permute(0, 2, 1, 3).unsqueeze(2)
    xv = xv.cast(dtypes.bfloat16).permute(0, 2, 1, 3).unsqueeze(2)
    scores = (xq @ xk.transpose(-2, -1)).float() * self.sm_scale + mask
    sink = sinks.reshape(1, self.n_kv_heads, self.n_rep, 1, 1).float()
    m = scores.max(-1, keepdim=True).maximum(sink)
    e = (scores - m).exp()
    w = (e / (e.sum(-1, keepdim=True) + (sink - m).exp())).cast(dtypes.bfloat16)
    attn = (w @ xv).permute(0, 3, 1, 2, 4).reshape(bsz, seqlen, self.n_heads * self.head_dim)

    out = matmul_mx(attn, wo, wo_scale) + wo_bias
    return out, [x_normed, rrms, attn]

  def feed_forward(self, x:Tensor, *, ffn_norm:Tensor, gate:Tensor, gate_bias:Tensor,
                   w_gate_up:Tensor, w_gate_up_scale:Tensor, w_gate_up_bias:Tensor,
                   w_down:Tensor, w_down_scale:Tensor, w_down_bias:Tensor):
    x_normed, rrms = rmsnorm(x, self.norm_eps)
    inp = x_normed * ffn_norm

    logits = inp.float() @ gate.float().T + gate_bias.float()
    thresh = logits.topk(self.experts_per_tok)[0][..., -1:]
    weights = (logits >= thresh).where(logits, -float("inf")).softmax(-1)

    out = None
    for e in range(self.n_experts):
      gate_up = matmul_mx(inp, w_gate_up[e], w_gate_up_scale[e]) + w_gate_up_bias[e]
      y = (matmul_mx(swiglu(gate_up, self.swiglu_limit), w_down[e], w_down_scale[e]) + w_down_bias[e]).contiguous()
      contrib = weights[..., e:e+1].cast(y.dtype) * y
      out = contrib if out is None else out + contrib
    return out, [x_normed, rrms]

  @function(precompile=True, precompile_backward=True)
  def run_layer(self, x:Tensor, freqs_cis:Tensor, mask:Tensor, attn_kwargs:dict, ffn_kwargs:dict, save:bool=True):
    attn, attn_saves = self.attention(x, freqs_cis, mask, **attn_kwargs)
    h = x + attn
    ffn, ffn_saves = self.feed_forward(h, **ffn_kwargs)
    h = h + ffn
    if save: return (h, *attn_saves, *ffn_saves)
    return (h,)

  def shard(self, device:tuple[str, ...], mp:bool=False):
    assert not mp, "MP not supported"
    from tinygrad.nn.state import get_parameters
    for v in get_parameters(self): v.shard_(device, axis=None)
    Tensor.realize(*get_parameters(self))

  def __call__(self, tokens:Tensor, save:bool=True):
    h = self.tok_embeddings(tokens)
    bsz, seqlen = tokens.shape
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, :seqlen, :, :, :]
    mask_full = self._attn_mask(seqlen, False, dtypes.float32)
    mask_sliding = self._attn_mask(seqlen, True, dtypes.float32)
    for i in range(self.n_layers):
      attn_kwargs = dict(attention_norm=self.attention_norm[i], wqkv=self.wqkv[i], wqkv_scale=self.wqkv_scale[i],
                         wqkv_bias=self.wqkv_bias[i], wo=self.wo[i], wo_scale=self.wo_scale[i], wo_bias=self.wo_bias[i],
                         sinks=self.sinks[i])
      ffn_kwargs = dict(ffn_norm=self.ffn_norm[i], gate=self.gate[i], gate_bias=self.gate_bias[i],
                        w_gate_up=self.w_gate_up[i], w_gate_up_scale=self.w_gate_up_scale[i], w_gate_up_bias=self.w_gate_up_bias[i],
                        w_down=self.w_down[i], w_down_scale=self.w_down_scale[i], w_down_bias=self.w_down_bias[i])
      mask = mask_sliding if i % 2 == 0 else mask_full
      h, *_ = self.run_layer(h, freqs_cis, mask, attn_kwargs, ffn_kwargs, save=save)

    logits = self.norm(h) @ self.output.T
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

GPT_OSS_20B = dict(dim=2880, n_layers=24, n_heads=64, n_kv_heads=8, head_dim=64, n_experts=32, experts_per_tok=4,
                   intermediate_size=2880, vocab_size=128256, norm_eps=1e-5, rope_theta=150000, sliding_window=128,
                   swiglu_limit=7.0)

if __name__ == "__main__":
  config = {}
  BS      = config["BS"]      = getenv("BS", 16)
  SEQLEN  = config["SEQLEN"]  = getenv("SEQLEN", 8192)

  model_params = GPT_OSS_20B
  real_vocab_size = model_params["vocab_size"]
  if (layers := getenv("LAYERS")) != 0: model_params["n_layers"] = layers

  model = GPTOSS(**model_params, max_context=SEQLEN)

  state = nn.state.get_state_dict(model)
  print("tensor count:", len(state))

  from tinygrad import Device
  is_dp = (DP := getenv("DP", 1)) > 1
  device_count = DP
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(device_count))

  if is_dp: model.shard(device)

  # preallocate all the grad buffers and zero them out
  grad_dtype = lambda x: dtypes.bfloat16 if x.dtype in dtypes.fp8s else x.dtype
  grads = {x:x.zeros_like(dtype=grad_dtype(x)).contiguous() for x in state.values() if x.is_param}

  # print model size
  sz = 0
  for k,v in state.items():
    print(f"{colored(k, 'green' if v in grads else 'white'):30s} {str(v.shape):30s} {str(v.dtype):20s} {v.device}  {v.nbytes()/1e9:.2f} GB")
    sz += v.nbytes()
  print(f"total sz: {sz/1e9:.2f} GB")

  with Timing("fake data: "): tokens = Tensor.randint(BS, SEQLEN+1, low=0, high=real_vocab_size, dtype=dtypes.int)
  with Timing("realize weights/grads/data: "): Tensor.realize(*state.values(), *grads.values(), tokens)
  print("mem per device: " + ', '.join(f"{dev}: {mem/1e9:.2f} GB" for dev, mem in sorted(GlobalCounters.mem_used_per_device.items())))
  if is_dp: tokens = tokens.shard(device, axis=0)

  @TinyJit
  def fwd_bwd(tokens:Tensor):
    with Timing("python forward: "):
      logits = model(tokens[:, :-1], save=True)
      loss = logits.sparse_categorical_crossentropy(tokens[:, 1:])
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
