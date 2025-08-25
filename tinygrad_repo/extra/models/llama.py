from typing import Union, Optional, Any
import collections, math
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv, DEBUG

# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)

# matches meta, non hugging face weights
# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
  a,b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)

def repeat_kv(x:Tensor, n_rep:int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1: return x
  # NOTE: this is different from x.repeat((1, 1, n_rep, 1))
  return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class Attention:
  def __init__(self, dim, n_heads, n_kv_heads=None, max_context=0, linear=nn.Linear, qk_norm:float|None=None):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.max_context = max_context

    self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)

    self.q_norm = nn.RMSNorm(dim, qk_norm) if qk_norm is not None else None
    self.k_norm = nn.RMSNorm(dim, qk_norm) if qk_norm is not None else None

  def __call__(self, x:Tensor, start_pos:Union[Variable,int], freqs_cis:Tensor, mask:Optional[Tensor]=None) -> Tensor:
    if getenv("WQKV"):
      if not hasattr(self, 'wqkv'): self.wqkv = Tensor.cat(self.wq.weight, self.wk.weight, self.wv.weight)
      xqkv = x @ self.wqkv.T
      xq, xk, xv = xqkv.split([self.wq.weight.shape[0], self.wk.weight.shape[0], self.wv.weight.shape[0]], dim=2)
    else:
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    if self.q_norm is not None and self.k_norm is not None:
      xq = self.q_norm(xq)
      xk = self.k_norm(xk)

    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    # create kv cache
    if self.max_context:
      if not hasattr(self, "cache_kv"):
        self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
        if isinstance(x.device, tuple):
          # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
          self.cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()

      # update the cache
      assert xk.dtype == xv.dtype == self.cache_kv.dtype, f"{xk.dtype=}, {xv.dtype=}, {self.cache_kv.dtype=}"
      self.cache_kv[:, :, start_pos:start_pos+seqlen, :, :].assign(Tensor.stack(xk, xv)).realize()

      keys = self.cache_kv[0, :, 0:start_pos+seqlen, :, :]
      values = self.cache_kv[1, :, 0:start_pos+seqlen, :, :]
    else:
      assert start_pos == 0
      keys, values = xk, xv

    keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)
    return self.wo(attn)

class FeedForward:
  def __init__(self, dim:int, hidden_dim:int, linear=nn.Linear):
    self.w1 = linear(dim, hidden_dim, bias=False)
    self.w2 = linear(hidden_dim, dim, bias=False)
    self.w3 = linear(dim, hidden_dim, bias=False) # the gate in Gated Linear Unit

  def __call__(self, x:Tensor) -> Tensor:
    w1 = self.w1(x).silu()
    w3 = self.w3(x.contiguous_backward())  # this fixes a strange fusion that makes tensor cores miss
    return self.w2(w1 * w3)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, max_context:int, linear=nn.Linear,
               feed_forward=FeedForward, qk_norm=None):
    self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear, qk_norm)
    self.feed_forward = feed_forward(dim, hidden_dim, linear)
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x:Tensor, start_pos:Union[Variable,int], freqs_cis:Tensor, mask:Optional[Tensor]):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    return (h + self.feed_forward(self.ffn_norm(h))).contiguous().contiguous_backward()

# standard openai sampling
def sample(logits: Tensor, temp: float, k: int, p: float, af: float, ap: float):
  assert logits.ndim == 1, "only works on 1d tensors"
  assert 0 <= p <= 1, "p must be between 0 and 1"
  assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

  # if temperature is very low just use argmax
  if temp < 1e-6: return logits.argmax()

  logits = logits.to(Device.DEFAULT)

  # alpha sampling
  if af or ap:
    if not hasattr(sample, "alpha_counter"):
      setattr(sample, "alpha_counter", Tensor.zeros_like(logits, dtype=dtypes.int32).contiguous())
    logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

  # replace NaNs with -inf
  logits = (logits != logits).where(-float("inf"), logits)

  # softmax
  t = (logits / temp).softmax()

  counter, counter2 = Tensor.arange(t.numel(), device=logits.device).contiguous(), Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous()
  # top k
  if k:
    output, output_indices = Tensor.zeros(k, device=logits.device).contiguous(), Tensor.zeros(k, device=logits.device, dtype=dtypes.int32).contiguous()
    for i in range(k):
      t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(dtypes.default_int)
      output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
      output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
      t = (counter == t_argmax).where(0, t)

    # approximate top p
    # because we are already limited to top k elements we can do top p "without sorting"
    output_cumsum = output[::-1].cumsum()[::-1] + t.sum()
    output = (output_cumsum >= (1 - p)) * output
    output_indices = (output_cumsum >= (1 - p)) * output_indices

    # sample
    output_idx = output.multinomial()
    output_token = output_indices[output_idx]
  else:
    output_token = t.multinomial()

  # increase alpha counter
  if af or ap:
    sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

  return output_token

class Transformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size, linear=nn.Linear, embedding=nn.Embedding,
               n_kv_heads=None, rope_theta=10000, max_context=1024, jit=True, feed_forward=FeedForward, qk_norm=None, disable_kv_cache=False):
    self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, 0 if disable_kv_cache else max_context,
                                    linear, feed_forward=feed_forward, qk_norm=qk_norm) for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False) if embedding == nn.Embedding else linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, self.max_context * 2, rope_theta).contiguous().requires_grad_(False)
    self.forward_jit = TinyJit(self.forward) if jit else None

  def forward(self, tokens:Tensor, start_pos:Union[Variable,int], temperature:float, top_k:int, top_p:float, alpha_f:float, alpha_p:float):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, start_pos:start_pos+seqlen, :, :, :]

    mask = Tensor.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), dtype=h.dtype, device=h.device).triu(start_pos+1) if seqlen > 1 else None
    for layer in self.layers: h = layer(h, start_pos, freqs_cis, mask)
    logits = self.output(self.norm(h))
    if math.isnan(temperature): return logits

    return sample(logits[:, -1, :].flatten(), temperature, top_k, top_p, alpha_f, alpha_p)

  def __call__(self, tokens:Tensor, start_pos:int, temperature:float=0.0, top_k:int=0, top_p:float=0.8, alpha_f:float=0.0, alpha_p:float=0.0):
    # TODO: better way to handle the first call v.s. the rest?
    if tokens.shape[0:2] == (1,1) and self.forward_jit is not None and start_pos != 0:
      return self.forward_jit(tokens, Variable("start_pos", 1, self.max_context-1).bind(start_pos), temperature, top_k, top_p, alpha_f, alpha_p)
    return self.forward(tokens, start_pos, temperature, top_k, top_p, alpha_f, alpha_p)

# *** helpers ***

# TODO: n_kv_heads should support None
def convert_from_huggingface(weights:dict[str, Tensor], n_layers: int, n_heads: int, n_kv_heads: int, permute_layers: bool = True):
  # huggingface stores Q and K permuted! it is mostly correct without this, but without it makes RoPE different, so it will diverge after 10+ toks.
  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])

  keymap = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.{x}_norm.weight": f"layers.{l}.attention.{x}_norm.weight" for x in ["q", "k"] for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
    **{f"model.layers.{l}.self_attn.{x}_proj.bias": f"layers.{l}.attention.w{x}.bias" for x in ["q", "k", "v", "o"] for l in range(n_layers)},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(n_layers)},
    **{f"model.layers.{l}.mlp.gate.weight": f"layers.{l}.feed_forward.gate.weight" for l in range(n_layers)},
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
  }
  sd = {}
  experts = collections.defaultdict(dict)
  for k, v in weights.items():
    if ".rotary_emb." in k: continue
    v = v.to(Device.DEFAULT)
    if "model.layers" in k:
      if ("q_proj" in k or "q_norm" in k) and permute_layers: v = permute(v, n_heads)
      elif ("k_proj" in k or "k_norm" in k) and permute_layers: v = permute(v, n_kv_heads)
    if '.mlp.experts.' in k:
      # support MoE models
      _, _, layer, _, _, expert, name, _ = k.split('.')
      experts[f'layers.{layer}.feed_forward.{name}'][int(expert)] = v
      continue
    sd[keymap[k]] = v
  for k,v in experts.items(): sd[k] = Tensor.stack(*[v[i] for i in range(len(v))])
  return sd

def convert_from_gguf(weights:dict[str, Tensor], n_layers:int):
  keymap = {
    "token_embd.weight": "tok_embeddings.weight",
    **{f"blk.{l}.attn_norm.weight": f"layers.{l}.attention_norm.weight" for l in range(n_layers)},
    **{f"blk.{l}.attn_{x}.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v"] for l in range(n_layers)},
    **{f"blk.{l}.attn_output.weight": f"layers.{l}.attention.wo.weight" for l in range(n_layers)},
    **{f"blk.{l}.ffn_norm.weight": f"layers.{l}.ffn_norm.weight" for l in range(n_layers)},
    **{f"blk.{l}.ffn_{x}.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(n_layers)},
    "output_norm.weight": "norm.weight",
    "rope_freqs.weight": "rope_freqs.weight",
  }
  sd = {keymap[k]: v for k,v in weights.items()}
  sd["output.weight"] = weights["token_embd.weight"]
  return sd

def fix_bf16(weights:dict[Any, Tensor]):
  if getenv("SUPPORT_BF16", 1):
    # TODO: without casting to float16, 70B llama OOM on tinybox.
    return {k:v.cast(dtypes.float32).cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k,v in weights.items()}
  # TODO: check if device supports bf16
  return {k:v.llvm_bf16_cast(dtypes.half).to(v.device) if v.dtype == dtypes.bfloat16 else v for k,v in weights.items()}
