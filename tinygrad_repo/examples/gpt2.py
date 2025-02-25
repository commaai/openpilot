#!/usr/bin/env python3
import os, argparse, contextlib
from typing import Optional, Union
with contextlib.suppress(ImportError): import tiktoken
from tinygrad import Tensor, TinyJit, Device, GlobalCounters, Variable, dtypes
from tinygrad.ops import UOp
from tinygrad.helpers import Timing, DEBUG, JIT, getenv, fetch, colored, trange
from tinygrad.nn import Embedding, Linear, LayerNorm
from tinygrad.nn.state import gguf_load, torch_load, load_state_dict, get_state_dict

MAX_CONTEXT = getenv("MAX_CONTEXT", 128)
HALF = getenv("HALF")

class Attention:
  def __init__(self, dim, n_heads):
    self.c_attn = Linear(dim, 3*dim, bias=True)
    self.c_proj = Linear(dim, dim, bias=True)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads

  def __call__(self, x:Tensor, start_pos:Variable, mask:Optional[Tensor]) -> Tensor:
    if mask is not None or start_pos.val == 0:
      # no symbolic shape qkv when consuming prompts
      start_pos = start_pos.val

    if HALF: x = x.half()
    xqkv = self.c_attn(x)
    xq, xk, xv = [xqkv.shrink((None, None, (i*self.dim, (i+1)*self.dim))).reshape(None, None, self.n_heads, self.head_dim) for i in range(3)]
    bsz, seqlen, _, _ = xq.shape

    # create kv cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, bsz, MAX_CONTEXT, self.n_heads, self.head_dim, dtype=x.dtype).contiguous().realize()

    # update the cache
    self.cache_kv.shrink((None, None,(start_pos,start_pos+seqlen),None,None)).assign(Tensor.stack(xk, xv)).realize()

    if start_pos > 0:
      keys = self.cache_kv[0].shrink((None, (0, start_pos+seqlen), None, None))
      values = self.cache_kv[1].shrink((None, (0, start_pos+seqlen), None, None))
    else:
      keys = xk
      values = xv

    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    return self.c_proj(xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, self.dim))

class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.c_fc = Linear(dim, hidden_dim, bias=True)
    self.c_proj = Linear(hidden_dim, dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.c_proj(self.c_fc(x).gelu())

class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim, n_heads)
    self.mlp = FeedForward(dim, 4*dim)
    self.ln_1 = LayerNorm(dim, norm_eps)
    self.ln_2 = LayerNorm(dim, norm_eps)

  def __call__(self, x:Tensor, start_pos:Variable, mask:Optional[Tensor]):
    h = x + self.attn(self.ln_1(x), start_pos, mask).float()
    return (h + self.mlp(self.ln_2(h)))

class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.vocab_size = vocab_size
    self.wte = Embedding(vocab_size, dim)
    self.wpe = Embedding(max_seq_len, dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
    self.ln_f = LayerNorm(dim, norm_eps)
    self.lm_head = Linear(dim, vocab_size, bias=False)
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Union[Tensor,UOp], start_pos:Variable, temperature:float=0.0):
    if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
    if isinstance(tokens, UOp):
      seqlen = 1
      tok_emb = self.wte.weight.shrink(((tokens, tokens+1), None))
    else:
      seqlen = tokens.shape[1]
      tok_emb = self.wte(tokens)

    pos_emb = self.wpe(self.allpos.shrink((None, (start_pos, start_pos+seqlen))))
    h = tok_emb + pos_emb

    if HALF: h = h.half()

    mask = Tensor.full((1, 1, seqlen, start_pos.val+seqlen), float("-inf"), dtype=h.dtype).triu(start_pos.val+1) if seqlen > 1 else None

    for hi in self.h: h = hi(h, start_pos, mask)

    logits = self.lm_head(self.ln_f(h))

    if logits.shape[1] == 0:
      # special case for empty prompt
      logits = Tensor.ones((logits.shape[0], self.vocab_size), dtype=logits.dtype, device=logits.device)
    else:
      logits = logits[:, -1, :]

    if temperature < 1e-6:
      ret = logits.argmax(-1)
    else:
      ret = (logits / temperature).softmax().multinomial()
    return ret.flatten().realize()

  def __call__(self, tokens:Union[Tensor,UOp], start_pos:Variable, temperature:float=0.0) -> Tensor:
    forward = (self.forward_jit if JIT and (isinstance(tokens, UOp) or tokens.shape[1] == 1) else self.forward)
    return forward(tokens, start_pos, temperature)

VOCAB_SIZE = 50257
MODEL_PARAMS = {
  'gpt2':         dict(n_layers=12, n_heads=12, dim=768, norm_eps=1e-5, vocab_size=VOCAB_SIZE),   # 124M params
  'gpt2-medium':  dict(n_layers=24, n_heads=16, dim=1024, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 350M params
  'gpt2-large':   dict(n_layers=36, n_heads=20, dim=1280, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 774M params
  'gpt2-xl':      dict(n_layers=48, n_heads=25, dim=1600, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 1558M params
}

class GPT2:
  @staticmethod
  def build(model_size="gpt2"):
    tokenizer = tiktoken.get_encoding("gpt2")

    model = Transformer(**MODEL_PARAMS[model_size])
    weights = torch_load(fetch(f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'))
    # special treatment for the Conv1D weights we need to transpose
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
    for k in weights:
      if k.endswith(transposed):
        weights[k] = weights[k].T
    # lm head and wte are tied
    weights['lm_head.weight'] = weights['wte.weight']

    load_state_dict(model, weights)

    if HALF:
      for l in get_state_dict(model).values():
        l.replace(l.half().realize())

    return GPT2(model, tokenizer)

  @staticmethod
  def build_gguf(model_size: str):
    q_type = model_size[len("gpt2_gguf_"):].upper()
    fn = fetch(f"https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.{q_type}.gguf?download=true")
    gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
    kv_data, state_dict = gguf_load(gguf_tensor)

    gpt2_params = {
      "dim": kv_data["gpt2.embedding_length"], "n_heads": kv_data["gpt2.attention.head_count"],
      "n_layers": kv_data["gpt2.block_count"], "norm_eps": kv_data["gpt2.attention.layer_norm_epsilon"],
      "vocab_size": VOCAB_SIZE, "max_seq_len": kv_data["gpt2.context_length"],
    }
    def _remap_gguf_key(key: str):
      replaces = [
        ("blk.", "h."), (".attn_qkv.bias", ".attn.c_attn.bias"), (".attn_qkv.weight", ".attn.c_attn.weight"),
        (".ffn_norm.bias", ".ln_2.bias"), (".ffn_norm.weight", ".ln_2.weight"), (".attn_norm.bias", ".ln_1.bias"),
        (".attn_norm.weight", ".ln_1.weight"), (".attn_output.bias", ".attn.c_proj.bias"), (".attn_output.weight", ".attn.c_proj.weight"),
        (".ffn_up.bias", ".mlp.c_fc.bias"), (".ffn_up.weight", ".mlp.c_fc.weight"), (".ffn_down.bias", ".mlp.c_proj.bias"),
        (".ffn_down.weight", ".mlp.c_proj.weight"), ("token_embd.weight", "wte.weight"), ("output.weight", "lm_head.weight"),
        ("output_norm.bias", "ln_f.bias"), ("output_norm.weight", "ln_f.weight"), ("position_embd.weight", "wpe.weight"),
      ]
      for ostr, ns in replaces: key = key.replace(ostr, ns)
      return key
    state_dict = { _remap_gguf_key(k): v for k, v in state_dict.items() }
    model = Transformer(**gpt2_params)
    load_state_dict(model, state_dict)
    return GPT2(model, tiktoken.get_encoding("gpt2"))

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def generate(self, prompt:str, max_length:int, temperature:float, timing:bool=False, batch_size:int=1):
    prompt_tokens = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    toks = [prompt_tokens[:] for _ in range(batch_size)]
    start_pos = 0
    for _ in trange(max_length, disable=(timing==True)):
      GlobalCounters.reset()
      if timing: print("")
      st = GlobalCounters.time_sum_s
      with Timing("ran model in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                  f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                  (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=timing):
        if batch_size == 1 and len(toks[0][start_pos:]) == 1:
          tokens = Variable("tokens", 0, VOCAB_SIZE).bind(toks[0][start_pos])
        else:
          tokens = Tensor([x[start_pos:] for x in toks])
        tok = self.model(tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT).bind(start_pos), temperature).tolist()
      start_pos = len(toks[0])
      for i,t in enumerate(tok): toks[i].append(t)
    return [self.tokenizer.decode(x) for x in toks]

# **** main code ****

if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")
  default_prompt = "What is the answer to life, the universe, and everything?"

  parser = argparse.ArgumentParser(description='Run GPT2 in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to start with")
  parser.add_argument('--count', type=int, default=100, help="Max number of tokens to generate")
  parser.add_argument('--temperature', type=float, default=0.8, help="Temperature in the softmax")
  parser.add_argument('--model_size', type=str, default="gpt2-medium", help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--seed', type=int, help="Set the random seed")
  parser.add_argument('--batch_size', type=int, default=1, help="Set the input batch size")
  parser.add_argument('--benchmark', type=int, default=-1, help="Benchmark GPT with the given number of tokens")
  parser.add_argument('--noshow', action='store_true', help="Don't show the output")
  args = parser.parse_args()

  if args.seed is not None:
    Tensor.manual_seed(args.seed)

  print(f"using {args.model_size}")
  gpt2 = GPT2.build_gguf(args.model_size) if args.model_size.startswith("gpt2_gguf_") else GPT2.build(args.model_size)

  if args.benchmark != -1:
    gpt2.model(Tensor.rand(args.batch_size, args.benchmark), Variable("a", 0, MAX_CONTEXT).bind(0)).realize()
  else:
    texts = gpt2.generate(args.prompt, args.count, args.temperature, timing=args.timing, batch_size=args.batch_size)
    if not args.noshow:
      print('Generating text...')
      if len(texts) == 1: print(texts[0])
      else:
        for i,text in enumerate(texts): print(colored(f"Response {i}:", "green"), text)

    # validate output!
    if args.temperature == 0 and args.model_size == "gpt2-medium" and args.count == 10:
      expected = {
        default_prompt: "What is the answer to life, the universe, and everything?\n\nThe answer is that we are all one",
        "Hello.": "Hello. I'm a little late to the party, but",
      }
      try:
        assert texts[0] == expected[args.prompt]
        print(colored("output validated", "green"))
      except KeyError:
        pass
