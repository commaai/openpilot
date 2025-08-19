from __future__ import annotations
import sys, argparse, typing, re, itertools, unicodedata
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, helpers

def gpt2_decode_vocab(voc: dict[str, int]): # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
  c2b = { chr(cp): cp for cp in itertools.chain(range(ord("!"), ord("~")+1), range(ord("¡"), ord("¬")+1), range(ord("®"), ord("ÿ")+1)) }
  c2b.update({ chr(256+off): cp for off, cp in enumerate(cp for cp in range(256) if chr(cp) not in c2b) })
  return { bytes(c2b[c] for c in tok): tid for tok, tid in voc.items() }

def get_llama_re():
  def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)).startswith(pre))
  r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
  # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
  return "(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
    f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"

class SimpleTokenizer:
  def __init__(self, pat: str, normal_tokens: dict[bytes, int], special_tokens: dict[str, int]):
    self._normal_tokens, self._special_tokens, self._pat = normal_tokens, special_tokens, re.compile(pat)
    self._tok2str = { tid: tok.encode() for tok, tid in special_tokens.items() } | { tid: tok for tok, tid in normal_tokens.items()  }
    self._special_re = re.compile("|".join(re.escape(tok) for tok in self._special_tokens.keys()) if special_tokens else r"(?!)")

  @staticmethod
  def from_gguf_kv(kv: dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    if kv["tokenizer.ggml.pre"] not in ("llama3","llama-v3","llama-bpe"): raise ValueError(f"Invalid tokenizer preset '{kv['tokenizer.ggml.pre']}'")
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = helpers.partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(get_llama_re(), gpt2_decode_vocab(dict(normal_tokens)), dict(special_tokens))

  def encode(self, text: str):
    tokens: list[int] = []
    pos = 0
    for match in self._special_re.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids: list[int]) -> str: return b''.join(self._tok2str[tid] for tid in ids).decode()
  def role(self, role:str): return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")

  def _encode_sentence(self, chunk: str): return [ tok for word in self._pat.findall(chunk) for tok in self._encode_word(word.encode()) ]
  def _encode_word(self, word: bytes):
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [word[i:i+1] for i in range(len(word))]
    while True:
      min_tid, min_idx = 2**32, -1
      for idx, (p1, p2) in enumerate(zip(parts[:-1], parts[1:])):
        tid = self._normal_tokens.get(p1 + p2, min_tid)
        if tid < min_tid: min_tid, min_idx = tid, idx
      if min_idx == -1: break
      parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx+1]] + parts[min_idx+2:]
    try: return [ self._normal_tokens[p] for p in parts ]
    except KeyError: raise RuntimeError("token not found")

def apply_rope(x:Tensor, start_pos:int|UOp, base:int=10000):
  B, H, T, Hd = x.shape
  # NOTE: this is usually in a RoPE cache, but tinygrad JIT should prune it outside the kernel
  # TODO: make it do that
  freq = base ** (-Tensor.arange(0, 1, 2/Hd, dtype='float32'))
  angles = Tensor.arange(start_pos, start_pos+T, dtype='float32')[None, None, :, None] * freq
  cos, sin = angles.cos(), angles.sin()
  x = x.reshape(B, H, T, Hd // 2, 2)    # split into pairs
  y1 = x[..., 0] * cos - x[..., 1] * sin
  y2 = x[..., 0] * sin + x[..., 1] * cos
  return Tensor.stack(y1, y2, dim=-1).reshape(B, H, T, Hd)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, max_context:int=0):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = dim // n_heads
    self.max_context  = max_context

    # --- attention projections (all linear, bias-free) ------------------
    kv_proj_out      = self.head_dim * n_kv_heads    # Llama-3 uses the same dim for K/V
    self.attn_q      = nn.Linear(dim, dim,         bias=False)
    self.attn_k      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_v      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_output = nn.Linear(dim, dim,         bias=False)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm    = nn.RMSNorm(dim, norm_eps)

    # --- feed-forward ----------------------------------------------------
    self.ffn_gate    = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_up      = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down    = nn.Linear(hidden_dim, dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)

    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)

    q = apply_rope(q, start_pos)
    k = apply_rope(k, start_pos)

    # TODO: remove these kv cache realizes
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, self.n_kv_heads, self.max_context, self.head_dim, dtype=k.dtype, device=k.device).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()  # type: ignore
    k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    attn = self.attn_output(attn)
    return x + attn

  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    gated  = self.ffn_gate(h_norm).silu() * self.ffn_up(h_norm)
    return h + self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    return self._feed_forward(self._attention(x, start_pos))

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, max_context):
    self.blk = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context) for _ in range(num_blocks)]
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    # JIT is used if T=1 and start_pos is a UOp. TODO: make this not needed by including T in the JIT and making start_pos always a UOp
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # TODO: add temperature
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return (self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward)(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = nn.state.gguf_load(gguf.to(None))

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    model = Transformer(num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'], hidden_dim=kv[f'{arch}.feed_forward_length'],
                        n_heads=kv[f'{arch}.attention.head_count'], n_kv_heads=kv[f'{arch}.attention.head_count_kv'],
                        norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'], vocab_size=len(kv['tokenizer.ggml.tokens']), max_context=max_context)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    v_start_pos = UOp.variable("start_pos", 1, self.max_context-1)
    start_pos = 0
    t = Tensor([tokens[start_pos:]], dtype="int32")
    self.forward_jit.reset()  # TODO: why is this required? root cause the issue and make it not be needed
    while len(tokens) < self.max_context:
      t = self(t, v_start_pos.bind(start_pos) if getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1 else start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id

models = {
  "1B": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "3B": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "3B_f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "8B": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--size", choices=list(models.keys()), default=list(models.keys())[0], help="Model size")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(Tensor.from_url(models[args.size]), args.max_context)

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int = kv['tokenizer.ggml.bos_token_id']
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  ids: list[int] = [bos_id]
  while 1:
    start_pos = len(ids) - 1
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + [eos_id] + tok.role("assistant")
    except EOFError:
      break
    for next_id in model.generate(ids, start_pos):
      sys.stdout.write(tok.decode([next_id]) if next_id != eos_id else "\n\n")
      sys.stdout.flush()
      if next_id == eos_id: break
