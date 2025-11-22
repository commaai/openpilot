import json, pprint
from tinygrad import fetch, nn, Tensor
from tinygrad.helpers import DEBUG

class FeedForward:
  def __init__(self, model_dim, intermediate_dim):
    self.proj_1 = nn.Linear(model_dim, 2*intermediate_dim, bias=False)
    self.proj_2 = nn.Linear(intermediate_dim, model_dim, bias=False)

  def __call__(self, x):
    y_12 = self.proj_1(x)
    y_1, y_2 = y_12.chunk(2, dim=-1)
    return self.proj_2(y_1.silu() * y_2)

# NOTE: this RoPE doesn't match LLaMA's?
def _rotate_half(x: Tensor) -> Tensor:
  x1, x2 = x.chunk(2, dim=-1)
  return Tensor.cat(-x2, x1, dim=-1)

def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
  return (x * pos_cos) + (_rotate_half(x) * pos_sin)

class Attention:
  def __init__(self, model_dim, num_query_heads, num_kv_heads, head_dim):
    self.qkv_proj = nn.Linear(model_dim, (num_query_heads + num_kv_heads*2) * head_dim, bias=False)
    self.num_query_heads, self.num_kv_heads = num_query_heads, num_kv_heads
    self.head_dim = head_dim
    self.q_norm = nn.RMSNorm(head_dim)
    self.k_norm = nn.RMSNorm(head_dim)
    self.out_proj = nn.Linear(num_query_heads * head_dim, model_dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    batch_size, seq_len, embed_dim = x.shape
    qkv = self.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, self.num_query_heads+self.num_kv_heads*2, self.head_dim).transpose(1, 2)
    xq,xk,xv = qkv.split([self.num_query_heads, self.num_kv_heads, self.num_kv_heads], dim=1)
    xq = self.q_norm(xq)
    xk = self.k_norm(xk)

    # add positional embedding (how many kernels is this?)
    freq_constant = 10000
    inv_freq = 1.0 / (freq_constant ** (Tensor.arange(0, self.head_dim, 2) / self.head_dim))
    pos_index_theta = Tensor.einsum("i,j->ij", Tensor.arange(seq_len), inv_freq)
    emb = Tensor.cat(pos_index_theta, pos_index_theta, dim=-1)
    cos_emb, sin_emb = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    xq = _apply_rotary_pos_emb(xq, sin_emb, cos_emb)
    xk = _apply_rotary_pos_emb(xk, sin_emb, cos_emb)

    # grouped-query attention
    num_groups = self.num_query_heads // self.num_kv_heads
    xk = xk.repeat_interleave(num_groups, dim=1)
    xv = xv.repeat_interleave(num_groups, dim=1)

    # masked attention
    #start_pos = 0
    #mask = Tensor.full((1, 1, seq_len, start_pos+seq_len), float("-inf"), dtype=xq.dtype, device=xq.device).triu(start_pos+1)
    #attn_output = xq.scaled_dot_product_attention(xk, xv, mask).transpose(1, 2)

    # causal is fine, no mask needed
    attn_output = xq.scaled_dot_product_attention(xk, xv, is_causal=True).transpose(1, 2)
    return self.out_proj(attn_output.reshape(batch_size, seq_len, self.num_query_heads * self.head_dim))

class Layer:
  def __init__(self, model_dim, intermediate_dim, num_query_heads, num_kv_heads, head_dim):
    self.ffn = FeedForward(model_dim, intermediate_dim)
    self.attn = Attention(model_dim, num_query_heads, num_kv_heads, head_dim)
    self.ffn_norm = nn.RMSNorm(model_dim)
    self.attn_norm = nn.RMSNorm(model_dim)

  def __call__(self, x:Tensor) -> Tensor: # (batch, seq_len, embed_dim)
    x = x + self.attn(self.attn_norm(x))
    x = x + self.ffn(self.ffn_norm(x))
    return x

# stupidly complex
def make_divisible(v, divisor):
  new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
  if new_v < 0.9 * v: new_v += divisor
  return new_v

class Transformer:
  def __init__(self, cfg):
    if DEBUG >= 3: pprint.pp(cfg)
    self.layers = [Layer(cfg['model_dim'], make_divisible(int(cfg["model_dim"] * cfg['ffn_multipliers'][i]), cfg['ffn_dim_divisor']),
                         cfg['num_query_heads'][i], cfg['num_kv_heads'][i], cfg['head_dim']) for i in range(cfg['num_transformer_layers'])]
    self.norm = nn.RMSNorm(cfg['model_dim'])
    self.token_embeddings = nn.Embedding(cfg['vocab_size'], cfg['model_dim'])

  def __call__(self, tokens:Tensor):
    # _bsz, seqlen = tokens.shape
    x = self.token_embeddings(tokens)
    for l in self.layers: x = l(x)
    return self.norm(x) @ self.token_embeddings.weight.T

if __name__ == "__main__":
  #model_name = "OpenELM-270M-Instruct"
  model_name = "OpenELM-270M"  # this is fp32
  model = Transformer(json.loads(fetch(f"https://huggingface.co/apple/{model_name}/resolve/main/config.json?download=true").read_bytes()))
  weights = nn.state.safe_load(fetch(f"https://huggingface.co/apple/{model_name}/resolve/main/model.safetensors?download=true"))
  if DEBUG >= 3:
    for k, v in weights.items(): print(k, v.shape)
  nn.state.load_state_dict(model, {k.removeprefix("transformer."):v for k,v in weights.items()})

  from sentencepiece import SentencePieceProcessor
  tokenizer = SentencePieceProcessor(fetch("https://github.com/karpathy/llama2.c/raw/master/tokenizer.model").as_posix())
  toks = [tokenizer.bos_id()] + tokenizer.encode("Some car brands include")
  for i in range(100):
    ttoks = Tensor([toks])
    out = model(ttoks).realize()
    t0 = out[0].argmax(axis=-1).tolist()
    toks.append(t0[-1])
    # hmmm...passthrough still doesn't match (it shouldn't, it outputs the most likely)
    print(tokenizer.decode(toks))
    #print(toks)
    #print(tokenizer.decode(t0))
    #print(t0)


