import os, sys, math, argparse, time
sys.path.append(os.getcwd())
from typing import Any, Optional, Dict

from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict, torch_load

from tqdm import tqdm
from transformers import AutoTokenizer

MODELS = {
  "130m": {"dim":  768, "n_layers": 24, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
  "370m": {"dim": 1024, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
  "790m": {"dim": 1536, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
  "1.4b": {"dim": 2048, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
  "2.8b": {"dim": 2560, "n_layers": 64, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
}

def fetch_weights(model_name: str) -> Dict[str, Tensor]:
  if model_name not in MODELS:
    raise ValueError(f"Requested unknown mamba model: {model_name}")
  downloaded = fetch(f"https://huggingface.co/state-spaces/mamba-{model_name}/resolve/main/pytorch_model.bin?download=true")
  return torch_load(downloaded)

def selective_scan_ref(
  u,
  delta,
  A,
  B,
  C,
  D=None,
  z=None,
  delta_bias=None,
  delta_softplus=False,
  return_last_state=False,
):
  """
  u: r(B D L)
  delta: r(B D L)
  A: c(D N) or r(D N)
  B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
  C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
  D: r(D)
  z: r(B D L)
  delta_bias: r(D), fp32

  out: r(B D L)
  last_state (optional): r(B D dstate) or c(B D dstate)
  """
  u = u.float()
  delta = delta.float()
  if delta_bias is not None:
    delta = delta + delta_bias[..., None].float()
  if delta_softplus:
    delta = delta.softplus()
  batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
  is_variable_B = len(B.shape) >= 3
  is_variable_C = len(C.shape) >= 3
  x = Tensor.zeros(batch, dim, dstate)
  ys = []
  deltaA = Tensor.einsum("bdl,dn->bdln", delta, A).exp()
  if not is_variable_B:
    deltaB_u = Tensor.einsum("bdl,dn,bdl->bdln", delta, B, u)
  else:
    if len(B.shape) == 3:
      deltaB_u = Tensor.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
      B = B.repeat((1, dim // B.shape[1], 1, 1))
      deltaB_u = Tensor.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
  if is_variable_C and len(C.shape) == 4:
    C = C.repeat((1, dim // C.shape[1], 1, 1))
  last_state = None
  for i in range(u.shape[2]):
    x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
    if not is_variable_C:
      y = Tensor.einsum("bdn,dn->bd", x, C)
    else:
      if len(C.shape) == 3:
        y = Tensor.einsum("bdn,bn->bd", x, C[:, :, i])
      else:
        y = Tensor.einsum("bdn,bdn->bd", x, C[:, :, :, i])
    if i == u.shape[2] - 1:
      last_state = x
    ys.append(y)
  y = Tensor.stack(*ys, dim=2)  # (batch dim L)
  out = y if D is None else y + u * D.reshape((-1, 1))
  if z is not None:
    out = out * z.silu()
  return out if not return_last_state else (out, last_state)

class MambaMixer:
  def __init__(
    self,
    dim,
    d_state=16,
    d_conv=4,
    expand=2,
    dt_rank="auto",
    dt_min=0.001,
    dt_max=0.1,
    dt_init="random",
    dt_scale=1.0,
    dt_init_floor=1e-4,
    conv_bias=True,
    bias=False,
    layer_idx=None,
  ):
    self.dim = dim
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = self.expand * self.dim
    self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else dt_rank
    self.layer_idx = layer_idx

    self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=bias)

    self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
                            kernel_size=d_conv, groups=self.d_inner, padding=d_conv-1)

    self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
    self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = self.dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
      self.dt_proj.weight = Tensor.full(self.dt_proj.weight.shape, dt_init_std)
    elif dt_init == "random":
      self.dt_proj.weight = Tensor.uniform(self.dt_proj.weight.shape, low=-dt_init_std, high=dt_init_std)
    else:
      raise NotImplementedError

    dt = Tensor.uniform(self.d_inner, low=math.log(dt_min), high=math.log(dt_max)).exp().maximum(dt_init_floor)
    inv_dt = dt + (1 - (-dt).exp()).log()

    self.dt_proj.bias.assign(inv_dt)

    # S4D real initialization
    self.A_log = Tensor.arange(1, self.d_state+1).repeat([self.d_inner, 1]).log()

    # D "skip" parameter
    self.D = Tensor.ones(self.d_inner)  # Keep in fp32

    self.out_proj = nn.Linear(self.d_inner, self.dim, bias=bias)

  def __call__(self, hidden_states: Tensor):
    batch, seqlen, _ = hidden_states.shape

    if not hasattr(self, 'conv_state'):
      self.conv_state = Tensor.zeros(batch, self.dim * self.expand, self.d_conv).contiguous().realize()
      self.ssm_state = Tensor.zeros(batch, self.dim * self.expand, self.d_state).realize()

      xz = self.in_proj.weight @ hidden_states.permute(2,0,1).reshape(hidden_states.shape[2],hidden_states.shape[1]*hidden_states.shape[0])
      xz = xz.reshape(xz.shape[0],xz.shape[1]//seqlen, seqlen).permute(1,0,2)

      if self.in_proj.bias is not None:
        xz = xz + self.in_proj.bias.reshape((-1, 1))

      A = -self.A_log.exp()
      x, z = xz.chunk(2, dim=1)
      # Compute short convolution
      self.conv_state.assign(x[:, :, -self.d_conv :])  # Update state (B D W)
      x = self.conv1d(x)[..., :seqlen].swish()

      x_dbl = self.x_proj(x.permute(0,2,1).reshape(x.shape[0]*x.shape[2], x.shape[1]))
      dt, B, C = Tensor.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
      dt = self.dt_proj.weight @ dt.T
      dt = dt.reshape(dt.shape[0], dt.shape[1]//seqlen, seqlen).permute(1,0,2)
      B = B.reshape(B.shape[0]//seqlen, seqlen, B.shape[1]).permute(0,2,1)
      C = C.reshape(C.shape[0]//seqlen, seqlen, C.shape[1]).permute(0,2,1)

      # TODO: actually implement selective_scan_fn
      y = selective_scan_ref(x, dt, A, B, C, self.D, z=z, delta_bias=self.dt_proj.bias, delta_softplus=True,
                            return_last_state=True)

      y, last_state = y
      self.ssm_state.assign(last_state).realize()
      y = y.permute(0,2,1)
      out = self.out_proj(y)
      return out
    else:
      return self.step(hidden_states)

  def step(self, hidden_states: Tensor):
    assert hidden_states.shape[1] == 1, f"Only support decoding with 1 token at a time for now, attempted {hidden_states.shape[1]}"
    xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
    x, z = xz.chunk(2, dim=-1)  # (B D)

    # Conv step
    self.conv_state.assign(self.conv_state[:, :, 1:].cat(x.unsqueeze(-1), dim=-1).realize())
    x = (self.conv_state * self.conv1d.weight.squeeze(1)).sum(-1)
    if self.conv1d.bias is not None:
      x = x + self.conv1d.bias
    x = x.swish()

    x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
    dt = x_db[:, : self.dt_rank]
    B = x_db[:, self.dt_rank : (self.dt_rank + self.d_state)]
    C = x_db[:, (self.dt_rank + self.d_state) :]
    # Don't add dt_bias here
    dt = self.dt_proj.weight @ dt.T
    A = -self.A_log.exp()

    # SSM step
    dt = (dt + self.dt_proj.bias.unsqueeze(-1)).softplus()
    dA = Tensor.einsum("db,dn->bdn", dt, A).exp()
    dB = Tensor.einsum("db,bn->bdn", dt, B)
    self.ssm_state.assign(self.ssm_state * dA + x.unsqueeze(-1) * dB)
    y = Tensor.einsum("bdn,bn->bd", self.ssm_state, C)
    y = y + self.D * x
    y = y * z.swish()  # (B D)

    out = self.out_proj(y)
    return out.unsqueeze(1)

class MambaBlock:
  def __init__(self, dim: int, norm_eps: float = 1e-5, rms_norm: bool = True, layer_idx: Optional[int] = None):
    self.mixer = MambaMixer(dim, layer_idx=layer_idx)
    if rms_norm:
      self.norm = nn.RMSNorm(dim, norm_eps)
    else:
      raise NotImplementedError

  def __call__(self, hidden_states: Tensor, residual: Optional[Tensor] = None):
    residual = (hidden_states + residual) if residual is not None else hidden_states
    hidden_states = self.norm(residual)
    hidden_states = self.mixer(hidden_states)
    return hidden_states, residual

class MambaBackbone:
  def __init__(self, dim: int, n_layers: int, vocab_size: int, rms_norm: bool = True, norm_eps: float = 1e-5):
    self.embedding = nn.Embedding(vocab_size, dim)
    self.layers = [MambaBlock(dim, rms_norm=rms_norm, layer_idx=i) for i in range(n_layers)]
    if rms_norm:
      self.norm_f = nn.RMSNorm(dim, norm_eps)

  def __call__(self, input_ids: Tensor) -> Any:
    hidden_states = self.embedding(input_ids)
    residual = None
    for layer in self.layers:
      hidden_states, residual = layer(hidden_states, residual)

    residual = (hidden_states + residual) if residual is not None else hidden_states
    hidden_states = self.norm_f(residual)
    return hidden_states

class Mamba:
  def __init__(self, dim: int, n_layers: int, vocab_size: int, pad_vocab_size_multiple: int = 1):
    if vocab_size % pad_vocab_size_multiple != 0:
      vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

    self.backbone = MambaBackbone(dim, n_layers, vocab_size)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    self.forward_jit = TinyJit(self.forward)

  def forward(self, input_ids:Tensor):
    hidden_states = self.backbone(input_ids)
    return self.lm_head(hidden_states).realize()

  def __call__(self, input_ids):
    return self.forward(input_ids)

  @staticmethod
  def from_pretrained(model_name: str):
    weights = fetch_weights(model_name)
    model = Mamba(**MODELS[model_name])
    load_state_dict(model, weights)

    return model


def generate(model, tokenizer, prompt: str, n_tokens_to_gen: int = 10, temp: bool = 1.0, sample: bool = False, top_k: int = None):
  tks = tokenizer(prompt)["input_ids"]
  while len(tks) < 4:
    tks = [50279] + tks

  # Loading in the prompt tokens
  logits = model.forward(Tensor([tks]))[:, -1, :]
  for _ in tqdm(range(n_tokens_to_gen), desc="Speed Gen"):
    # TODO: topk
    if sample:
      tok_Tens = (logits/temp).softmax().multinomial()
    else:
      tok_Tens = logits.argmax(axis=-1).unsqueeze(0)
    tok = tok_Tens.item()
    tks.append(tok)
    logits = model.forward_jit(tok_Tens)[:, -1, :]

  output_completions = ''.join([tokenizer.decode(output) for output in tks])
  return output_completions

if __name__ == "__main__":
  ORIG_PROMPT = "Why is gravity "
  parser = argparse.ArgumentParser(description="Run Mamba in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--prompt", type=str, default="Why is gravity ", help="Prompt for LLM completion")
  parser.add_argument("--size", type=str, default="370m",
                      help=f"Size of model to use [{', '.join([k for k in MODELS.keys()])}]")
  parser.add_argument("--n_tokens", type=int, default=10, help="Number of tokens to generate")
  parser.add_argument("--sample", dest="sample", action="store_true", help="Sample flag")
  parser.add_argument("--temp", type=float, default=1.0, help="Sampling temp has to be <=1.0")
  args = parser.parse_args()

  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
  model = Mamba.from_pretrained(args.size)
  prompt = args.prompt
  num_toks = args.n_tokens
  sample = args.sample
  temp = args.temp
  s = time.time()
  tinyoutput = generate(model, tokenizer, prompt, n_tokens_to_gen=num_toks, sample=sample, temp=temp)
  print(tinyoutput)
  print('TIME: ', time.time() - s)
  TORCHOUTPUT = "Why is gravity \nso important?\nBecause it's the only"
  if ORIG_PROMPT == prompt and not sample and num_toks==10 and args.size=='370m': print('Outputs Match:', tinyoutput == TORCHOUTPUT)