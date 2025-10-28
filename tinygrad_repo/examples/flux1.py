# pip3 install sentencepiece

# This file incorporates code from the following:
# Github Name                    | License | Link
# black-forest-labs/flux         | Apache  | https://github.com/black-forest-labs/flux/tree/main/model_licenses

from tinygrad import Tensor, nn, dtypes, TinyJit
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import fetch, tqdm, colored
from sdxl import FirstStage
from extra.models.clip import FrozenClosedClipEmbedder
from extra.models.t5 import T5Embedder
import numpy as np

import math, time, argparse, tempfile
from typing import List, Dict, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

urls:dict = {
  "flux-schnell": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
  "flux-dev": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.sft",
  "ae": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors",
  "T5_1_of_2": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00001-of-00002.safetensors",
  "T5_2_of_2": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00002-of-00002.safetensors",
  "T5_tokenizer": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/tokenizer_2/spiece.model",
  "clip": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder/model.safetensors"
}

def tensor_identity(x:Tensor) -> Tensor: return x

class AutoEncoder:
  def __init__(self, scale_factor:float, shift_factor:float):
    self.decoder = FirstStage.Decoder(128, 3, 3, 16, [1, 2, 4, 4], 2, 256)
    self.scale_factor = scale_factor
    self.shift_factor = shift_factor

  def decode(self, z:Tensor) -> Tensor:
    z = z / self.scale_factor + self.shift_factor
    return self.decoder(z)

# Conditioner
class ClipEmbedder(FrozenClosedClipEmbedder):
  def __call__(self, texts:Union[str, List[str], Tensor]) -> Tensor:
    if isinstance(texts, str): texts = [texts]
    assert isinstance(texts, (list,tuple)), f"expected list of strings, got {type(texts).__name__}"
    tokens = Tensor.cat(*[Tensor(self.tokenizer.encode(text)) for text in texts], dim=0)
    return self.transformer.text_model(tokens.reshape(len(texts),-1))[:, tokens.argmax(-1)]

# https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def attention(q:Tensor, k:Tensor, v:Tensor, pe:Tensor) -> Tensor:
  q, k = apply_rope(q, k, pe)
  x = Tensor.scaled_dot_product_attention(q, k, v)
  return x.rearrange("B H L D -> B L (H D)")

def rope(pos:Tensor, dim:int, theta:int) -> Tensor:
  assert dim % 2 == 0
  scale = Tensor.arange(0, dim, 2, dtype=dtypes.float32, device=pos.device) / dim # NOTE: this is torch.float64 in reference implementation
  omega = 1.0 / (theta**scale)
  out = Tensor.einsum("...n,d->...nd", pos, omega)
  out = Tensor.stack(Tensor.cos(out), -Tensor.sin(out), Tensor.sin(out), Tensor.cos(out), dim=-1)
  out = out.rearrange("b n d (i j) -> b n d i j", i=2, j=2)
  return out.float()

def apply_rope(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor]:
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return xq_out.reshape(*xq.shape).cast(xq.dtype), xk_out.reshape(*xk.shape).cast(xk.dtype)


# https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class EmbedND:
  def __init__(self, dim:int, theta:int, axes_dim:List[int]):
    self.dim = dim
    self.theta = theta
    self.axes_dim = axes_dim

  def __call__(self, ids:Tensor) -> Tensor:
    n_axes = ids.shape[-1]
    emb = Tensor.cat(*[rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
    return emb.unsqueeze(1)

class MLPEmbedder:
  def __init__(self, in_dim:int, hidden_dim:int):
    self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
    self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.out_layer(self.in_layer(x).silu())

class QKNorm:
  def __init__(self, dim:int):
    self.query_norm = nn.RMSNorm(dim)
    self.key_norm = nn.RMSNorm(dim)

  def __call__(self, q:Tensor, k:Tensor) -> Tuple[Tensor, Tensor]:
    return self.query_norm(q), self.key_norm(k)

class SelfAttention:
  def __init__(self, dim:int, num_heads:int = 8, qkv_bias:bool = False):
    self.num_heads = num_heads
    head_dim = dim // num_heads

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.norm = QKNorm(head_dim)
    self.proj = nn.Linear(dim, dim)

  def __call__(self, x:Tensor, pe:Tensor) -> Tensor:
    qkv = self.qkv(x)
    q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k)
    x = attention(q, k, v, pe=pe)
    return self.proj(x)

@dataclass
class ModulationOut:
  shift:Tensor
  scale:Tensor
  gate:Tensor

class Modulation:
  def __init__(self, dim:int, double:bool):
    self.is_double = double
    self.multiplier = 6 if double else 3
    self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

  def __call__(self, vec:Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
    out = self.lin(vec.silu())[:, None, :].chunk(self.multiplier, dim=-1)
    return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None

class DoubleStreamBlock:
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float, qkv_bias:bool = False):
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.img_mod = Modulation(hidden_size, double=True)
    self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_mlp = [nn.Linear(hidden_size, mlp_hidden_dim, bias=True), Tensor.gelu, nn.Linear(mlp_hidden_dim, hidden_size, bias=True)]

    self.txt_mod = Modulation(hidden_size, double=True)
    self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_mlp = [nn.Linear(hidden_size, mlp_hidden_dim, bias=True), Tensor.gelu, nn.Linear(mlp_hidden_dim, hidden_size, bias=True)]

  def __call__(self, img:Tensor, txt:Tensor, vec:Tensor, pe:Tensor) -> tuple[Tensor, Tensor]:
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    assert img_mod2 is not None and txt_mod2 is not None
    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    img_q, img_k = self.img_attn.norm(img_q, img_k)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

    # run actual attention
    q = Tensor.cat(txt_q, img_q, dim=2)
    k = Tensor.cat(txt_k, img_k, dim=2)
    v = Tensor.cat(txt_v, img_v, dim=2)

    attn = attention(q, k, v, pe=pe)
    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    # calculate the img bloks
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * ((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift).sequential(self.img_mlp)

    # calculate the txt bloks
    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * ((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift).sequential(self.txt_mlp)
    return img, txt


class SingleStreamBlock:
  """
  A DiT block with parallel linear layers as described in
  https://arxiv.org/abs/2302.05442 and adapted modulation interface.
  """

  def __init__(self,hidden_size:int, num_heads:int, mlp_ratio:float=4.0, qk_scale:Optional[float]=None):
    self.hidden_dim = hidden_size
    self.num_heads = num_heads
    head_dim = hidden_size // num_heads
    self.scale = qk_scale or head_dim**-0.5

    self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
    # qkv and mlp_in
    self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
    # proj and mlp_out
    self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

    self.norm = QKNorm(head_dim)

    self.hidden_size = hidden_size
    self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    self.mlp_act = Tensor.gelu
    self.modulation = Modulation(hidden_size, double=False)

  def __call__(self, x:Tensor, vec:Tensor, pe:Tensor) -> Tensor:
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = Tensor.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k)

    # compute attention
    attn = attention(q, k, v, pe=pe)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(Tensor.cat(attn, self.mlp_act(mlp), dim=2))
    return x + mod.gate * output


class LastLayer:
  def __init__(self, hidden_size:int, patch_size:int, out_channels:int):
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation:List[Callable[[Tensor], Tensor]] = [Tensor.silu, nn.Linear(hidden_size, 2 * hidden_size, bias=True)]

  def __call__(self, x:Tensor, vec:Tensor) -> Tensor:
    shift, scale = vec.sequential(self.adaLN_modulation).chunk(2, dim=1)
    x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
    return self.linear(x)

def timestep_embedding(t:Tensor, dim:int, max_period:int=10000, time_factor:float=1000.0) -> Tensor:
  """
  Create sinusoidal timestep embeddings.
  :param t: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an (N, D) Tensor of positional embeddings.
  """
  t = time_factor * t
  half = dim // 2
  freqs = Tensor.exp(-math.log(max_period) * Tensor.arange(0, stop=half, dtype=dtypes.float32) / half).to(t.device)

  args = t[:, None].float() * freqs[None]
  embedding = Tensor.cat(Tensor.cos(args), Tensor.sin(args), dim=-1)
  if dim % 2:  embedding = Tensor.cat(*[embedding, Tensor.zeros_like(embedding[:, :1])], dim=-1)
  if Tensor.is_floating_point(t):  embedding = embedding.cast(t.dtype)
  return embedding

# https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
class Flux:
  """
  Transformer model for flow matching on sequences.
  """

  def __init__(
      self,
      guidance_embed:bool,
      in_channels:int = 64,
      vec_in_dim:int = 768,
      context_in_dim:int = 4096,
      hidden_size:int = 3072,
      mlp_ratio:float = 4.0,
      num_heads:int = 24,
      depth:int = 19,
      depth_single_blocks:int = 38,
      axes_dim:Optional[List[int]] = None,
      theta:int = 10_000,
      qkv_bias:bool = True,
      ):

    axes_dim = axes_dim or [16, 56, 56]
    self.guidance_embed = guidance_embed
    self.in_channels = in_channels
    self.out_channels = self.in_channels
    if hidden_size % num_heads != 0:
      raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")
    pe_dim = hidden_size // num_heads
    if sum(axes_dim) != pe_dim:
      raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
    self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
    self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
    self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
    self.guidance_in:Callable[[Tensor], Tensor] = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if guidance_embed else tensor_identity
    self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

    self.double_blocks = [DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias) for _ in range(depth)]
    self.single_blocks = [SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_single_blocks)]
    self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

  def __call__(self, img:Tensor, img_ids:Tensor, txt:Tensor, txt_ids:Tensor, timesteps:Tensor, y:Tensor, guidance:Optional[Tensor] = None) -> Tensor:
    if img.ndim != 3 or txt.ndim != 3:
      raise ValueError("Input img and txt tensors must have 3 dimensions.")
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256))
    if self.guidance_embed:
      if guidance is None:
        raise ValueError("Didn't get guidance strength for guidance distilled model.")
      vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)
    ids = Tensor.cat(txt_ids, img_ids, dim=1)
    pe = self.pe_embedder(ids)
    for double_block in self.double_blocks:
      img, txt = double_block(img=img, txt=txt, vec=vec, pe=pe)

    img = Tensor.cat(txt, img, dim=1)
    for single_block in self.single_blocks:
      img = single_block(img, vec=vec, pe=pe)

    img = img[:, txt.shape[1] :, ...]

    return self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

# https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py
def load_flow_model(name:str, model_path:str):
  # Loading Flux
  print("Init model")
  model = Flux(guidance_embed=(name != "flux-schnell"))
  if not model_path: model_path = fetch(urls[name])
  state_dict = {k.replace("scale", "weight"): v for k, v in safe_load(model_path).items()}
  load_state_dict(model, state_dict)
  return model

def load_T5(max_length:int=512):
  # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
  print("Init T5")
  T5 = T5Embedder(max_length, fetch(urls["T5_tokenizer"]))
  pt_1 = fetch(urls["T5_1_of_2"])
  pt_2 = fetch(urls["T5_2_of_2"])
  load_state_dict(T5.encoder, safe_load(pt_1) | safe_load(pt_2), strict=False)
  return T5

def load_clip():
  print("Init Clip")
  clip = ClipEmbedder()
  load_state_dict(clip.transformer, safe_load(fetch(urls["clip"])))
  return clip

def load_ae() -> AutoEncoder:
  # Loading the autoencoder
  print("Init AE")
  ae = AutoEncoder(0.3611, 0.1159)
  load_state_dict(ae, safe_load(fetch(urls["ae"])))
  return ae

# https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py
def prepare(T5:T5Embedder, clip:ClipEmbedder, img:Tensor, prompt:Union[str, List[str]]) -> Dict[str, Tensor]:
  bs, _, h, w = img.shape
  if bs == 1 and not isinstance(prompt, str):
    bs = len(prompt)

  img = img.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
  if img.shape[0] == 1 and bs > 1:
    img = img.expand((bs, *img.shape[1:]))

  img_ids = Tensor.zeros(h // 2, w // 2, 3).contiguous()
  img_ids[..., 1] = img_ids[..., 1] + Tensor.arange(h // 2)[:, None]
  img_ids[..., 2] = img_ids[..., 2] + Tensor.arange(w // 2)[None, :]
  img_ids = img_ids.rearrange("h w c -> 1 (h w) c")
  img_ids = img_ids.expand((bs, *img_ids.shape[1:]))

  if isinstance(prompt, str):
    prompt = [prompt]
  txt = T5(prompt).realize()
  if txt.shape[0] == 1 and bs > 1:
    txt = txt.expand((bs, *txt.shape[1:]))
  txt_ids = Tensor.zeros(bs, txt.shape[1], 3)

  vec = clip(prompt).realize()
  if vec.shape[0] == 1 and bs > 1:
    vec = vec.expand((bs, *vec.shape[1:]))

  return {"img": img, "img_ids": img_ids.to(img.device), "txt": txt.to(img.device), "txt_ids": txt_ids.to(img.device), "vec": vec.to(img.device)}


def get_schedule(num_steps:int, image_seq_len:int, base_shift:float=0.5, max_shift:float=1.15, shift:bool=True) -> List[float]:
  # extra step for zero
  step_size = -1.0 / num_steps
  timesteps = Tensor.arange(1, 0 + step_size, step_size)

  # shifting the schedule to favor high timesteps for higher signal images
  if shift:
    # estimate mu based on linear estimation between two points
    mu = 0.5 + (max_shift - base_shift) * (image_seq_len - 256) / (4096 - 256)
    timesteps = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1))
  return timesteps.tolist()

@TinyJit
def run(model, *args): return model(*args).realize()

def denoise(model, img:Tensor, img_ids:Tensor, txt:Tensor, txt_ids:Tensor, vec:Tensor, timesteps:List[float], guidance:float=4.0) -> Tensor:
  # this is ignored for schnell
  guidance_vec = Tensor((guidance,), device=img.device, dtype=img.dtype).expand((img.shape[0],))
  for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:])), "Denoising"):
    t_vec = Tensor((t_curr,), device=img.device, dtype=img.dtype).expand((img.shape[0],))
    pred = run(model, img, img_ids, txt, txt_ids, t_vec, vec, guidance_vec)
    img = img + (t_prev - t_curr) * pred

  return img

def unpack(x:Tensor, height:int, width:int) -> Tensor:
  return x.rearrange("b (h w) (c ph pw) -> b c (h ph) (w pw)", h=math.ceil(height / 16), w=math.ceil(width / 16), ph=2, pw=2)

# https://github.com/black-forest-labs/flux/blob/main/src/flux/cli.py
if __name__ == "__main__":
  default_prompt = "bananas and a can of coke"
  parser = argparse.ArgumentParser(description="Run Flux.1", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--name",       type=str,   default="flux-schnell", help="Name of the model to load")
  parser.add_argument("--model_path", type=str,   default="",             help="path of the model file")
  parser.add_argument("--width",      type=int,   default=512,            help="width of the sample in pixels (should be a multiple of 16)")
  parser.add_argument("--height",     type=int,   default=512,            help="height of the sample in pixels (should be a multiple of 16)")
  parser.add_argument("--seed",       type=int,   default=None,           help="Set a seed for sampling")
  parser.add_argument("--prompt",     type=str,   default=default_prompt, help="Prompt used for sampling")
  parser.add_argument('--out',        type=str,   default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument("--num_steps",  type=int,   default=None,           help="number of sampling steps (default 4 for schnell, 50 for guidance distilled)") #noqa:E501
  parser.add_argument("--guidance",   type=float, default=3.5,            help="guidance value used for guidance distillation")
  parser.add_argument("--output_dir", type=str,   default="output",       help="output directory")
  args = parser.parse_args()

  if args.name not in ["flux-schnell", "flux-dev"]:
    raise ValueError(f"Got unknown model name: {args.name}, chose from flux-schnell and flux-dev")

  if args.num_steps is None:
    args.num_steps = 4 if args.name == "flux-schnell" else 50

  # allow for packing and conversion to latent space
  height = 16 * (args.height // 16)
  width = 16 * (args.width // 16)

  if args.seed is None: args.seed = Tensor._seed
  else: Tensor.manual_seed(args.seed)

  print(f"Generating with seed {args.seed}:\n{args.prompt}")
  t0 = time.perf_counter()

  # prepare input noise
  x = Tensor.randn(1, 16, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16), dtype="bfloat16")

  # load text embedders
  T5 = load_T5(max_length=256 if args.name == "flux-schnell" else 512)
  clip = load_clip()

  # embed text to get inputs for model
  inp = prepare(T5, clip, x, prompt=args.prompt)
  timesteps = get_schedule(args.num_steps, inp["img"].shape[1], shift=(args.name != "flux-schnell"))

  # done with text embedders
  del T5, clip

  # load model
  model = load_flow_model(args.name, args.model_path)

  # denoise initial noise
  x = denoise(model, **inp, timesteps=timesteps, guidance=args.guidance)

  # done with model
  del model, run

  # load autoencoder
  ae = load_ae()

  # decode latents to pixel space
  x = unpack(x.float(), height, width)
  x = ae.decode(x).realize()

  t1 = time.perf_counter()
  print(f"Done in {t1 - t0:.1f}s. Saving {args.out}")

  # bring into PIL format and save
  x = x.clamp(-1, 1)
  x = x[0].rearrange("c h w -> h w c")
  x = (127.5 * (x + 1.0)).cast("uint8")

  img = Image.fromarray(x.numpy())

  img.save(args.out)

  # validation!
  if args.prompt == default_prompt and args.name=="flux-schnell" and args.seed == 0 and args.width == args.height == 512:
    ref_image = Tensor(np.array(Image.open("examples/flux1_seed0.png")))
    distance = (((x.cast(dtypes.float) - ref_image.cast(dtypes.float)) / ref_image.max())**2).mean().item()
    assert distance < 4e-3, colored(f"validation failed with {distance=}", "red")
    print(colored(f"output validated with {distance=}", "green"))