# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md
import tempfile
from pathlib import Path
import argparse
from collections import namedtuple
from typing import Dict, Any

from PIL import Image
import numpy as np
from tinygrad import Device, GlobalCounters, dtypes, Tensor, TinyJit
from tinygrad.helpers import Timing, Context, getenv, fetch, colored, tqdm
from tinygrad.nn import Conv2d, GroupNorm
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
from extra.models.clip import Closed, Tokenizer
from extra.models.unet import UNetModel
from extra.bench_log import BenchEvent, WallTimeEvent

class AttnBlock:
  def __init__(self, in_channels):
    self.norm = GroupNorm(32, in_channels)
    self.q = Conv2d(in_channels, in_channels, 1)
    self.k = Conv2d(in_channels, in_channels, 1)
    self.v = Conv2d(in_channels, in_channels, 1)
    self.proj_out = Conv2d(in_channels, in_channels, 1)

  # copied from AttnBlock in ldm repo
  def __call__(self, x):
    h_ = self.norm(x)
    q,k,v = self.q(h_), self.k(h_), self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q,k,v = [x.reshape(b,c,h*w).transpose(1,2) for x in (q,k,v)]
    h_ = Tensor.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w)
    return x + self.proj_out(h_)

class ResnetBlock:
  def __init__(self, in_channels, out_channels=None):
    self.norm1 = GroupNorm(32, in_channels)
    self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm2 = GroupNorm(32, out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
    self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

  def __call__(self, x):
    h = self.conv1(self.norm1(x).swish())
    h = self.conv2(self.norm2(h).swish())
    return self.nin_shortcut(x) + h

class Mid:
  def __init__(self, block_in):
    self.block_1 = ResnetBlock(block_in, block_in)
    self.attn_1 = AttnBlock(block_in)
    self.block_2 = ResnetBlock(block_in, block_in)

  def __call__(self, x):
    return x.sequential([self.block_1, self.attn_1, self.block_2])

class Decoder:
  def __init__(self):
    sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
    self.conv_in = Conv2d(4,512,3, padding=1)
    self.mid = Mid(512)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[1], s[0]),
         ResnetBlock(s[0], s[0]),
         ResnetBlock(s[0], s[0])]})
      if i != 0: arr[-1]['upsample'] = {"conv": Conv2d(s[0], s[0], 3, padding=1)}
    self.up = arr

    self.norm_out = GroupNorm(32, 128)
    self.conv_out = Conv2d(128, 3, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)
    x = self.mid(x)

    for l in self.up[::-1]:
      print("decode", x.shape)
      for b in l['block']: x = b(x)
      if 'upsample' in l:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html ?
        bs,c,py,px = x.shape
        x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
        x = l['upsample']['conv'](x)
      x.realize()

    return self.conv_out(self.norm_out(x).swish())

class Encoder:
  def __init__(self):
    sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
    self.conv_in = Conv2d(3,128,3, padding=1)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[0], s[1]),
         ResnetBlock(s[1], s[1])]})
      if i != 3: arr[-1]['downsample'] = {"conv": Conv2d(s[1], s[1], 3, stride=2, padding=(0,1,0,1))}
    self.down = arr

    self.mid = Mid(512)
    self.norm_out = GroupNorm(32, 512)
    self.conv_out = Conv2d(512, 8, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)

    for l in self.down:
      print("encode", x.shape)
      for b in l['block']: x = b(x)
      if 'downsample' in l: x = l['downsample']['conv'](x)

    x = self.mid(x)
    return self.conv_out(self.norm_out(x).swish())

class AutoencoderKL:
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.quant_conv = Conv2d(8, 8, 1)
    self.post_quant_conv = Conv2d(4, 4, 1)

  def __call__(self, x):
    latent = self.encoder(x)
    latent = self.quant_conv(latent)
    latent = latent[:, 0:4]  # only the means
    print("latent", latent.shape)
    latent = self.post_quant_conv(latent)
    return self.decoder(latent)

def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
  betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
  alphas = 1.0 - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  return Tensor(alphas_cumprod)

unet_params: Dict[str,Any] = {
  "adm_in_ch": None,
  "in_ch": 4,
  "out_ch": 4,
  "model_ch": 320,
  "attention_resolutions": [4, 2, 1],
  "num_res_blocks": 2,
  "channel_mult": [1, 2, 4, 4],
  "n_heads": 8,
  "transformer_depth": [1, 1, 1, 1],
  "ctx_dim": 768,
  "use_linear": False,
}

class StableDiffusion:
  def __init__(self):
    self.alphas_cumprod = get_alphas_cumprod()
    self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model = UNetModel(**unet_params))
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(transformer = namedtuple("Transformer", ["text_model"])(text_model = Closed.ClipTextTransformer()))

  def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
    temperature = 1
    sigma_t = 0
    sqrt_one_minus_at = (1-a_t).sqrt()
    #print(a_t, a_prev, sigma_t, sqrt_one_minus_at)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

    x_prev = a_prev.sqrt() * pred_x0 + dir_xt
    return x_prev, pred_x0

  def get_model_output(self, unconditional_context, context, latent, timestep, unconditional_guidance_scale):
    # put into diffuser
    latents = self.model.diffusion_model(latent.expand(2, *latent.shape[1:]), timestep, unconditional_context.cat(context, dim=0))
    unconditional_latent, latent = latents[0:1], latents[1:2]

    e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
    return e_t

  def decode(self, x):
    x = self.first_stage_model.post_quant_conv(1/0.18215 * x)
    x = self.first_stage_model.decoder(x)

    # make image correct size and scale
    x = (x + 1.0) / 2.0
    x = x.reshape(3,512,512).permute(1,2,0).clip(0,1)*255
    return x.cast(dtypes.uint8)

  def __call__(self, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
    e_t = self.get_model_output(unconditional_context, context, latent, timestep, guidance)
    x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, alphas, alphas_prev)
    #e_t_next = get_model_output(x_prev)
    #e_t_prime = (e_t + e_t_next) / 2
    #x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t_prime, index)
    return x_prev.realize()

# ** ldm.models.autoencoder.AutoencoderKL (done!)
# 3x512x512 <--> 4x64x64 (16384)
# decode torch.Size([1, 4, 64, 64]) torch.Size([1, 3, 512, 512])
# section 4.3 of paper
# first_stage_model.encoder, first_stage_model.decoder

# ** ldm.modules.diffusionmodules.openaimodel.UNetModel
# this is what runs each time to sample. is this the LDM?
# input:  4x64x64
# output: 4x64x64
# model.diffusion_model
# it has attention?

# ** ldm.modules.encoders.modules.FrozenCLIPEmbedder
# cond_stage_model.transformer.text_model

if __name__ == "__main__":
  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps', type=int, default=6, help="Number of steps in diffusion")
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to render")
  parser.add_argument('--out', type=str, default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument('--noshow', action='store_true', help="Don't show the image")
  parser.add_argument('--fp16', action='store_true', help="Cast the weights to float16")
  parser.add_argument('--timing', action='store_true', help="Print timing per step")
  parser.add_argument('--seed', type=int, help="Set the random latent seed")
  parser.add_argument('--guidance', type=float, default=7.5, help="Prompt strength")
  args = parser.parse_args()

  model = StableDiffusion()

  # load in weights
  with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
    load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)

    if args.fp16:
      for k,v in get_state_dict(model).items():
        if k.startswith("model"):
          v.replace(v.cast(dtypes.float16).realize())

  # run through CLIP to get context
  tokenizer = Tokenizer.ClipTokenizer()
  prompt = Tensor([tokenizer.encode(args.prompt)])
  context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got CLIP context", context.shape)

  prompt = Tensor([tokenizer.encode("")])
  unconditional_context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got unconditional CLIP context", unconditional_context.shape)

  # done with clip model
  del model.cond_stage_model

  timesteps = list(range(1, 1000, 1000//args.steps))
  print(f"running for {timesteps} timesteps")
  alphas = model.alphas_cumprod[Tensor(timesteps)]
  alphas_prev = Tensor([1.0]).cat(alphas[:-1])

  # start with random noise
  if args.seed is not None: Tensor.manual_seed(args.seed)
  latent = Tensor.randn(1,4,64,64)

  @TinyJit
  def run(model, *x): return model(*x).realize()

  # this is diffusion
  with Context(BEAM=getenv("LATEBEAM")):
    for index, timestep in (t:=tqdm(list(enumerate(timesteps))[::-1])):
      GlobalCounters.reset()
      t.set_description("%3d %3d" % (index, timestep))
      with Timing("step in ", enabled=args.timing, on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
        with WallTimeEvent(BenchEvent.STEP):
          tid = Tensor([index])
          latent = run(model, unconditional_context, context, latent, Tensor([timestep]), alphas[tid], alphas_prev[tid], Tensor([args.guidance]))
          if args.timing: Device[Device.DEFAULT].synchronize()
    del run

  # upsample latent space to image with autoencoder
  x = model.decode(latent)
  print(x.shape)

  # save image
  im = Image.fromarray(x.numpy())
  print(f"saving {args.out}")
  im.save(args.out)
  # Open image.
  if not args.noshow: im.show()

  # validation!
  if args.prompt == default_prompt and args.steps == 6 and args.seed == 0 and args.guidance == 7.5:
    ref_image = Tensor(np.array(Image.open(Path(__file__).parent / "stable_diffusion_seed0.png")))
    distance = (((x.cast(dtypes.float) - ref_image.cast(dtypes.float)) / ref_image.max())**2).mean().item()
    assert distance < 3e-3, colored(f"validation failed with {distance=}", "red")  # higher distance with WINO
    print(colored(f"output validated with {distance=}", "green"))
