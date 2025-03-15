from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from examples.stable_diffusion import AutoencoderKL, get_alphas_cumprod
from examples.sdxl import DPMPP2MSampler, append_dims, LegacyDDPMDiscretization
from extra.models.unet import UNetModel
from extra.models.clip import FrozenOpenClipEmbedder

from typing import Dict
import argparse, tempfile, os
from pathlib import Path
from PIL import Image

class DiffusionModel:
  def __init__(self, model:UNetModel):
    self.diffusion_model = model

@TinyJit
def run(model, x, tms, ctx, c_out, add):
  return (model(x, tms, ctx)*c_out + add).realize()

# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L521
class StableDiffusionV2:
  def __init__(self, unet_config:Dict, cond_stage_config:Dict, parameterization:str="v"):
    self.model             = DiffusionModel(UNetModel(**unet_config))
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model  = FrozenOpenClipEmbedder(**cond_stage_config)
    self.alphas_cumprod    = get_alphas_cumprod()
    self.parameterization  = parameterization

    self.discretization = LegacyDDPMDiscretization()
    self.sigmas = self.discretization(1000, flip=True)

  def denoise(self, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:

    def sigma_to_idx(s:Tensor) -> Tensor:
      dists = s - self.sigmas.unsqueeze(1)
      return dists.abs().argmin(axis=0).view(*s.shape)

    sigma = self.sigmas[sigma_to_idx(sigma)]
    sigma_shape = sigma.shape
    sigma = append_dims(sigma, x)

    c_skip = 1.0 / (sigma**2 + 1.0)
    c_out = -sigma / (sigma**2 + 1.0) ** 0.5
    c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
    c_noise = sigma_to_idx(sigma.reshape(sigma_shape))

    def prep(*tensors:Tensor):
      return tuple(t.cast(dtypes.float16).realize() for t in tensors)

    return run(self.model.diffusion_model, *prep(x*c_in, c_noise, cond["crossattn"], c_out, x*c_skip))

  def decode(self, x:Tensor, height:int, width:int) -> Tensor:
    x = self.first_stage_model.post_quant_conv(1/0.18215 * x)
    x = self.first_stage_model.decoder(x)

    # make image correct size and scale
    x = (x + 1.0) / 2.0
    x = x.reshape(3,height,width).permute(1,2,0).clip(0,1).mul(255).cast(dtypes.uint8)
    return x

params: Dict = {
  "unet_config": {
    "adm_in_ch": None,
    "in_ch": 4,
    "out_ch": 4,
    "model_ch": 320,
    "attention_resolutions": [4, 2, 1],
    "num_res_blocks": 2,
    "channel_mult": [1, 2, 4, 4],
    "d_head": 64,
    "transformer_depth": [1, 1, 1, 1],
    "ctx_dim": 1024,
    "use_linear": True,
  },
  "cond_stage_config": {
    "dims": 1024,
    "n_heads": 16,
    "layers": 24,
    "return_pooled": False,
    "ln_penultimate": True,
  }
}

if __name__ == "__main__":
  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description='Run Stable Diffusion v2.X', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps',       type=int,   default=10, help="The number of diffusion steps")
  parser.add_argument('--prompt',      type=str,   default=default_prompt, help="Description of image to generate")
  parser.add_argument('--out',         type=str,   default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument('--seed',        type=int,   help="Set the random latent seed")
  parser.add_argument('--guidance',    type=float, default=7.5, help="Prompt strength")
  parser.add_argument('--width',       type=int,   default=768, help="The output image width")
  parser.add_argument('--height',      type=int,   default=768, help="The output image height")
  parser.add_argument('--weights-fn',  type=str,   help="Filename of weights to use")
  parser.add_argument('--weights-url', type=str,   help="Custom URL to download weights from")
  parser.add_argument('--timing',      action='store_true', help="Print timing per step")
  parser.add_argument('--noshow',      action='store_true', help="Don't show the image")
  parser.add_argument('--fp16',        action='store_true', help="Cast the weights to float16")
  args = parser.parse_args()

  N = 1
  C = 4
  F = 8
  assert args.width  % F == 0, f"img_width must be multiple of {F}, got {args.width}"
  assert args.height % F == 0, f"img_height must be multiple of {F}, got {args.height}"

  Tensor.no_grad = True
  if args.seed is not None:
    Tensor.manual_seed(args.seed)

  model = StableDiffusionV2(**params)

  default_weights_url = 'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors'
  weights_fn = args.weights_fn
  if not weights_fn:
    weights_url = args.weights_url if args.weights_url else default_weights_url
    weights_fn  = fetch(weights_url, os.path.basename(str(weights_url)))
  load_state_dict(model, safe_load(weights_fn), strict=False)

  if args.fp16:
    for k,v in get_state_dict(model).items():
      if k.startswith("model"):
        v.replace(v.cast(dtypes.float16).realize())

  c  = { "crossattn": model.cond_stage_model(args.prompt) }
  uc = { "crossattn": model.cond_stage_model("") }
  del model.cond_stage_model
  print("created conditioning")

  shape = (N, C, args.height // F, args.width // F)
  randn = Tensor.randn(shape)

  sampler = DPMPP2MSampler(args.guidance)
  z = sampler(model.denoise, randn, c, uc, args.steps, timing=args.timing)
  print("created samples")
  x = model.decode(z, args.height, args.width).realize()
  print("decoded samples")
  print(x.shape)

  im = Image.fromarray(x.numpy())
  print(f"saving {args.out}")
  im.save(args.out)

  if not args.noshow:
    im.show()
