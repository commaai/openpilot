# This file incorporates code from the following:
# Github Name                    | License | Link
# Stability-AI/generative-models | MIT     | https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/LICENSE-CODE
# mlfoundations/open_clip        | MIT     | https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/LICENSE

from tinygrad import Tensor, TinyJit, dtypes, GlobalCounters
from tinygrad.nn import Conv2d, GroupNorm
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import fetch, trange, colored, Timing
from extra.models.clip import Embedder, FrozenClosedClipEmbedder, FrozenOpenClipEmbedder
from extra.models.unet import UNetModel, Upsample, Downsample, timestep_embedding
from extra.bench_log import BenchEvent, WallTimeEvent
from examples.stable_diffusion import ResnetBlock, Mid
import numpy as np

from typing import Dict, List, Callable, Optional, Any, Set, Tuple, Union, Type
import argparse, tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image


# configs:
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_base.yaml
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_refiner.yaml
configs: Dict = {
  "SDXL_Base": {
    "model": {"adm_in_ch": 2816, "in_ch": 4, "out_ch": 4, "model_ch": 320, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 64, "transformer_depth": [1, 2, 10], "ctx_dim": 2048, "use_linear": True},
    "conditioner": {"concat_embedders": ["original_size_as_tuple", "crop_coords_top_left", "target_size_as_tuple"]},
    "first_stage_model": {"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256},
    "denoiser": {"num_idx": 1000},
  },
  "SDXL_Refiner": {
    "model": {"adm_in_ch": 2560, "in_ch": 4, "out_ch": 4, "model_ch": 384, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4, 4], "d_head": 64, "transformer_depth": [4, 4, 4, 4], "ctx_dim": [1280, 1280, 1280, 1280], "use_linear": True},
    "conditioner": {"concat_embedders": ["original_size_as_tuple", "crop_coords_top_left", "aesthetic_score"]},
    "first_stage_model": {"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256},
    "denoiser": {"num_idx": 1000},
  }
}


def tensor_identity(x:Tensor) -> Tensor:
  return x


class DiffusionModel:
  def __init__(self, *args, **kwargs):
    self.diffusion_model = UNetModel(*args, **kwargs)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L913
class ConcatTimestepEmbedderND(Embedder):
  def __init__(self, outdim:int, input_key:str):
    self.outdim = outdim
    self.input_key = input_key

  def __call__(self, x:Union[str,List[str],Tensor]):
    assert isinstance(x, Tensor) and len(x.shape) == 2
    emb = timestep_embedding(x.flatten(), self.outdim)
    emb = emb.reshape((x.shape[0],-1))
    return emb


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L71
class Conditioner:
  OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
  KEY2CATDIM      = {"vector": 1, "crossattn": 2, "concat": 1}
  embedders: List[Embedder]

  def __init__(self, concat_embedders:List[str]):
    self.embedders = [
      FrozenClosedClipEmbedder(ret_layer_idx=11),
      FrozenOpenClipEmbedder(dims=1280, n_heads=20, layers=32, return_pooled=True),
    ]
    for input_key in concat_embedders:
      self.embedders.append(ConcatTimestepEmbedderND(256, input_key))

  def get_keys(self) -> Set[str]:
    return set(e.input_key for e in self.embedders)

  def __call__(self, batch:Dict, force_zero_embeddings:List=[]) -> Dict[str,Tensor]:
    output: Dict[str,Tensor] = {}

    for embedder in self.embedders:
      emb_out = embedder(batch[embedder.input_key])

      if isinstance(emb_out, Tensor):
        emb_out = (emb_out,)
      assert isinstance(emb_out, (list, tuple))

      for emb in emb_out:
        if embedder.input_key in force_zero_embeddings:
          emb = Tensor.zeros_like(emb)

        out_key = self.OUTPUT_DIM2KEYS[len(emb.shape)]
        if out_key in output:
          output[out_key] = Tensor.cat(output[out_key], emb, dim=self.KEY2CATDIM[out_key])
        else:
          output[out_key] = emb

    return output


class FirstStage:
  """
  Namespace for First Stage Model components
  """

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L487
  class Encoder:
    def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
      self.conv_in = Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
      in_ch_mult = (1,) + tuple(ch_mult)

      class BlockEntry:
        def __init__(self, block:List[ResnetBlock], downsample):
          self.block = block
          self.downsample = downsample
      self.down: List[BlockEntry] = []
      for i_level in range(len(ch_mult)):
        block = []
        block_in  = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult   [i_level]
        for _ in range(num_res_blocks):
          block.append(ResnetBlock(block_in, block_out))
          block_in = block_out

        downsample = tensor_identity if (i_level == len(ch_mult)-1) else Downsample(block_in)
        self.down.append(BlockEntry(block, downsample))

      self.mid = Mid(block_in)

      self.norm_out = GroupNorm(32, block_in)
      self.conv_out = Conv2d(block_in, 2*z_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, x:Tensor) -> Tensor:
      h = self.conv_in(x)
      for down in self.down:
        for block in down.block:
          h = block(h)
        h = down.downsample(h)

      h = h.sequential([self.mid.block_1, self.mid.attn_1, self.mid.block_2])
      h = h.sequential([self.norm_out,    Tensor.swish,    self.conv_out   ])
      return h


  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L604
  class Decoder:
    def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
      block_in = ch * ch_mult[-1]
      curr_res = resolution // 2 ** (len(ch_mult) - 1)
      self.z_shape = (1, z_ch, curr_res, curr_res)

      self.conv_in = Conv2d(z_ch, block_in, kernel_size=3, stride=1, padding=1)

      self.mid = Mid(block_in)

      class BlockEntry:
        def __init__(self, block:List[ResnetBlock], upsample:Callable[[Any],Any]):
          self.block = block
          self.upsample = upsample
      self.up: List[BlockEntry] = []
      for i_level in reversed(range(len(ch_mult))):
        block = []
        block_out = ch * ch_mult[i_level]
        for _ in range(num_res_blocks + 1):
          block.append(ResnetBlock(block_in, block_out))
          block_in = block_out

        upsample = tensor_identity if i_level == 0 else Upsample(block_in)
        self.up.insert(0, BlockEntry(block, upsample)) # type: ignore

      self.norm_out = GroupNorm(32, block_in)
      self.conv_out = Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z:Tensor) -> Tensor:
      h = z.sequential([self.conv_in, self.mid.block_1, self.mid.attn_1, self.mid.block_2])

      for up in self.up[::-1]:
        for block in up.block:
          h = block(h)
        h = up.upsample(h)

      h = h.sequential([self.norm_out, Tensor.swish, self.conv_out])
      return h


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L102
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L437
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L508
class FirstStageModel:
  def __init__(self, embed_dim:int=4, **kwargs):
    self.encoder = FirstStage.Encoder(**kwargs)
    self.decoder = FirstStage.Decoder(**kwargs)
    self.quant_conv = Conv2d(2*kwargs["z_ch"], 2*embed_dim, 1)
    self.post_quant_conv = Conv2d(embed_dim, kwargs["z_ch"], 1)

  def decode(self, z:Tensor) -> Tensor:
    return z.sequential([self.post_quant_conv, self.decoder])


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/discretizer.py#L42
class LegacyDDPMDiscretization:
  def __init__(self, linear_start:float=0.00085, linear_end:float=0.0120, num_timesteps:int=1000):
    self.num_timesteps = num_timesteps
    betas = np.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    self.alphas_cumprod = np.cumprod(alphas, axis=0)

  def __call__(self, n:int, flip:bool=False) -> Tensor:
    if n < self.num_timesteps:
      timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
      alphas_cumprod = self.alphas_cumprod[timesteps]
    elif n == self.num_timesteps:
      alphas_cumprod = self.alphas_cumprod
    sigmas = Tensor((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    sigmas = Tensor.cat(Tensor.zeros((1,)), sigmas)
    return sigmas if flip else sigmas.flip(axis=0) # sigmas is "pre-flipped", need to do oposite of flag


def append_dims(x:Tensor, t:Tensor) -> Tensor:
  dims_to_append = len(t.shape) - len(x.shape)
  assert dims_to_append >= 0
  return x.reshape(x.shape + (1,)*dims_to_append)


@TinyJit
def run(model, x, tms, ctx, y, c_out, add):
  return (model(x, tms, ctx, y)*c_out + add).realize()


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L19
class SDXL:
  def __init__(self, config:Dict):
    self.conditioner = Conditioner(**config["conditioner"])
    self.first_stage_model = FirstStageModel(**config["first_stage_model"])
    self.model = DiffusionModel(**config["model"])

    self.discretization = LegacyDDPMDiscretization()
    self.sigmas = self.discretization(config["denoiser"]["num_idx"], flip=True)

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L173
  def create_conditioning(self, pos_prompts:List[str], img_width:int, img_height:int, aesthetic_score:float=5.0) -> Tuple[Dict,Dict]:
    N = len(pos_prompts)
    batch_c : Dict = {
      "txt": pos_prompts,
      "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
      "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
    }
    batch_uc: Dict = {
      "txt": [""]*N,
      "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
      "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
    }
    return self.conditioner(batch_c), self.conditioner(batch_uc, force_zero_embeddings=["txt"])

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/denoiser.py#L42
  def denoise(self, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:

    def sigma_to_idx(s:Tensor) -> Tensor:
      dists = s - self.sigmas.unsqueeze(1)
      return dists.abs().argmin(axis=0).view(*s.shape)

    sigma = self.sigmas[sigma_to_idx(sigma)]
    sigma_shape = sigma.shape
    sigma = append_dims(sigma, x)

    c_out   = -sigma
    c_in    = 1 / (sigma**2 + 1.0) ** 0.5
    c_noise = sigma_to_idx(sigma.reshape(sigma_shape))

    def prep(*tensors:Tensor):
      return tuple(t.cast(dtypes.float16).realize() for t in tensors)

    return run(self.model.diffusion_model, *prep(x*c_in, c_noise, cond["crossattn"], cond["vector"], c_out, x))

  def decode(self, x:Tensor) -> Tensor:
    return self.first_stage_model.decode(1.0 / 0.13025 * x)


class Guider(ABC):
  def __init__(self, scale:float):
    self.scale = scale

  @abstractmethod
  def __call__(self, denoiser, x:Tensor, s:Tensor, c:Dict, uc:Dict) -> Tensor:
    pass

class VanillaCFG(Guider):
  def __call__(self, denoiser, x:Tensor, s:Tensor, c:Dict, uc:Dict) -> Tensor:
    c_out = {}
    for k in c:
      assert k in ["vector", "crossattn", "concat"]
      c_out[k] = Tensor.cat(uc[k], c[k], dim=0)

    x_u, x_c = denoiser(Tensor.cat(x, x), Tensor.cat(s, s), c_out).chunk(2)
    x_pred = x_u + self.scale*(x_c - x_u)
    return x_pred

class SplitVanillaCFG(Guider):
  def __call__(self, denoiser, x:Tensor, s:Tensor, c:Dict, uc:Dict) -> Tensor:
    x_u = denoiser(x, s, uc).clone().realize()
    x_c = denoiser(x, s, c)
    x_pred = x_u + self.scale*(x_c - x_u)
    return x_pred


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L21
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L287
class DPMPP2MSampler:
  def __init__(self, cfg_scale:float, guider_cls:Type[Guider]=VanillaCFG):
    self.discretization = LegacyDDPMDiscretization()
    self.guider = guider_cls(cfg_scale)

  def sampler_step(self, old_denoised:Optional[Tensor], prev_sigma:Optional[Tensor], sigma:Tensor, next_sigma:Tensor, denoiser, x:Tensor, c:Dict, uc:Dict) -> Tuple[Tensor,Tensor]:
    denoised = self.guider(denoiser, x, sigma, c, uc)

    t, t_next = sigma.log().neg(), next_sigma.log().neg()
    h = t_next - t
    r = None if prev_sigma is None else (t - prev_sigma.log().neg()) / h

    mults = [t_next.neg().exp()/t.neg().exp(), (-h).exp().sub(1)]
    if r is not None:
      mults.extend([1 + 1/(2*r), 1/(2*r)])
    mults = [append_dims(m, x) for m in mults]

    x_standard = mults[0]*x - mults[1]*denoised
    if (old_denoised is None) or (next_sigma.sum().numpy().item() < 1e-14):
      return x_standard, denoised

    denoised_d = mults[2]*denoised - mults[3]*old_denoised
    x_advanced = mults[0]*x        - mults[1]*denoised_d
    x = Tensor.where(append_dims(next_sigma, x) > 0.0, x_advanced, x_standard)
    return x, denoised

  def __call__(self, denoiser, x:Tensor, c:Dict, uc:Dict, num_steps:int, timing=False) -> Tensor:
    sigmas = self.discretization(num_steps).to(x.device)
    x *= Tensor.sqrt(1.0 + sigmas[0] ** 2.0)
    num_sigmas = len(sigmas)

    old_denoised = None
    for i in trange(num_sigmas - 1):
      with Timing("step in ", enabled=timing, on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
        GlobalCounters.reset()
        with WallTimeEvent(BenchEvent.STEP):
          x, old_denoised = self.sampler_step(
            old_denoised=old_denoised,
            prev_sigma=(None if i==0 else sigmas[i-1].expand(x.shape[0])),
            sigma=sigmas[i].expand(x.shape[0]),
            next_sigma=sigmas[i+1].expand(x.shape[0]),
            denoiser=denoiser,
            x=x,
            c=c,
            uc=uc,
          )
          x.realize(old_denoised)

    return x


if __name__ == "__main__":
  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description="Run SDXL", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps',    type=int,   default=10, help="The number of diffusion steps")
  parser.add_argument('--prompt',   type=str,   default=default_prompt, help="Description of image to generate")
  parser.add_argument('--out',      type=str,   default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument('--seed',     type=int,   help="Set the random latent seed")
  parser.add_argument('--guidance', type=float, default=6.0, help="Prompt strength")
  parser.add_argument('--width',    type=int,   default=1024, help="The output image width")
  parser.add_argument('--height',   type=int,   default=1024, help="The output image height")
  parser.add_argument('--weights',  type=str,   help="Custom path to weights")
  parser.add_argument('--timing',   action='store_true', help="Print timing per step")
  parser.add_argument('--noshow',   action='store_true', help="Don't show the image")
  parser.add_argument('--fakeweights',  action='store_true', help="Load fake weights")
  args = parser.parse_args()

  if args.seed is not None:
    Tensor.manual_seed(args.seed)

  model = SDXL(configs["SDXL_Base"])

  if not args.fakeweights:
    default_weight_url = 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors'
    weights = args.weights if args.weights else fetch(default_weight_url, 'sd_xl_base_1.0.safetensors')
    loaded_weights = load_state_dict(model, safe_load(weights), strict=False, verbose=False, realize=False)

    start_mem_used = GlobalCounters.mem_used
    with Timing("loaded weights in ", lambda et_ns: f", {(B:=(GlobalCounters.mem_used-start_mem_used))/1e9:.2f} GB loaded at {B/et_ns:.2f} GB/s"):
      with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
        Tensor.realize(*loaded_weights)
      del loaded_weights

  N = 1
  C = 4
  F = 8

  assert args.width  % F == 0, f"img_width must be multiple of {F}, got {args.width}"
  assert args.height % F == 0, f"img_height must be multiple of {F}, got {args.height}"

  c, uc = model.create_conditioning([args.prompt], args.width, args.height)
  del model.conditioner
  Tensor.realize(*c.values(), *uc.values())
  print("created batch")

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L101
  shape = (N, C, args.height // F, args.width // F)
  randn = Tensor.randn(shape)

  sampler = DPMPP2MSampler(args.guidance)
  z = sampler(model.denoise, randn, c, uc, args.steps, timing=args.timing)
  print("created samples")
  x = model.decode(z).realize()
  print("decoded samples")

  # make image correct size and scale
  x = (x + 1.0) / 2.0
  x = x.reshape(3,args.height,args.width).permute(1,2,0).clip(0,1).mul(255).cast(dtypes.uint8)
  print(x.shape)

  im = Image.fromarray(x.numpy())
  print(f"saving {args.out}")
  im.save(args.out)

  if not args.noshow:
    im.show()

  # validation!
  if args.prompt == default_prompt and args.steps == 10 and args.seed == 0 and args.guidance == 6.0 and args.width == args.height == 1024 \
    and not args.weights:
    ref_image = Tensor(np.array(Image.open(Path(__file__).parent / "sdxl_seed0.png")))
    distance = (((x.cast(dtypes.float) - ref_image.cast(dtypes.float)) / ref_image.max())**2).mean().item()
    assert distance < 4e-3, colored(f"validation failed with {distance=}", "red")
    print(colored(f"output validated with {distance=}", "green"))
