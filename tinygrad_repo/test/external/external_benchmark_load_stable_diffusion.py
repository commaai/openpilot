from tinygrad.helpers import fetch, Timing
from tinygrad.device import Device
from tinygrad.nn.state import torch_load, load_state_dict
from examples.stable_diffusion import StableDiffusion

# run "sudo purge" before testing on OS X to avoid the memory cache

if __name__ == "__main__":
  fn = fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt')
  model = StableDiffusion()
  with Timing():
    load_state_dict(model, torch_load(fn)['state_dict'], strict=False)
    Device[Device.DEFAULT].synchronize()
