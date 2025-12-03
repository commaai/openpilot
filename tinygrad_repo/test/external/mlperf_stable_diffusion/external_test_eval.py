import unittest, os
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from tinygrad import Device, Tensor
from tinygrad.helpers import getenv, Context
from tinygrad.nn.state import safe_save, torch_load, get_parameters
from examples.mlperf.model_eval import eval_stable_diffusion, vae_decode
from examples.stable_diffusion import AutoencoderKL

def set_eval_params():
  # override these as needed from cli
  for k,v in {"MODEL": "stable_diffusion", "GPUS": "8", "EVAL_SAMPLES": "600", "CONTEXT_BS": "816", "DENOISE_BS": "600", "DECODE_BS": "384",
   "INCEPTION_BS": "560", "CLIP_BS": "240", "DATADIR": "/raid/datasets/stable_diffusion", "CKPTDIR": "/raid/weights/stable_diffusion",
   "AMD_LLVM": "0"}.items():
    os.environ[k] = getenv(k, v)

class TestEval(unittest.TestCase):
  def test_eval_ckpt(self):
    set_eval_params()
    with TemporaryDirectory(prefix="test-eval") as tmp:
      os.environ["EVAL_CKPT_DIR"] = tmp
      # NOTE Although this checkpoint has the original fully trained model from StabilityAI, we are using mlperf code that uses different
      #   GroupNorm num_groups. Therefore, eval results may not reflect eval results on the original model.
      # The purpose of using this checkpoint is to have reproducible eval outputs.
      # Eval code expects file and weight names in a specific format, as .safetensors (not .ckpt), which is why we resave the checkpoint
      sd_v2 = torch_load(Path(getenv("CKPTDIR", "")) / "sd" / "512-base-ema.ckpt")["state_dict"]
      sd_v2 = {k.replace("model.diffusion_model.", "", 1): v for k,v in sd_v2.items() if k.startswith("model.diffusion_model.")}
      safe_save(sd_v2, f"{tmp}/0.safetensors")
      clip, fid, ckpt = eval_stable_diffusion()
    assert ckpt == 0
    if Device.DEFAULT == "NULL":
      assert clip == 0
      assert fid > 0 and fid < 1000
    else:
      # observed:
      # clip=0.08369670808315277, fid=301.05236173709545 (if SEED=12345, commit=c01b2c93076e80ae6d1ebca64bb8e83a54dadba6)
      # clip=0.08415728807449341, fid=300.3710877072948 (if SEED=12345, commit=179c7fcfe132f1a6344b57c9d8cef4eded586867)
      # clip=0.0828116238117218, fid=301.241909555543 (if SEED=98765, commit=c01b2c93076e80ae6d1ebca64bb8e83a54dadba6)
      np.testing.assert_allclose(fid, 301.147, rtol=0.1, atol=0)
      np.testing.assert_allclose(clip, 0.08325, rtol=0.1, atol=0)

  # only tested on 8xMI300x system
  @unittest.skipUnless(getenv("HANG_OK"), "expected to hang")
  def test_decoder_beam_hang(self):
    set_eval_params()
    for k,v in {"BEAM": "2", "HCQDEV_WAIT_TIMEOUT_MS": "300000", "BEAM_UOPS_MAX": "8000", "BEAM_UPCAST_MAX": "256", "BEAM_LOCAL_MAX": "1024",
                "BEAM_MIN_PROGRESS": "5", "IGNORE_JIT_FIRST_BEAM": "1"}.items():
      os.environ[k] = getenv(k, v)
    with Context(BEAM=int(os.environ["BEAM"])): # necessary because helpers.py has already set BEAM=0 and cached getenv for "BEAM"
      GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 8))]
      vae = AutoencoderKL()
      for p in get_parameters(vae): p.to_(GPUS).realize()
      x = Tensor.zeros(48,4,64,64).contiguous().to(GPUS).realize()
      x.uop = x.uop.multi(0)
      for _ in range(2): vae_decode(x, vae)

if __name__=="__main__":
  unittest.main()