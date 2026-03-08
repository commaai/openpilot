import unittest, os
from tempfile import TemporaryDirectory
from tinygrad import Tensor
from tinygrad.helpers import getenv
from examples.mlperf.model_train import train_stable_diffusion

class TestTrain(unittest.TestCase):
  def test_train_to_ckpt(self):
    # train for num_steps, save checkpoint, and stop training
    num_steps = 42
    os.environ.update({"MODEL": "stable_diffusion", "TOTAL_CKPTS": "1", "CKPT_STEP_INTERVAL": str(num_steps), "GPUS": "8", "BS": "304"})
    # NOTE: update these based on where data/checkpoints are on your system
    if not getenv("DATADIR", ""): os.environ["DATADIR"] = "/raid/datasets/stable_diffusion"
    if not getenv("CKPTDIR", ""): os.environ["CKPTDIR"] = "/raid/weights/stable_diffusion"
    with TemporaryDirectory(prefix="test-train") as tmp:
      os.environ["UNET_CKPTDIR"] = tmp
      with Tensor.train():
        saved_ckpts = train_stable_diffusion()
      expected_ckpt = f"{tmp}/{num_steps}.safetensors"
      assert len(saved_ckpts) == 1 and saved_ckpts[0] == expected_ckpt

if __name__=="__main__":
  unittest.main()