import unittest
import numpy as np
from pathlib import Path
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_parameters
from extra.models import clip
from examples.mlperf.initializers import gelu_erf, init_stable_diffusion, attn_f32_softmax
from typing import Literal

clip_params = {"dims": 1024, "n_heads": 16, "layers": 24, "return_pooled": False, "ln_penultimate": True, "clip_tokenizer_version": "sd_mlperf_v5_0"}
def get_cond_stage_model(GPUS:list[str]|None=None) -> clip.FrozenOpenClipEmbedder:
  clip.gelu = gelu_erf
  model = clip.FrozenOpenClipEmbedder(**clip_params)
  if GPUS and len(GPUS) > 1:
    for p in get_parameters(model): p.to_(GPUS)
  return model
def get_tokens(BS:int) -> Tensor: return Tensor([0] * 77 * BS, dtype=dtypes.int32).reshape(-1, 77)

class TestOpenClip(unittest.TestCase):
  def test_tokenizer(self):
    prompt = "Beautiful is better than ugly.\nExplicit is better than implicit.\nSimple is better than complex.\nComplex is better than complicated."
    model = get_cond_stage_model()
    tokens = model.tokenizer.encode(prompt, pad_with_zeros=True)
    expected = [49406, 1215, 533, 1539, 1126, 8159, 269, 33228, 533, 1539, 1126, 15269, 585, 269, 4129, 533, 1539, 1126, 6324, 269, 6324, 533,
                1539, 1126, 16621, 269, 49407] + [0]*50
    self.assertEqual(tokens, expected)

  def test_clip_gelu_init(self):
    for resblock in get_cond_stage_model().model.transformer.resblocks:
      self.assertEqual(resblock.mlp.gelu, gelu_erf)

  def test_multigpu_clip_embed(self):
    BS = 304
    GPUS = [f"{Device.DEFAULT}:{i}" for i in range(8)]
    model = get_cond_stage_model(GPUS)
    tokens = get_tokens(BS)
    embeds = model.embed_tokens(tokens.shard(GPUS, axis=0)).realize()
    self.assertEqual(embeds.shape, (BS, 77, 1024))
    self.assertEqual(embeds.dtype, dtypes.float32)

  def test_multigpu_clip_score(self):
    BS = 240
    GPUS = [f"{Device.DEFAULT}:{i}" for i in range(8)]
    vision_cfg = {'width': 1280, 'layers': 32, 'd_head': 80, 'image_size': 224, 'patch_size': 14}
    text_cfg = {'width': 1024, 'n_heads': 16, 'layers': 24, 'vocab_size': 49408, 'ctx_length': 77}
    clip.gelu = gelu_erf
    clip_encoder = clip.OpenClipEncoder(1024, text_cfg, vision_cfg)
    for p in get_parameters(clip_encoder): p.to_(GPUS)
    tokens = get_tokens(BS)
    imgs = Tensor.zeros(BS,3,224,224).contiguous()
    scores = clip_encoder.get_clip_score(tokens.shard(GPUS, axis=0), imgs.shard(GPUS, axis=0)).realize()
    self.assertEqual(scores.shape, (BS,))
    self.assertEqual(scores.dtype, dtypes.float32)

class TestInitStableDiffusion(unittest.TestCase):
  def setUp(self):
    # NOTE: set env variable based on where checkpoints are on the system
    self.CKPTDIR = Path(getenv("CKPTDIR", "/raid/weights/stable_diffusion"))

  def helper_test_init(self, version:Literal["v2-mlperf-train", "v2-mlperf-eval"]):
    model, unet, sqrt_acp, sqrt_omacp = init_stable_diffusion(version, self.CKPTDIR / "sd" / "512-base-ema.ckpt", ["CPU"])

    with self.subTest("test that StableDiffusion has correct models"):
      self.assertEqual(model.model.diffusion_model, unet)
      has_encoder = True if version=="v2-mlperf-eval" else False
      self.assertEqual(hasattr(model, "first_stage_model"), has_encoder, "only the eval model uses the encoder")
      self.assertTrue(isinstance(model.cond_stage_model, clip.FrozenOpenClipEmbedder))

    with self.subTest("test for mlperf unique attributes"):
      self.assertEqual(model.cond_stage_model.tokenizer.version, 'sd_mlperf_v5_0')
      self.assertEqual(unet.out[0].num_groups, 16)
      self.assertEqual(unet.input_blocks[1][1].norm.eps, 1e-6)
      self.assertEqual(unet.input_blocks[1][1].transformer_blocks[0].attn1.attn, attn_f32_softmax)

    with self.subTest("test loaded clip parameters"):
      sample = model.cond_stage_model.model.transformer.resblocks[8].mlp.c_fc.bias.flatten()[42:46].numpy()
      expected = np.array([-0.49812260270118713, -0.3039605915546417, -0.40284937620162964, -0.45069342851638794], dtype=np.float32)
      np.testing.assert_allclose(sample, expected, rtol=1e-7, atol=0, err_msg="loaded clip parameters are incorrect")

    if version=="v2-mlperf-train":
      with self.subTest("test that zero_module worked"):
        self.assertTrue((unet.out[2].weight == 0).all().item(), "expected all zeroes")
        self.assertTrue((unet.out[2].bias == 0).all().item(), "expected all zeroes")
    elif version=="v2-mlperf-eval":
      with self.subTest("test loaded vae parameters"):
        sample = model.first_stage_model.decoder.up[0]['block'][1].conv2.weight.flatten()[42:46].numpy()
        expected = np.array([0.08192943036556244, 0.040095631033182144, 0.07541035860776901, 0.1475081741809845], dtype=np.float32)
        np.testing.assert_allclose(sample, expected, rtol=1e-7, atol=0, err_msg="loaded vae parameters are incorrect")

    with self.subTest("check schedules"):
      expected = np.array([0.9995748996734619, 0.06826484948396683], dtype=np.float32)
      np.testing.assert_allclose(sqrt_acp[[0,-1]].numpy(), expected, rtol=1e-7, atol=0, err_msg="sqrt_acp is incorrect")
      expected = np.array([0.029155133292078972, 0.9976672530174255], dtype=np.float32)
      np.testing.assert_allclose(sqrt_omacp[[0,-1]].numpy(), expected, rtol=1e-7, atol=0, err_msg="sqrt_omacp is incorrect")

    with self.subTest("check mixed precision"):
      out = unet.input_blocks[2][1].proj_in(Tensor.randn(320, dtype=dtypes.float32))
      self.assertEqual(out.dtype, dtypes.bfloat16, "expected float32 to be downcast to bfloat16 by Linear")
      out = unet.out[2](Tensor.randn(304,320,64,64, dtype=dtypes.float32))
      self.assertEqual(out.dtype, dtypes.bfloat16, "expected float32 to be downcast to bfloat16 by Conv2d")
      out = unet.input_blocks[1][1].transformer_blocks[0].norm1(Tensor.randn(320, dtype=dtypes.bfloat16))
      self.assertEqual(out.dtype, dtypes.float32, "expected bfloat16 to be upcast to float32 by LayerNorm")
      out = unet.input_blocks[5][0].in_layers[0](Tensor.randn(304, 640, dtype=dtypes.bfloat16))
      self.assertEqual(out.dtype, dtypes.float32, "expected bfloat16 to be upcast to float32 by GroupNorm")

  def test_train_model(self):
    self.helper_test_init("v2-mlperf-train")

  def test_eval_model(self):
    self.helper_test_init("v2-mlperf-eval")

if __name__=="__main__":
  unittest.main()