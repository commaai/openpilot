import itertools, pathlib, tempfile, unittest
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType as Q, GGML_QUANT_SIZES
from tinygrad import Device, Tensor, nn
from tinygrad.helpers import DEV
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from tinygrad.llm.gguf import gguf_load, gguf_parse, gguf_shard, ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported
from test.helpers import not_support_multi_device

DEVICES = (Device.DEFAULT, f"{Device.DEFAULT}:1")

def write_gguf(writer:GGUFWriter):
  writer.write_header_to_file()
  writer.write_kv_data_to_file()
  writer.write_tensors_to_file()
  writer.close()

def random_data(shape:tuple[int, ...], typ:Q, seed=0) -> np.ndarray:
  # the data of a tensor as a GGUF stores it, every byte < 0x3c: any fp16 scale read from a block is finite
  rng = np.random.default_rng(seed)
  if typ == Q.F32: return rng.normal(size=shape).astype(np.float32)
  block, size = GGML_QUANT_SIZES[typ]
  return rng.integers(0, 0x3c, (*shape[:-1], shape[-1]//block*size), dtype=np.uint8)

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestGGUFShard(unittest.TestCase):
  def test_shards(self):
    # name: (shape, type, sharded axis)
    tensors = {'rows': ((8, 1024), Q.Q4_K, 0), 'cols': ((8, 1024), Q.Q6_K, 1), 'experts_rows': ((4, 8, 512), Q.Q4_K, 1),
               'experts_cols': ((4, 8, 512), Q.Q8_0, 2), 'vector': ((8,), Q.F32, 0), 'copied': ((8, 512), Q.Q4_K, None)}
    with tempfile.TemporaryDirectory() as folder:
      writer = GGUFWriter(path:=pathlib.Path(folder)/'model.gguf', 'test')
      for name, (shape, typ, _) in tensors.items(): writer.add_tensor(name, random_data(shape, typ), raw_dtype=typ)
      write_gguf(writer)
      full = gguf_load(path)[1]
      out = gguf_shard(gguf_parse(path)[1], DEVICES, {name: axis for name, (_, _, axis) in tensors.items() if axis is not None})
      for name, (_, _, axis) in tensors.items():
        with self.subTest(name=name):
          self.assertEqual((out[name].device, out[name].uop.axis), (DEVICES, axis))
          np.testing.assert_array_equal(out[name].to(Device.DEFAULT).numpy(), full[name].numpy())

  def test_split_quantization_block(self):
    # two devices would each get one and a half Q4_K blocks of every row
    data = Tensor(random_data((8, 768), Q.Q4_K).reshape(-1), device='CPU')
    with self.assertRaises(ValueError): gguf_shard({'w': (data, (8, 768), Q.Q4_K)}, DEVICES, {'w': 1})

  def test_quantized_shards(self):
    # the shards are still recognized as packed quantized weights, and keep the direction of their split
    if Tensor.empty(1, device=DEVICES[0]).uop.contiguous_view() is None: self.skipTest("requires buffer views")
    for typ, axis in itertools.product((Q.Q4_K, Q.Q6_K, Q.IQ4_XS), (0, 1, None)):
      with self.subTest(typ=typ.name, axis=axis):
        data = Tensor(random_data((8, 1024), typ).reshape(-1), device='CPU')
        layer = Linear(1024, 8, bias=False)
        layer.set_quantized(gguf_shard({'w': (data, (8, 1024), typ)}, DEVICES, {} if axis is None else {'w': axis})['w'].half())
        self.assertEqual((layer.ggml_type, layer.shard_axis), (typ, axis))

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestTensorParallel(unittest.TestCase):
  def test_moe(self):
    with tempfile.TemporaryDirectory() as folder:
      config = TransformerConfig(num_blocks=1, dim=32, hidden_dim=32, n_heads=4, n_kv_heads=4, norm_eps=1e-5,
        vocab_size=32, head_dim=8, v_head_dim=8, rope_theta=10000, rope_dim=8, max_context=16,
        num_experts=4, num_experts_per_tok=2)
      writer = GGUFWriter(path:=pathlib.Path(folder)/'model.gguf', 'deepseek2')
      for key,value in {'context_length':16, 'embedding_length':32, 'expert_feed_forward_length':32, 'block_count':1,
                        'attention.head_count':4, 'attention.head_count_kv':4, 'attention.key_length':8,
                        'rope.dimension_count':8, 'expert_count':4, 'expert_used_count':2}.items(): writer.add_uint32('deepseek2.'+key, value)
      writer.add_float32('deepseek2.rope.freq_base', 10000)
      writer.add_float32('deepseek2.attention.layer_norm_rms_epsilon', 1e-5)
      writer.add_array('tokenizer.ggml.tokens', [str(i) for i in range(32)])
      rng = np.random.default_rng(42)
      for name,weight in nn.state.get_state_dict(Transformer(config)).items():
        value = rng.normal(1, .1, weight.shape) if name.endswith('norm.weight') else rng.normal(0, .1, weight.shape)
        writer.add_tensor(name, value.astype(np.float32))
      write_gguf(writer)
      single, parallel = Transformer.from_gguf(path, 16)[0], Transformer.from_gguf(path, 16, shard=2)[0]
      block = parallel.blk[0]
      self.assertEqual((block.ffn_gate_exps.weight.uop.axis, block.ffn_up_exps.weight.uop.axis, block.ffn_down_exps.weight.uop.axis), (1, 1, 2))
      x = Tensor(rng.normal(size=(1, 4, config.dim)).astype(np.float32)).realize()
      ref = single.blk[0]._feed_forward(x).numpy()
      out = parallel.blk[0]._feed_forward(x.shard(DEVICES)).to(Device.DEFAULT).numpy()
      np.testing.assert_allclose(out, ref, atol=2e-3, rtol=2e-3)

  def test_quantized_linear(self):
    # the AMD kernels on the packed shards: decode (1 token) and WMMA (16 tokens), output and input features split
    if not amd_custom_kernels_supported(DEVICES[0]): self.skipTest("needs the AMD custom kernels")
    for typ, axis, tokens in itertools.product((Q.Q4_K, Q.Q6_K), (0, 1), (1, 16)):
      with self.subTest(typ=typ.name, axis=axis, tokens=tokens):
        data = random_data((64, 1024), typ).reshape(-1)
        layer = Linear(1024, 64, bias=False)
        layer.set_quantized(gguf_shard({'w': (Tensor(data, device='CPU'), (64, 1024), typ)}, DEVICES, {'w': axis})['w'].half())
        self.assertEqual(layer.ggml_type, typ)
        x = np.random.default_rng(1).normal(size=(1, tokens, 1024)).astype(np.float16)
        w = ggml_data_to_tensor(Tensor(data, device='CPU'), 64*1024, typ).reshape(64, 1024).float().numpy()
        ref = x.astype(np.float32) @ w.T
        got = layer(Tensor(x).shard(DEVICES, 2 if axis == 1 else None)).float().to(Device.DEFAULT).numpy()
        np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2*np.abs(ref).max())

  @unittest.skipIf(DEV.interface.startswith("MOCK"), "too heavy for mock GPUs")
  def test_model(self):
    rng = np.random.default_rng(42)
    config = TransformerConfig(num_blocks=2, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, norm_eps=1e-5, vocab_size=64, head_dim=64,
      v_head_dim=64, rope_theta=10000, rope_dim=16, max_context=64, qk_norm=64, attn_output_gate=True, ssm=SSMConfig(4, 32, 2, 4, 128),
      ssm_layers=(True, False))
    with tempfile.TemporaryDirectory() as folder:
      writer = GGUFWriter(path:=pathlib.Path(folder)/'model.gguf', 'qwen35')
      for key,value in {'context_length':64, 'embedding_length':256, 'feed_forward_length':512, 'block_count':2, 'full_attention_interval':2,
                        'ssm.conv_kernel':4, 'ssm.state_size':32, 'ssm.group_count':2, 'ssm.time_step_rank':4, 'ssm.inner_size':128,
                        'attention.head_count':4, 'attention.head_count_kv':2, 'attention.key_length':64, 'rope.dimension_count':16}.items():
        writer.add_uint32('qwen35.'+key, value)
      writer.add_float32('qwen35.rope.freq_base', 10000)
      writer.add_float32('qwen35.attention.layer_norm_rms_epsilon', 1e-5)
      writer.add_array('tokenizer.ggml.tokens', [str(i) for i in range(64)])
      for name,weight in nn.state.get_state_dict(Transformer(config)).items():
        value = -rng.random(weight.shape) if name.endswith('ssm_a') else rng.normal(1, .1, weight.shape) if 'norm' in name else \
                rng.normal(0, .5 if 'token_embd' in name else .05, weight.shape)
        writer.add_tensor(name.replace('ffn_norm', 'post_attention_norm'), value.astype(np.float32))
      write_gguf(writer)
      single, parallel = Transformer.from_gguf(path, 64)[0], Transformer.from_gguf(path, 64, shard=2)[0]
      prompt = [int(x) for x in rng.integers(0, 64, 40)]
      self.assertEqual(list(itertools.islice(parallel.generate(list(prompt)), 6)), list(itertools.islice(single.generate(list(prompt)), 6)))
      # the kv cache is sharded on its heads, the gated deltanet is copied to every device
      self.assertEqual(parallel.blk[1].cache_kv.uop.axis, 2)
      for t in (parallel.blk[0].attn_qkv.weight, parallel.blk[0].recurrent_state): self.assertEqual((t.device, t.uop.axis), (DEVICES, None))

if __name__ == '__main__': unittest.main()
