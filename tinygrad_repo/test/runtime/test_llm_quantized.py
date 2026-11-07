import gc, itertools, unittest, weakref
import numpy as np
from tinygrad import Tensor, UOp, dtypes, function, Device, TinyJit
from tinygrad.helpers import Context
from tinygrad.llm.kernels.amd import (Linear, fp8_matmul, amd_custom_kernels_supported, QUANT_SIZES, HALFWORD_QUANTS, SMALL_BLOCK_QUANTS,
                                      iq4_half_lut, _iq_grid)
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import ExpertWeights
from tinygrad.llm.hf import e4m3fn_to_fnuz
from test.helpers import not_support_multi_device

class QuantLinearMixin:
  def _test_quant_linear(self, ggml_type, block_bytes, in_features=2048, out_features=64, token_counts=(1, 3, 32, 64, 128),
                         bias=False, custom=True, symbolic=False):
    if custom and not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, (out_features*in_features//256, block_bytes), dtype=np.uint8)
    if ggml_type in (11, 14): packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
    elif ggml_type == 10:
      packed[:, 80:82] = np.array([0.001], dtype=np.float16).view(np.uint8)
      packed[:, 82:84] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    elif ggml_type in SMALL_BLOCK_QUANTS:
      packed.reshape(-1, SMALL_BLOCK_QUANTS[ggml_type]*2)[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    else: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    if ggml_type in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features*in_features, ggml_type).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    linear.weight = decoded
    bias_value = rng.normal(size=out_features).astype(np.float32) if bias else 0
    if bias: linear.bias = Tensor(bias_value)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    for tokens in token_counts:
      with self.subTest(tokens=tokens):
        x = rng.normal(size=(tokens, in_features)).astype(np.float32 if tokens == 3 else np.float16)
        reference_x = x.astype(np.float32)
        wmma = custom and (32 if symbolic else tokens) % 16 == 0 and out_features % 16 == 0
        if wmma: reference_x = x.astype(np.float16).astype(np.float32)
        if custom and not wmma:
          grouped = reference_x.reshape(tokens, -1, 32)
          scale = np.maximum(np.abs(grouped).max(-1, keepdims=True) / 127, 1e-8)
          reference_x = (np.clip(np.rint(grouped/scale), -127, 127)*scale).reshape(tokens, in_features)
        reference_w = weight.astype(np.float16).astype(np.float32) if wmma else weight
        inp = Tensor(x) if not symbolic else Tensor(np.pad(x, ((0, 32-tokens), (0, 0)))).contiguous()[:
          UOp.variable("wmma_tokens", 1, 32).bind(tokens)]
        actual = (run if tokens == 1 or symbolic else linear)(inp)[:tokens].numpy()
        self.assertEqual(linear.ggml_type, ggml_type if custom else None)
        np.testing.assert_allclose(actual, reference_x @ reference_w.T + bias_value, rtol=3e-3, atol=2e-2)
        if not symbolic and tokens == 3 and ggml_type not in (12, 13, 14, 23):
          sym = Tensor(np.pad(x, ((0, 1), (0, 0)))).contiguous()[:UOp.variable("tokens", 1, 4).bind(3)]
          np.testing.assert_allclose(linear(sym)[:3].numpy(), reference_x @ reference_w.T + bias_value, rtol=3e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, ggml_type if custom else None)

class TestQ8Quantize(QuantLinearMixin, unittest.TestCase):
  def test_quant_tables_not_retained(self):
    # one _iq_grid table and the iq4 lut cover both table creation paths
    for typ in (18, 23):
      table = (iq4_half_lut(Device.DEFAULT) if typ == 23 else _iq_grid(Device.DEFAULT, typ)).realize()
      ref = weakref.ref(table)
      del table
      gc.collect()
      self.assertIsNone(ref())

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_quant_tables_multidevice(self):
    devices = (Device.DEFAULT, f"{Device.DEFAULT}:1")
    for typ in (18, 23):
      with self.subTest(ggml_type=typ):
        single = iq4_half_lut(Device.DEFAULT) if typ == 23 else _iq_grid(Device.DEFAULT, typ)
        table = (iq4_half_lut(devices) if typ == 23 else _iq_grid(devices, typ)).realize()
        self.assertEqual(table.device, devices)
        for i in range(len(devices)):
          np.testing.assert_array_equal(Tensor(table.uop.mselect(i)).numpy(), single.numpy())

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_quant_linear_multidevice(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest("RDNA3 required")
    devices = (Device.DEFAULT, f"{Device.DEFAULT}:1")
    rng = np.random.default_rng(42)
    # Exercise both the grid-backed decode path and the IQ4 WMMA lookup table without a TP model.
    for typ, tokens in ((18, 1), (23, 16)):
      with self.subTest(ggml_type=typ):
        raw = Tensor(rng.integers(0, 0x3c, 16*QUANT_SIZES[typ]+4, dtype=np.uint8))
        x = Tensor(rng.normal(size=(tokens, 256)).astype(np.float32))
        def run(device):
          linear = Linear(256, 16, bias=False)
          linear.weight = ggml_data_to_tensor(raw.to(device).realize()[4:], 16*256, typ).reshape(16, 256)
          out = linear(x.to(device)).realize()
          self.assertEqual(linear.ggml_type, typ)
          return out
        single, parallel = run(Device.DEFAULT).numpy(), run(devices)
        for i in range(len(devices)):
          np.testing.assert_allclose(Tensor(parallel.uop.mselect(i)).numpy(), single, atol=1e-5, rtol=1e-5)

  def test_quant_weights_share_storage(self):
    for (ggml_type, type_size), offset in itertools.product(QUANT_SIZES.items(), (0, 4)):
      with self.subTest(ggml_type=ggml_type, offset=offset):
        packed = np.arange(type_size + offset, dtype=np.uint8)
        raw = Tensor(packed).realize()[offset:]
        if raw.uop.contiguous_view() is None: self.skipTest("requires buffer views")
        decoded = ggml_data_to_tensor(raw, 256, ggml_type).reshape(1, 256)
        linear = Linear(256, 1, bias=False)
        linear.set_quantized(decoded)
        linear.weight.realize()
        self.assertEqual(linear.ggml_type, ggml_type)
        self.assertEqual(linear.weight.dtype, dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32)
        self.assertEqual(linear.weight.nbytes(), type_size)
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), packed[offset:])
        raw.assign(raw.full_like(1)).realize()
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), np.ones(type_size, dtype=np.uint8))

  def test_small_block_quant_weights(self):
    for ggml_type, halfwords in SMALL_BLOCK_QUANTS.items():
      with self.subTest(ggml_type=ggml_type):
        packed = np.arange(25*halfwords*2, dtype=np.uint8)
        raw = Tensor(packed).realize()[0:]
        if raw.uop.contiguous_view() is None: self.skipTest("requires buffer views")
        linear = Linear(800, 1, bias=False)
        linear.set_quantized(ggml_data_to_tensor(raw, 800, ggml_type).reshape(1, 800))
        self.assertEqual(linear.ggml_type, ggml_type)
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), packed)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "slow on WEBGPU")
  def test_quant_linear_fallback(self):
    if amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("run with DISABLE_AMD_KERNELS=1")
    # per-type dequant math on the generic path is covered by test_gguf, spot check a representative set here
    for typ in (12, 14, 17, 23):
      with self.subTest(ggml_type=typ):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=16, token_counts=(1, 3), bias=True, custom=False)

def fp8_decode(bits, fnuz=False):
  exponent, mantissa = ((bits >> 3) & 15).astype(np.int32), (bits & 7).astype(np.float32)
  value = np.ldexp(np.where(exponent == 0, mantissa, 8 + mantissa), np.maximum(exponent, 1) - (11 if fnuz else 10))
  return np.where(bits & 128, -value, value)

def fp8_reference(x, bits, scale, block_size, fnuz=False):
  bn, bk = block_size
  if fnuz: bits, scale = np.where(bits == 128, 0, bits), scale * 2
  w = fp8_decode(bits, fnuz)
  k, n = w.shape[-1], w.shape[-2]
  x = np.pad(x, [(0, 0)] * (x.ndim-1) + [(0, (-k) % bk)]).reshape(*x.shape[:-1], -1, bk)
  limit = 224.0 if fnuz else 448.0
  xs = np.maximum(np.abs(x).max(-1, keepdims=True), 1e-10) / limit
  q = np.clip(x / xs, -limit, limit)
  step = np.exp2(np.maximum(np.frexp(q)[1] - 4, -10 if fnuz else -9)).astype(np.float32)
  x = (np.rint(q / step) * step * xs).reshape(*x.shape[:-2], -1)[..., :k]
  w = w * scale.repeat(bn, -2).repeat(bk, -1)[..., :n, :k]
  return x @ w.swapaxes(-1, -2)

class TestBlockFP8(unittest.TestCase):
  def setUp(self):
    supported = Device[Device.DEFAULT].renderer.supported_dtypes()
    self.formats = [fnuz for fnuz, dt in ((False, dtypes.fp8e4m3), (True, dtypes.fp8e4m3fnuz)) if dt in supported]

  def test_fnuz_conversion(self):
    bits = np.array([i for i in range(256) if i not in (127, 255)], np.uint8).reshape(2, 127)
    scale = np.array([[0.125], [0.5]], np.float32)
    w, s = e4m3fn_to_fnuz(Tensor(bits).bitcast(dtypes.fp8e4m3), Tensor(scale))
    converted = w.bitcast(dtypes.uint8).numpy()
    np.testing.assert_array_equal(converted, np.where(bits == 128, 0, bits))
    expected = fp8_decode(bits) * scale
    np.testing.assert_array_equal(fp8_decode(converted, fnuz=True) * s.numpy(), expected)
    if True in self.formats: np.testing.assert_array_equal((w.float() * s).numpy(), expected)

  def test_dense(self):
    if not self.formats: self.skipTest('requires native FP8')
    rng = np.random.default_rng(1)
    for fnuz, tc, (n, k, tokens) in itertools.product(self.formats, (0, 1), ((132, 260, 3), (256, 256, 16), (16, 256, 1))):
      with Context(TC=tc), self.subTest(fnuz=fnuz, tc=tc, n=n, k=k):
        x = rng.normal(size=(2, tokens, k)).astype(np.float32)
        x[0, 0] = 0
        bits = rng.integers(0, 73, size=(n, k), dtype=np.uint8)  # finite magnitudes from 0 to 4
        bits |= rng.integers(0, 2, size=bits.shape, dtype=np.uint8) << 7
        scales = rng.uniform(.01, .5, ((n+127)//128, (k+127)//128)).astype(np.float32)
        w, s = Tensor(bits).bitcast(dtypes.fp8e4m3), Tensor(scales)
        if fnuz: w, s = e4m3fn_to_fnuz(w, s)
        layer = Linear(k, n, bias=False)
        layer.weight, layer.weight_scale_inv = w, s
        expected = fp8_reference(x, bits, scales, (128, 128), fnuz)
        np.testing.assert_allclose(layer(Tensor(x)).numpy(), expected, atol=2e-5, rtol=2e-5)

  def test_fp32_fallback(self):
    for tc, fnuz in itertools.product((0, 1), (False, True)):
      with Context(TC=tc), self.subTest(tc=tc, fnuz=fnuz):
        x = Tensor([[[1.125, 0., 0., 0.]], [[-1.125, 0., 0., 0.]], [[0., 0., 0., 0.]]])
        w = Tensor([[57, 0, 0, 0], [60, 0, 0, 0]], dtype=dtypes.uint8).bitcast(dtypes.fp8e4m3)
        scale = Tensor.ones(1, 1)
        if fnuz: w, scale = e4m3fn_to_fnuz(w, scale)
        np.testing.assert_allclose(fp8_matmul(x, w, scale, (2, 4)).numpy(), [[[1.265625, 1.6875]], [[-1.265625, -1.6875]], [[0., 0.]]], rtol=1e-6)

  def _routed_experts(self, fnuz):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3, 1, 24)).astype(np.float32)
    bits = rng.integers(0, 73, size=(4, 16, 24), dtype=np.uint8)  # finite magnitudes from 0 to 4
    bits |= rng.integers(0, 2, size=bits.shape, dtype=np.uint8) << 7
    scales = rng.uniform(.01, .5, (4, 4, 3)).astype(np.float32)
    selected = np.array([[[0, 2], [2, 0], [3, 1]], [[1, 1], [0, 3], [2, 1]]], np.int32)
    layer = ExpertWeights(4, 24, 16)
    w, scale = Tensor(bits).bitcast(dtypes.fp8e4m3), Tensor(scales)
    if fnuz: w, scale = e4m3fn_to_fnuz(w, scale)
    layer.weight, layer.weight_scale_inv, layer.weight_block_size = w, scale, (4, 8)
    return layer, x, bits, scales, selected

  def test_routed_experts(self):
    if not self.formats: self.skipTest('requires native FP8')
    for fnuz in self.formats:
      with self.subTest(fnuz=fnuz):
        layer, x, bits, scales, selected = self._routed_experts(fnuz)
        expected = fp8_reference(x[..., None, :], bits[selected], scales[selected], (4, 8), fnuz).squeeze(-2)
        np.testing.assert_allclose(layer(Tensor(selected), Tensor(x)).numpy(), expected, atol=2e-6, rtol=2e-5)

  def test_routed_experts_jit(self):
    if not self.formats: self.skipTest('requires native FP8')
    for fnuz in self.formats:
      with self.subTest(fnuz=fnuz):
        layer, x, bits, scales, selected = self._routed_experts(fnuz)
        run = TinyJit(lambda sel, h: layer(sel, h).realize())
        for ids in (selected, selected, selected[..., ::-1].copy()):
          got = run(Tensor(ids).realize(), Tensor(x).realize()).numpy()
          expected = fp8_reference(x[..., None, :], bits[ids], scales[ids], (4, 8), fnuz).squeeze(-2)
          np.testing.assert_allclose(got, expected, atol=2e-6, rtol=2e-5)

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_tensor_parallel(self):
    if not self.formats: self.skipTest('requires native FP8')
    rng, devices = np.random.default_rng(3), (Device.DEFAULT, f"{Device.DEFAULT}:1")
    x = rng.normal(size=(1, 3, 256)).astype(np.float32)
    bits = rng.integers(0, 73, size=(256, 256), dtype=np.uint8)  # finite magnitudes from 0 to 4
    bits |= rng.integers(0, 2, size=bits.shape, dtype=np.uint8) << 7
    scales = rng.uniform(.01, .5, (2, 2)).astype(np.float32)
    for fnuz, axis in itertools.product(self.formats, (0, 1)):
      with self.subTest(fnuz=fnuz, axis=axis):
        w, scale = Tensor(bits).bitcast(dtypes.fp8e4m3), Tensor(scales)
        if fnuz: w, scale = e4m3fn_to_fnuz(w, scale)
        expected = fp8_reference(x, bits, scales, (128, 128), fnuz)
        w = w.bitcast(dtypes.uint8).shard(devices, axis).bitcast(w.dtype)
        got = fp8_matmul(Tensor(x).shard(devices, 2 if axis == 1 else None), w, scale.shard(devices, axis), (128, 128))
        np.testing.assert_allclose(got.to(Device.DEFAULT).numpy(), expected, atol=1e-5, rtol=2e-5)

if __name__ == "__main__": unittest.main()
