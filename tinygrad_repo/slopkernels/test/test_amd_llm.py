from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit, UOp, dtypes, function, nn
from tinygrad.helpers import DEV, ceildiv, unwrap
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import ExpertWeights, HCConfig, HyperConnection, Linear, TransformerConfig
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops, resolve

import slopkernels
from slopkernels.src.amd_gguf import (
  HALFWORD_QUANTS,
  IQ1_S,
  IQ3_S,
  IQ4_XS,
  MXFP4,
  NVFP4,
  Q4_K,
  Q6_K,
  Q8_0,
  QUANT_SIZES,
  SMALL_BLOCK_QUANTS,
)


def _slop_target(device:str|tuple[str, ...]|None) -> tuple[int, ...]|None:
  dev = device[0] if isinstance(device, tuple) else device
  if dev is None or dev.split(":")[0] != "AMD": return None
  with Context(ALLOW_DEVICE_USAGE=1):
    t = getattr(Device[dev], "target", None)
    return tuple(int(x) for x in t) if t is not None and isinstance(Device[dev].renderer, HIPRenderer) else None

def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool: return _slop_target(device) is not None
def _is_rdna4(dev:str) -> bool: return (_slop_target(dev) or (0,))[0] == 12
def _is_cdna(dev:str) -> bool: return (_slop_target(dev) or (0,))[0] == 9


def _split_offset(v:int|UOp) -> tuple[int|UOp, int]:
  if isinstance(v, int): return v, 0
  if v.op is Ops.ADD and len(v.src) == 2:
    if v.src[1].op is Ops.CONST: return v.src[0], int(v.src[1].arg)
    if v.src[0].op is Ops.CONST: return v.src[1], int(v.src[0].arg)
  return v, 0


def flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  T_real = q.shape[2]
  decode = resolve(T_real == 1, False)
  if (target := _slop_target(q.device)) is not None:
    q_in = (q if not isinstance(T_real, UOp) else q.pad_to((*q.shape[:2], q.max_shape[2], q.shape[3]))).half()
    B, H, M, D = (int(d) for d in q_in.shape)
    _, _, H_KV, N_phys, _ = (int(d) for d in assigned_kv.shape)
    v_base, v_off = _split_offset(valid_end)
    vkl_u = v_base if isinstance(v_base, UOp) else UOp.const(int(v_base))
    if decode and M == 1 and D in (64, 128, 256) and (grp := H // H_KV) in (1, 2, 4, 8) and N_phys % 64 == 0:
      ch = min(48, ceildiv(N_phys, 64))
      kf = unwrap(slopkernels.get("amd_flash_decode")).specialize(
        B=B, H_KV=H_KV, G=grp, d_blks=D // 64, n_blks=N_phys // 64, C=ch, valid_kv_offset=v_off)
      if kf.op is Ops.SINK and kf.arg is None:
        kf = kf.substitute({b: b.replace(src=b.src[:1] + UOp.device_range_src(q.device), arg=replace(b.arg, device=q.device))
                            for b in kf.toposort() if b.op is Ops.ALLOC and b.arg.device is None})
      out = UOp.alloc((B, H, 1, D), dtypes.float32, device=q.device)
      return Tensor(out.after(kf.call(out, q_in.uop, assigned_kv.uop, vkl_u)))
    if not decode and target[0] != 9 and D in (64, 128, 256, 320) and N_phys % 32 == 0 and M % 32 == 0:
      k_fa = unwrap(slopkernels.get(f"{'rdna4' if target[0] == 12 else 'rdna3'}_flash_attn")).specialize(
        B=B, H=H, H_KV=H_KV, m_blks=M // 32, d_blks=D // 64, n_blks=N_phys // 32, valid_kv_offset=v_off, q_start_from_end=1)
      ql_u = T_real if isinstance(T_real, UOp) else UOp.const(M)
      out = UOp.alloc((B, H, M, D), dtypes.float32, device=q.device)
      ret = Tensor(out.after(k_fa.call(out, q_in.uop, assigned_kv.uop, vkl_u, ql_u)))
      return ret if not isinstance(T_real, UOp) else ret[:, :, :T_real]
  k, v = (assigned_kv[i, :, :, :valid_end].float() for i in range(2))
  mask = None if decode else Tensor.full((T_real, valid_end), -math.inf, dtype=dtypes.float32, device=q.device).triu(valid_end - T_real + 1)
  return q.float().scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)


def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
  assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
  assert state.uop.contiguous_view_offset() is not None, "recurrent state must be contiguous"
  B, H, T, K, V = int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3]), int(v.shape[3])
  ak = K if alpha.ndim == 4 else 1
  target = _slop_target(q.device)
  kern = unwrap(slopkernels.get("amd_gated_delta")).specialize(
    batch=B, heads=H, tokens=T, v_tiles=V // 4, k_blks=K // 32, alpha_k=ak, arch=target[0] if target else 9)
  out = UOp.alloc((B, H, T, V), dtypes.float32, device=q.device)
  srcs = [s.uop if s.uop.op is Ops.AFTER else s.contiguous().uop for s in (q, k, v, beta)]
  srcs += [alpha.uop.reshape(B, H, T, ak) if alpha.uop.op is Ops.AFTER else alpha.contiguous().reshape(B, H, T, ak).uop]
  srcs += [state.uop if state.uop.op is Ops.AFTER else state.contiguous().uop]
  sp = start_pos.uop if start_pos is not None else UOp.const(1)
  return Tensor(out.after(kern.call(out, *srcs, sp)))

def reference_weight(packed:np.ndarray, ggml_type:int, *shape:int) -> np.ndarray:
  from gguf import GGMLQuantizationType, dequantize
  return dequantize(packed.ravel(), GGMLQuantizationType(ggml_type)).reshape(shape)


class TestQ8Quantize(unittest.TestCase):
  def _test_quant_linear(self, ggml_type, block_bytes, in_features=2048, out_features=64, token_counts=(1, 3, 32, 64, 128),
                         bias=False, custom=True, symbolic=False):
    if custom and not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, (out_features * in_features // 256, block_bytes), dtype=np.uint8)
    if ggml_type in (11, 14): packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
    elif ggml_type == 10:
      packed[:, 80:82] = np.array([0.001], dtype=np.float16).view(np.uint8)
      packed[:, 82:84] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    elif ggml_type == 39: packed.reshape(-1, 17)[:, 0] = rng.integers(122, 128, packed.size // 17)
    elif ggml_type == 40: packed.reshape(-1, 36)[:, :4] = rng.integers(0x18, 0x20, (packed.size // 36, 4))
    elif ggml_type in SMALL_BLOCK_QUANTS:
      packed.reshape(-1, SMALL_BLOCK_QUANTS[ggml_type][1] * 2)[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    else: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    if ggml_type in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features * in_features, ggml_type).reshape(out_features, in_features)
    weight = reference_weight(packed, ggml_type, out_features, in_features)
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
        if custom: reference_x = x.astype(np.float16).astype(np.float32)
        reference_w = weight.astype(np.float16).astype(np.float32) if wmma else weight
        inp = (Tensor(x).realize() if not symbolic else
               Tensor(np.pad(x, ((0, 32 - tokens), (0, 0)))).contiguous().realize()[:UOp.variable("wmma_tokens", 1, 32).bind(tokens)])
        actual = (run if tokens == 1 or symbolic else linear)(inp)[:tokens].numpy()
        self.assertEqual(linear.ggml_type, ggml_type if custom else None)
        np.testing.assert_allclose(actual, reference_x @ reference_w.T + bias_value, rtol=3e-3, atol=2e-2)
        if not symbolic and tokens == 3 and ggml_type not in (12, 13, 14, 23):
          sym = Tensor(np.pad(x, ((0, 1), (0, 0)))).contiguous()[:UOp.variable("tokens", 1, 4).bind(3)]
          np.testing.assert_allclose(linear(sym)[:3].numpy(), reference_x @ reference_w.T + bias_value, rtol=3e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, ggml_type if custom else None)
  def test_wmma_architecture_selection(self):
    if not amd_custom_kernels_supported(device:=Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    self.assertEqual(_is_rdna4(device), Device[device].target[0] == 12)

  def test_q6_linear_compiles_in_function(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, 210, dtype=np.uint8)
    packed[-2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (2, 0))).contiguous().realize()[2:]
    decoded = ggml_data_to_tensor(raw, 256, 14).reshape(1, 256)
    linear = Linear(256, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    self.assertTrue(np.isfinite(run(Tensor.randn(1, 256)).realize().item()))
    packed_w = p if (p := getattr(linear, "packed", None)) is not None else linear.weight
    self.assertEqual(packed_w.nbytes(), 210)
    self.assertEqual(packed_w.dtype, dtypes.uint8)

  def test_q4_k_linear(self): self._test_quant_linear(12, 144)
  def test_iq4_linear(self): self._test_quant_linear(23, 136)
  def test_q5_linear(self): self._test_quant_linear(13, 176)
  def test_q8_0_linear_small_blocks(self): self._test_quant_linear(8, 272, in_features=288, out_features=16, token_counts=(1, 16))
  def test_q6_linear_odd_blocks(self): self._test_quant_linear(14, 210, in_features=768, out_features=3, token_counts=(1, 3, 16))
  # Exercise the format-specific dot product with a ragged second split-K chunk (WMMA for every format is tested in test_wmma_dequantization_basis).
  def _test_split_k_and_wmma(self, typ, token_counts=(1,)):
    self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=1280, out_features=16, token_counts=token_counts)
  def test_q2_k_linear(self): self._test_split_k_and_wmma(10)
  def test_q3_k_linear(self): self._test_split_k_and_wmma(11)
  def test_iq2_xxs_linear(self): self._test_split_k_and_wmma(16)
  def test_iq2_xs_linear(self): self._test_split_k_and_wmma(17, (1, 3))
  def test_iq3_xxs_linear(self): self._test_split_k_and_wmma(18)
  def test_iq1_s_linear(self): self._test_split_k_and_wmma(19)
  def test_iq4_nl_linear(self): self._test_split_k_and_wmma(20)
  def test_iq3_s_linear(self): self._test_split_k_and_wmma(21)
  def test_iq2_s_linear(self): self._test_split_k_and_wmma(22)
  def test_mxfp4_linear(self): self._test_quant_linear(39, 136, in_features=1056, token_counts=(1, 3))
  def test_mxfp4_linear_wmma(self): self._test_quant_linear(39, 136, in_features=288, token_counts=(16,))
  def test_nvfp4_linear(self): self._test_quant_linear(40, 144, in_features=320, out_features=16, token_counts=(1,))
  def test_nvfp4_linear_gemm(self): self._test_quant_linear(40, 144, in_features=320, out_features=16, token_counts=(16,))

  def test_wmma_linear_symbolic(self):
    def _no_gemv(name:str):
      if "_gemv_" in name: raise AssertionError(f"expected WMMA, got {name}")
      return slopkernels.get(name)
    with patch("tinygrad.llm.model._load_kernel", side_effect=_no_gemv):
      # Symbolic padding is shared; cover the Q4/Q5, generic/grid, and IQ4 LUT WMMA kernel families.
      for typ in (Q4_K, IQ3_S, IQ4_XS):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=16, token_counts=(17,), bias=True, symbolic=True)

  def test_mixed_quant_linear_wmma_tiles(self):
    def _no_gemv(name:str):
      if "_gemv_" in name: raise AssertionError(f"expected WMMA, got {name}")
      return slopkernels.get(name)
    with patch("tinygrad.llm.model._load_kernel", side_effect=_no_gemv):
      # Token tiling is independent of the format. Q6 also covers odd super-block counts; the basis test covers every format.
      for typ, tokens, in_features in ((Q8_0, 64, 256), (Q6_K, 32, 768), (IQ1_S, 16, 256)):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=in_features, out_features=64, token_counts=(tokens,), bias=True)

  def test_wmma_dequantization_basis(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(1907)
    scales = np.array([0, 2**-24, -2**-24, 2**-14, -0.00035, .001, .0037, -.125, 1, -4, 8], np.float16)
    identity = Tensor(np.eye(256, dtype=np.float16)).realize()
    for typ, size in QUANT_SIZES.items():
      if typ == 40: continue
      with self.subTest(ggml_type=typ):
        packed = rng.integers(0, 256, (16, size), dtype=np.uint8)
        blocks = packed.reshape(-1, SMALL_BLOCK_QUANTS[typ][1]*2) if typ in SMALL_BLOCK_QUANTS and typ != 39 else packed
        offset = size-2 if typ in (11, 14) else 80 if typ == 10 else 0
        if typ == 39: packed.reshape(-1, 17)[:, 0] = rng.integers(100, 141, 16*8)  # mxfp4 e8m0 scales: fp16 rounds e < 100 to 0, e > 140 overflows
        else: blocks[:, offset:offset+2] = np.resize(scales, len(blocks)).view(np.uint8).reshape(-1, 2)
        if typ in (10, 12, 13):
          offset = 82 if typ == 10 else 2
          blocks[:, offset:offset+2] = np.resize(scales[::-1], len(blocks)).view(np.uint8).reshape(-1, 2)
        expected = reference_weight(packed, typ, 16, 256).astype(np.float16).astype(np.float32).T
        # Halfword-only (mxfp4 byte-only) alignment exercises unaligned storage views.
        pad = 1 if typ == 39 else 2 if typ in HALFWORD_QUANTS else 4
        raw = Tensor(np.pad(packed.flatten(), (pad, 0))).realize()[pad:]
        linear = Linear(256, 16, bias=False)
        linear.weight = ggml_data_to_tensor(raw, 16*256, typ).reshape(16, 256)
        actual = linear(identity).numpy()
        self.assertEqual(linear.ggml_type, typ)
        # Bare RDNA3 WMMA has a small FP32 accumulation bias even for identity products.
        np.testing.assert_array_equal(actual.astype(np.float16), expected.astype(np.float16))
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2**-36)

  def test_quant_linear_unaligned_storage_fallback(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(73)
    x = rng.normal(size=(16, 256)).astype(np.float32)
    for typ, offset in ((12, 1), (12, 2), (12, 3), (21, 1), (21, 3)):
      with self.subTest(ggml_type=typ, offset=offset):
        packed = rng.integers(0, 256, (16, QUANT_SIZES[typ]), dtype=np.uint8)
        packed[:, :2] = np.array([.001], np.float16).view(np.uint8)
        if typ == 12: packed[:, 2:4] = np.array([.0002], np.float16).view(np.uint8)
        weight = reference_weight(packed, typ, 16, 256)
        raw = Tensor(np.pad(packed.flatten(), (offset, 0))).realize()[offset:]
        linear = Linear(256, 16, bias=False)
        linear.weight = ggml_data_to_tensor(raw, 16*256, typ).reshape(16, 256)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), x @ weight.T, rtol=2e-5, atol=1e-5)
        self.assertIsNone(linear.ggml_type)

  def test_quant_linear_decode_row_groups(self):
    # One, two, and four waves per workgroup; ragged split-K and multiple tokens must stay independent.
    for typ, outputs in ((12, 3), (13, 6), (14, 12)):
      with self.subTest(ggml_type=typ, out_features=outputs):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=1280, out_features=outputs, token_counts=(1, 3))

  def test_quant_linear_bias(self):
    # Bias is a shared epilogue after both decode and WMMA; per-format math doesn't need to be repeated here.
    self._test_quant_linear(Q4_K, QUANT_SIZES[Q4_K], in_features=256, out_features=16, token_counts=(1, 3, 16), bias=True)

  def test_quant_linear_partial_output_tile(self):
    # Cover a sub-tile output, a trailing tile, and IQ4's larger-output tile selection.
    for typ, size, outputs, tokens in ((12, 144, 48, 32), (13, 176, 48, 16), (23, 136, 4112, 32)):
      with self.subTest(ggml_type=typ, out_features=outputs):
        self._test_quant_linear(typ, size, in_features=256, out_features=outputs, token_counts=(tokens,))

  def test_quant_linear_preserves_rope_permutation(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    for typ, size in ((12, 144), (13, 176), (14, 210), (23, 136)):
      with self.subTest(ggml_type=typ):
        packed = rng.integers(0, 256, (16, size), dtype=np.uint8)
        packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if typ != 14: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if typ in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
        decoded = ggml_data_to_tensor(raw, 16*256, typ).reshape(16, 256).half()
        original = reference_weight(packed, typ, 16, 256).astype(np.float16)
        x = rng.normal(size=(3, 256)).astype(np.float16)
        for prefix in (None, 0, 4):
          with self.subTest(prefix=prefix):
            w = decoded.reshape(2, 8, 256)
            if prefix is None:
              weight = w.rearrange("n (h two) d -> n (two h) d", two=2)
            else:
              weight = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1)
            start = prefix or 0
            rows = np.arange(16).reshape(2, 8)
            order = np.concatenate((rows[:, :start], rows[:, start:].reshape(2, -1, 2).transpose(0, 2, 1).reshape(2, -1)), axis=1)
            linear = Linear(256, 16, bias=False)
            linear.weight = weight.reshape(16, 256)
            np.testing.assert_allclose(linear(Tensor(x)).numpy(), x.astype(np.float32) @ original[order.flatten()].astype(np.float32).T,
                                       rtol=3e-3, atol=2e-2)
            self.assertIsNone(linear.ggml_type)

  def test_quant_linear_rejects_unaligned_rows_and_integer_casts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for width in (128, 256):
      with self.subTest(width=width):
        packed = np.zeros((2*width//256, 136), dtype=np.uint8)
        packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        packed[:, 8:] = np.arange(128, dtype=np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).realize()[4:]
        weight = ggml_data_to_tensor(raw, 2*width, 23).reshape(2, width)
        if width == 256: weight = weight.int().float()
        expected = weight.numpy().sum(-1)[None]
        linear = Linear(width, 2, bias=False)
        linear.weight = weight
        np.testing.assert_allclose(linear(Tensor.ones(1, width)).numpy(), expected, rtol=1e-3, atol=1e-3)
        self.assertIsNone(linear.ggml_type)

  def test_dense_gemv_preserves_integer_casts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    linear = Linear(128, 1)
    linear.weight = Tensor.full((1, 128), 0.75).contiguous().realize().int().float()
    linear.bias = Tensor.full((1,), 0.75).contiguous().realize().int().float()
    np.testing.assert_array_equal(linear(Tensor.ones(1, 128)).numpy(), 0)

  def test_dense_gemv_float32_range(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    linear = Linear(128, 1, bias=False)
    linear.weight = Tensor.full((1, 128), 1/128, dtype=dtypes.float32).realize()
    np.testing.assert_array_equal(linear(Tensor.full((1, 128), 65536, dtype=dtypes.float32)).numpy(), 65536)

  def test_grouped_quant_linear(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(6)
    G, OUT, IN = 2, 16, 512
    packed = rng.integers(0, 256, (G*OUT*IN//32, 34), dtype=np.uint8)
    packed[:, :2] = np.array([0.001], np.float16).view(np.uint8)
    decoded = ggml_data_to_tensor(Tensor(packed.flatten()).contiguous().realize(), G*OUT*IN, 8).reshape(G*OUT, IN)
    w, x = reference_weight(packed, 8, G, OUT, IN), rng.standard_normal((1, 2, G, IN)).astype(np.float32)
    linear = Linear(IN, G*OUT, bias=False, groups=G)
    linear.weight = decoded
    got = linear(Tensor(x)).numpy()
    self.assertEqual(linear.ggml_type, 8)
    ref = np.einsum("btgi,goi->btgo", x, w)
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2*np.abs(ref).max())

  def test_gated_delta_state_and_precision(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for case in ("view", "reset", "half"):
      with self.subTest(case=case):
        q = Tensor.full((1, 1, 1, 32), 256 if case == "half" else 1, dtype=dtypes.half if case == "half" else dtypes.float32)
        state = Tensor.full((1, 1, 32, 4), int(case == "reset"), dtype=dtypes.float32).contiguous().realize().transpose(-1, -2)
        if case != "view": state = state.contiguous().realize()
        start = Tensor(UOp.variable("start_pos", 0, 10).bind(0)) if case == "reset" else None
        beta = Tensor.full((1, 1, 1), 1/2097152 if case == "half" else 1, dtype=dtypes.float32)
        if case != "reset":
          message = "recurrent state must be contiguous" if case == "view" else "recurrent Q/K must be float32"
          with self.assertRaisesRegex(AssertionError, message):
            gated_delta_prefill(q, q, Tensor.ones(1, 1, 1, 4), beta, Tensor.ones(1, 1, 1), state, start)
          continue
        out = gated_delta_prefill(q, q, Tensor.ones(1, 1, 1, 4), beta, Tensor.ones(1, 1, 1), state, start)
        np.testing.assert_array_equal(out.numpy(), 32)
        np.testing.assert_array_equal(state.numpy(), 1)

  def _test_moe_decode(self, typ:int, IN:int=256):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(5)
    E, T, K, OUT, size = 4, 1, 2, 16, QUANT_SIZES[typ]
    blocks = rng.integers(0, 256, (E*OUT*IN//256, size), dtype=np.uint8)
    if typ == MXFP4: blocks.reshape(-1, 17)[:, 0] = rng.integers(122, 128, blocks.size//17)  # e8m0 scales
    elif typ == NVFP4: blocks.reshape(-1, 36)[:, :4] = rng.integers(0x18, 0x20, (blocks.size//36, 4))
    else: blocks[:, :4] = np.array([0.001, 0.0001], np.float16).view(np.uint8)
    decoded = ggml_data_to_tensor(Tensor(blocks.flatten()).contiguous().realize(), E*OUT*IN, typ).reshape(E, OUT, IN)
    w = reference_weight(blocks, typ, E, OUT, IN)
    act = rng.standard_normal((1, T, K, IN)).astype(np.float32)
    sel = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])[None].astype(np.int32)
    ew = ExpertWeights(E, IN, OUT)
    ew.weight = decoded.half()
    ew.set_quantized()
    self.assertEqual(ew.ggml_type, typ)
    got = ew(Tensor(sel), Tensor(act)).numpy()[0]
    for t in range(T):
      for j in range(K):
        ref = w[sel[0, t, j]] @ act[0, t, j]
        np.testing.assert_allclose(got[t, j], ref, rtol=2e-2, atol=2e-2*np.abs(ref).max())

  def test_moe_decode(self): self._test_moe_decode(Q4_K)
  def test_moe_decode_iq1_s(self): self._test_moe_decode(IQ1_S)
  def test_moe_decode_iq3_s(self): self._test_moe_decode(IQ3_S)
  def test_moe_decode_mxfp4(self): self._test_moe_decode(MXFP4)
  def test_moe_decode_nvfp4(self): self._test_moe_decode(NVFP4, IN=320)

  def test_gated_delta_random_scan(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(42)
    q, k = (rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32) for _ in range(2))
    v = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    beta = rng.uniform(0, 1, (1, 2, 4)).astype(np.float32)
    beta[:, :, -1] = 0  # padded step must leave the recurrent state unchanged
    initial = rng.normal(0, 0.1, (1, 2, 8, 32)).astype(np.float32)
    for per_channel in (False, True):
      alpha = rng.uniform(0.8, 1, (1, 2, 4, 32) if per_channel else (1, 2, 4)).astype(np.float32)
      alpha[:, :, -1] = 1
      for start in (0, 3):
        with self.subTest(per_channel=per_channel, start=start):
          expected_state = np.zeros_like(initial) if start == 0 else initial.copy()
          outputs = []
          for t in range(4):
            decay = alpha[:, :, t, None, :] if per_channel else alpha[:, :, t, None, None]
            expected_state *= decay
            delta = (v[:, :, t] - (expected_state*k[:, :, t, None]).sum(-1)) * beta[:, :, t, None]
            expected_state += delta[..., None]*k[:, :, t, None]
            outputs.append((expected_state*q[:, :, t, None]).sum(-1))
          state = Tensor(initial).contiguous().realize()
          out = gated_delta_prefill(*map(Tensor, (q, k, v, beta, alpha)), state, Tensor(UOp.variable("start", 0, 3).bind(start)))
          np.testing.assert_allclose(out.numpy(), np.stack(outputs, axis=2), rtol=2e-4, atol=2e-6)
          np.testing.assert_allclose(state.numpy(), expected_state, rtol=2e-4, atol=2e-6)

  def test_hc_coeffs(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    config = TransformerConfig(num_blocks=1, dim=32, hidden_dim=32, n_heads=1, n_kv_heads=1, norm_eps=1e-6, vocab_size=32, head_dim=32,
                               rope_theta=1e4, rope_dim=32, v_head_dim=32, max_context=4, hc=HCConfig(count=4, sinkhorn_iters=3, eps=1e-6))
    hc = HyperConnection(config)
    rng = np.random.default_rng(0)
    hc.fn.weight, hc.base["weight"] = Tensor(rng.normal(0, 0.1, (24, 128)).astype(np.float16)), Tensor(rng.normal(size=24).astype(np.float16))
    hc.scale["weight"], xs = Tensor(rng.uniform(0.5, 1.5, 3).astype(np.float16)), Tensor(rng.normal(size=(1, 3, 4, 32)).astype(np.float32))
    self.assertIsNotNone(slopkernels.get("amd_hc_coeffs"))
    got = hc(xs, lambda h: (h * 0.5, None))[0].numpy()
    hc.fn.weight, hc.base["weight"], hc.scale["weight"] = (t.to("CPU") for t in (hc.fn.weight, hc.base["weight"], hc.scale["weight"]))
    np.testing.assert_allclose(got, hc(xs.to("CPU"), lambda h: (h * 0.5, None))[0].numpy(), rtol=1e-4, atol=1e-5)

  def test_dense_gemv_bias(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    w, bias = rng.normal(size=(32, 128)).astype(np.float16), rng.normal(size=32).astype(np.float16)
    linear = Linear(128, 32)
    linear.weight, linear.bias = Tensor(w), Tensor(bias)
    for tokens in (1, 3):
      with self.subTest(tokens=tokens):
        x = rng.normal(size=(tokens, 128)).astype(np.float32)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), x @ w.astype(np.float32).T + bias, rtol=2e-3, atol=2e-3)

  def test_q6_linear_multiple_tokens(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, blocks = 2048, 16*2048//256
    packed = rng.integers(0, 256, blocks*210, dtype=np.uint8)
    for i in range(blocks): packed[i*210+208:i*210+210] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16*in_features, 14).reshape(16, in_features)
    weight = reference_weight(packed, Q6_K, 16, in_features)
    linear = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    ref = x.astype(np.float16).astype(np.float32) @ weight.T
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), ref, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 14)

    # symbolic token counts take the padded kernel path and give the same results
    generic = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(generic, {"weight":decoded}, verbose=False, realize=False)
    sym = Tensor(np.concatenate([x, np.zeros((1, in_features), np.float32)])).contiguous().realize()[:UOp.variable("tokens", 1, 4).bind(3)]
    np.testing.assert_allclose(generic(sym)[:3].numpy(), ref, rtol=2e-3, atol=2e-2)
    self.assertTrue(generic.use_custom_quant)
    self.assertEqual(generic.ggml_type, 14)

  def test_attention_fallback_shapes(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for tokens, capacity, dim in ((1, 65, 64), (32, 64, 32), (32, 64, 384), (32, 64, 512)):
      with self.subTest(tokens=tokens, capacity=capacity, dim=dim):
        valid = 33
        cache = np.full((2, 1, 1, capacity, dim), np.nan, dtype=np.float16)
        cache[0, :, :, :valid] = 0
        cache[1, :, :, :valid] = np.arange(valid)[:, None]
        q = Tensor.zeros(1, 2, tokens, dim, dtype=dtypes.half)
        expected = np.broadcast_to(np.arange(valid-tokens, valid)[None, None, :, None]/2, q.shape)
        np.testing.assert_allclose(flash_attention(q, Tensor(cache), valid).numpy(), expected, rtol=1e-3, atol=1e-3)

  def test_attention_uses_physical_cache_length(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    q, k, v = Tensor.zeros(1, 2, 1, 32), Tensor.randn(1, 1, 1, 32), Tensor.randn(1, 1, 1, 32)
    cache = Tensor.empty(2, 1, 1, 256, 32, dtype=dtypes.half).contiguous()
    assigned = Tensor(cache.uop.after(cache[:, :, :, 0:1, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, 1).realize()
    np.testing.assert_allclose(out.numpy(), v.expand(1, 2, 1, 32).numpy(), rtol=2e-2, atol=2e-2)

  def test_flash_attention_decode_symbolic_gqa(self):
    with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected custom decode")):
      self._test_flash_decode(8, 2, 256, 128, 37, symbolic=True)

  def test_flash_attention_decode_gqa_tail(self): self._test_flash_decode(3, 1, 192, 64, 37)

  def test_flash_attention_decode_gqa_output_layout(self): self._test_flash_decode(4, 1, 128, 256, 3)
  def test_flash_attention_decode_large_gqa_group(self): self._test_flash_decode(8, 1, 256, 256, 73)

  def _test_flash_decode(self, heads, kv_heads, dim, n, valid, symbolic=False):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = rng.normal(size=(1, heads, 1, dim)).astype(np.float16)
    cache = rng.normal(size=(2, 1, kv_heads, n, dim)).astype(np.float16)
    k, v = (np.repeat(c[0, :, :valid].astype(np.float32), heads//kv_heads, axis=0) for c in cache)
    scores = q[0].astype(np.float32) @ k.transpose(0, 2, 1) / np.sqrt(dim)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (probs / probs.sum(-1, keepdims=True)) @ v
    cache_tensor = Tensor(cache).realize()
    if symbolic:
      start_pos = UOp.variable("start_pos", 0, n-1).bind(valid-1)
      valid = start_pos + 1
    np.testing.assert_allclose(flash_attention(Tensor(q), cache_tensor, valid).numpy(), expected[None], rtol=2e-3, atol=2e-3)

  def test_attention_symbolic_length_jit(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    cache = np.zeros((2, 1, 1, 128, 128), dtype=np.float16)
    cache[1] = np.arange(128, dtype=np.float16)[:, None]
    kv = Tensor(cache).realize()
    for tokens in (1, 32):
      q = Tensor(np.zeros((1, 2, tokens, 128), dtype=np.float16)).realize()
      run = TinyJit(lambda q, kv, end: flash_attention(q, kv, end).realize())
      for valid in (tokens, 33, 97, 128, 64, tokens):
        with self.subTest(tokens=tokens, valid=valid):
          end = UOp.variable("valid_length", tokens, 128).bind(valid)
          expected = np.broadcast_to(np.arange(valid-tokens, valid)[None, None, :, None]/2, q.shape)
          np.testing.assert_allclose(run(q, kv, end).numpy(), expected, rtol=1e-3, atol=1e-3)

  def test_prefill_attention_random(self):
    if not amd_custom_kernels_supported(d:=Tensor.empty(1).device) or _is_cdna(d): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(42)
    cases = ((2, 4, 2, 32, 64, 45, 64, False), (1, 3, 1, 64, 128, 83, 128, False),
             (1, 4, 2, 17, 128, 128, 128, True), (1, 2, 1, 3, 64, 33, 64, True),
             (1, 2, 1, 32, 320, 33, 64, False))
    for batch, heads, kv_heads, tokens, dim, valid, capacity, symbolic in cases:
      with self.subTest(batch=batch, tokens=tokens, dim=dim, symbolic=symbolic):
        q = rng.normal(size=(batch, heads, tokens, dim)).astype(np.float16)
        cache = rng.normal(size=(2, batch, kv_heads, capacity, dim)).astype(np.float16)
        cache[:, :, :, valid:] = np.nan
        k, v = (np.repeat(c[:, :, :valid].astype(np.float32), heads//kv_heads, axis=1) for c in cache)
        scores = q.astype(np.float32) @ k.swapaxes(-1, -2) / np.sqrt(dim)
        scores = np.where(np.arange(valid)[None, :] <= np.arange(valid-tokens, valid)[:, None], scores, -np.inf)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        expected = (probs / probs.sum(-1, keepdims=True)) @ v
        query, kv = Tensor(q), Tensor(cache).realize()
        end = UOp.variable("valid", 1, capacity).bind(valid)
        if symbolic:
          count = UOp.variable("query_tokens", 1, 32).bind(tokens)
          query = Tensor(np.pad(q, ((0, 0), (0, 0), (0, 32-tokens), (0, 0)))).contiguous()[:, :, :count]
        @function(allow_implicit=True)
        def run(q:Tensor, kv:Tensor, end:UOp): return flash_attention(q, kv, end)
        with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected WMMA attention")):
          actual = run(query, kv, end)[:, :, :tokens].numpy()
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=2e-3)

  def test_prefill_attention_nonfinite_cache_tail(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = Tensor.zeros(1, 2, 32, 128, dtype=dtypes.half)
    values = rng.normal(size=(33, 128)).astype(np.float16)
    expected = np.stack([values[:i+2].astype(np.float32).mean(0) for i in range(32)])[None, None].repeat(2, axis=1)
    for tail in (np.nan, np.inf, -np.inf):
      with self.subTest(tail=tail):
        cache = np.full((2, 1, 1, 64, 128), tail, dtype=np.float16)
        cache[0, :, :, :33] = 0
        cache[1, :, :, :33] = values
        valid = UOp.variable("valid_end", 32, 64).bind(33)
        cache_tensor = Tensor(cache).realize()
        assigned = Tensor(cache_tensor.uop.after(Tensor(valid).uop))
        out = flash_attention(q, assigned, valid)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_flash_attention_decode_beyond_256_chunks(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    n = 257 * 64
    q = Tensor.zeros(1, 1, 1, 32, dtype=dtypes.half).realize()
    k = Tensor.zeros(1, 1, n, 32, dtype=dtypes.half)
    v = Tensor.zeros(1, 1, n-64, 32, dtype=dtypes.half).cat(Tensor.ones(1, 1, 64, 32, dtype=dtypes.half), dim=2)
    cache = Tensor.stack(k, v).contiguous().realize()
    for valid, expected in ((1, 0), (n, 1/257)):
      with self.subTest(valid=valid):
        valid_kv_len = UOp.variable("valid_kv_len", 1, n).bind(valid)
        assigned = Tensor(cache.uop.after(Tensor(valid_kv_len).uop))
        np.testing.assert_allclose(flash_attention(q, assigned, valid_kv_len).numpy(), expected, rtol=2e-3, atol=2e-4)

  def test_flash_attention_decode_long_context_random(self):
    self._test_flash_decode(8, 2, 128, 257*64, 257*64-13)  # past 256 chunks, with a ragged tail

  def test_flash_attention_decode_chunk_round_accumulator_range(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    valid_kv_len, max_kv_len = 6749, 6784  # three chunk rounds, with a ragged tail
    q = Tensor.zeros(1, 8, 1, 32, dtype=dtypes.half).realize()
    cache = Tensor.stack(Tensor.zeros(1, 1, max_kv_len, 32, dtype=dtypes.half),
                         Tensor.full((1, 1, max_kv_len, 32), 5500, dtype=dtypes.half)).contiguous().realize()
    np.testing.assert_allclose(flash_attention(q, cache, valid_kv_len).numpy(), 5500, rtol=2e-3, atol=2e-3)

  def test_prefill_attention_unaligned_start(self):
    if not amd_custom_kernels_supported(d:=Tensor.empty(1).device): self.skipTest("RDNA3/4 or CDNA3/4 required")
    if _is_cdna(d) and DEV.interface.startswith("MOCK"): self.skipTest("too slow on the CDNA emulator")
    rng = np.random.default_rng(42)
    start_pos = 1718
    q = Tensor.zeros(1, 8, 32, 128)
    old_kv = rng.normal(size=(2, 1, 1, start_pos, 128)).astype(np.float32)
    new_kv = rng.normal(size=(2, 1, 1, 32, 128)).astype(np.float32)
    cache = Tensor.zeros(2, 1, 1, 2048, 128, dtype=dtypes.half).contiguous()
    Tensor.realize(cache[:, :, :, :start_pos].assign(Tensor(old_kv).cast(dtypes.half)))
    sp = UOp.variable("start_pos", 0, 2047).bind(start_pos)
    assigned = Tensor(cache.uop.after(cache[:, :, :, sp:sp+32, :].uop.store(Tensor(new_kv).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, sp+32).realize()
    values = np.concatenate([old_kv[1, 0, 0], new_kv[1, 0, 0]]).astype(np.float16).astype(np.float32)
    expected = np.stack([values[:start_pos+i+1].mean(0) for i in range(32)])[None, None].repeat(8, axis=1)
    np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

if __name__ == "__main__": unittest.main()
