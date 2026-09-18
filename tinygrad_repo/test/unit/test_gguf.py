import os, struct, unittest, tempfile, pathlib, sys
from tinygrad import dtypes, Tensor, fetch, Device
from tinygrad.helpers import disable_gc
from tinygrad.llm.gguf import _ggml_iq_grid, ggml_data_to_tensor, gguf_load
from tinygrad.runtime.autogen import ggml_common as _ggml
import numpy as np
from gguf import GGUFReader, GGUFValueType, GGMLQuantizationType, GGML_QUANT_SIZES, dequantize, quantize
from gguf.quants import IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS

ggml_test_block_count = 4
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

class TestGGUFTables(unittest.TestCase):
  def test_iq2_xxs_grid_matches_gguf_py(self):
    IQ2_XXS.init_grid()
    grid = _ggml_iq_grid(Device.DEFAULT, _ggml.iq2xxs_grid, (256, 8)).numpy()
    np.testing.assert_equal(grid, IQ2_XXS.grid.reshape(256, 8))

  def test_iq2_xs_grid_matches_gguf_py(self):
    IQ2_XS.init_grid()
    grid = _ggml_iq_grid(Device.DEFAULT, _ggml.iq2xs_grid, (512, 8)).numpy()
    np.testing.assert_equal(grid, IQ2_XS.grid.reshape(512, 8))

  def test_iq2_s_grid_matches_gguf_py(self):
    IQ2_S.init_grid()
    grid = _ggml_iq_grid(Device.DEFAULT, _ggml.iq2s_grid, (1024, 8)).numpy()
    np.testing.assert_equal(grid, IQ2_S.grid.reshape(1024, 8))

  def test_iq1_s_grid_matches_gguf_py(self):
    IQ1_S.init_grid()
    grid = _ggml_iq_grid(Device.DEFAULT, _ggml.iq1s_grid, (2048, 8)).numpy()
    grid = np.where(grid > 127, grid - 256, grid)
    np.testing.assert_equal(grid, IQ1_S.grid.reshape(2048, 8))

  def test_iq3_xxs_grid_matches_gguf_py(self):
    IQ3_XXS.init_grid()
    grid = _ggml_iq_grid(Device.DEFAULT, _ggml.iq3xxs_grid, (256, 4)).numpy()
    np.testing.assert_equal(grid, IQ3_XXS.grid.reshape(256, 4))

  def test_iq3_s_grid_matches_gguf_py(self):
    IQ3_S.init_grid()
    grid = _ggml_iq_grid(Device.DEFAULT, _ggml.iq3s_grid, (512, 4)).numpy()
    np.testing.assert_equal(grid, IQ3_S.grid.reshape(512, 4))

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.half in supported_dtypes, "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def test_load_tinyllama_q8_0(self): self._test_gguf_load("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q8_0.gguf?download=true")
  def test_load_tinyllama_q4_0(self): self._test_gguf_load("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf?download=true")
  def test_load_gpt2_q4_1(self): self._test_gguf_load("https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q4_1.gguf?download=true")
  def test_load_sample_q6_k(self): self._test_gguf_load("https://huggingface.co/Isotr0py/test-gguf-sample/resolve/main/Quant_Q6_K_1024.gguf?download=true")

  def test_dequantization_q8_0_hardcoded(self):
    # Q8_0: 2 bytes float16 scale + 32 bytes int8 values, dequant = scale * values
    block = np.frombuffer(np.float16(2.0).tobytes() + np.arange(1, 33, dtype=np.int8).tobytes(), dtype=np.uint8).copy()
    expected = np.arange(1, 33, dtype=np.float32) * 2.0
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 32, GGMLQuantizationType.Q8_0.value).numpy().flatten(), expected)

  def test_dequantization_q2_k_hardcoded(self):
    # Q2_K: scales[16] + qs[64] + d(fp16) + dmin(fp16). 16 sub-blocks of 16, x = d*(scale&0xF)*q - dmin*(scale>>4)
    scales, qs = bytes([0x11]*16), bytes([0x55]*64)  # scale=1, min=1; qs=0x55 -> 2-bit quants of 1
    d, dmin = np.float16(1.0).tobytes(), np.float16(0.0).tobytes()
    block = np.frombuffer(scales + qs + d + dmin, dtype=np.uint8).copy()
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 256, 10).numpy().flatten(), np.ones(256, dtype=np.float32))

  def test_dequantization_q3_k_hardcoded(self):
    # Q3_K: hmask[32] + qs[64] + scales[12] + d(fp16). 16 sub-blocks of 16, x = d * (scale-32) * (q - (hbit?0:4))
    # 6-bit scales 32..47 so (scale-32) = 0..15; qs=0x55 -> 2-bit quants of 1; d=1.0
    scales = bytes([0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7, 0xAA, 0xAA, 0xAA, 0xAA])
    d = np.float16(1.0).tobytes()
    qs, ones = bytes([0x55]*64), np.ones(16, dtype=np.float32)
    # hmask all-ones: high bit set, q=1; hmask zeros: subtract 4, q=-3
    for hmask, q in ((bytes([0xFF]*32), 1.0), (bytes([0x00]*32), -3.0)):
      block = np.frombuffer(hmask + qs + scales + d, dtype=np.uint8).copy()
      expected = np.concatenate([q * s * ones for s in range(16)])
      np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 256, 11).numpy().flatten(), expected)

  def test_dequantization_iq2_xxs_hardcoded(self):
    # IQ2_XXS: d + 8 groups of (4 grid bytes + uint32 signs/scale). grid[0]=all 0x08, scale=0, signs=0
    # db = 1.0 * (0.5 + 0) * 0.25 = 0.125; 0.125 * 8 = 1.0
    block = np.frombuffer(np.float16(1.0).tobytes() + bytes(64), dtype=np.uint8).copy()
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 256, 16).numpy().flatten(), np.ones(256, dtype=np.float32))

  def test_dequantization_iq2_xs_hardcoded(self):
    # IQ2_XS: d + 32 uint16 qs + 8 scale bytes. qs=0 -> grid[0]=all 0x08, signs=0; scales=0
    block = np.frombuffer(np.float16(1.0).tobytes() + bytes(64) + bytes(8), dtype=np.uint8).copy()
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 256, 17).numpy().flatten(), np.ones(256, dtype=np.float32))

  def test_dequantization_iq1_s_hardcoded(self):
    # IQ1_S: d + qs[32] + qh[16]. qs=qh=0 -> grid[0]=all -1, scale=1, delta=+0.125 -> -0.875
    block = np.frombuffer(np.float16(1.0).tobytes() + bytes(48), dtype=np.uint8).copy()
    expected = np.full(256, -0.875, dtype=np.float32)
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 256, 19).numpy().flatten(), expected)

  def test_dequantization_iq1_m_hardcoded(self):
    # IQ1_M: qs[32] + qh[16] + scales[8]. f16 1.0=0x3C00 packed in high nibbles; qs=qh=0 -> -0.875
    scales = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x30])
    block = np.frombuffer(bytes(48) + scales, dtype=np.uint8).copy()
    expected = np.full(256, -0.875, dtype=np.float32)
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 256, 29).numpy().flatten(), expected)

  def test_dequantization_iq4_nl_hardcoded(self):
    # IQ4_NL: 2-byte fp16 scale + 16 packed bytes. low nibbles first, then high
    lut = list(_ggml.kvalues_iq4nl)
    block = np.frombuffer(np.float16(1.0).tobytes() + bytes(range(16)), dtype=np.uint8).copy()
    expected = np.array(lut + [lut[0]]*16, dtype=np.float32)
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 32, 20).numpy().flatten(), expected)

  def test_dequantization_mxfp4_hardcoded(self):
    # MXFP4: 1 byte shared exponent E + 16 packed bytes (32 x 4-bit values)
    # nibble: bit3=sign, bit2:1=exp, bit0=mant; E=128 gives scale=1.0
    # codes 0-7 = [0, 1, 2, 3, 4, 6, 8, 12], codes 8-15 are their negatives
    block = np.array([0x80] + list(range(16)), dtype=np.uint8)  # E=128, nibbles 0-15 in low, zeros in high
    expected = np.array([0., 1., 2., 3., 4., 6., 8., 12., -0., -1., -2., -3., -4., -6., -8., -12.] + [0.]*16, dtype=np.float32)
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 32, GGMLQuantizationType.MXFP4.value).numpy().flatten(), expected)

  def test_dequantization_q4_0(self): self._test_dequantization(GGMLQuantizationType.Q4_0)
  def test_dequantization_q4_1(self): self._test_dequantization(GGMLQuantizationType.Q4_1)
  def test_dequantization_q5_0(self): self._test_dequantization(GGMLQuantizationType.Q5_0)
  def test_dequantization_q5_1(self): self._test_dequantization(GGMLQuantizationType.Q5_1)
  def test_dequantization_q8_0(self): self._test_dequantization(GGMLQuantizationType.Q8_0)
  def test_dequantization_q2_k(self): self._test_dequantization(GGMLQuantizationType.Q2_K)
  def test_dequantization_q3_k(self): self._test_dequantization(GGMLQuantizationType.Q3_K)
  def test_dequantization_q4_k(self): self._test_dequantization(GGMLQuantizationType.Q4_K)
  def test_dequantization_q5_k(self): self._test_dequantization(GGMLQuantizationType.Q5_K)
  def test_dequantization_q6_k(self): self._test_dequantization(GGMLQuantizationType.Q6_K)
  def test_dequantization_iq2_xxs(self): self._test_dequantization(GGMLQuantizationType.IQ2_XXS)
  def test_dequantization_iq2_xs(self): self._test_dequantization(GGMLQuantizationType.IQ2_XS)
  def test_dequantization_iq3_xxs(self): self._test_dequantization(GGMLQuantizationType.IQ3_XXS)
  def test_dequantization_iq1_s(self): self._test_dequantization(GGMLQuantizationType.IQ1_S)
  def test_dequantization_iq4_nl(self): self._test_dequantization(GGMLQuantizationType.IQ4_NL)
  def test_dequantization_iq3_s(self): self._test_dequantization(GGMLQuantizationType.IQ3_S)
  def test_dequantization_iq2_s(self): self._test_dequantization(GGMLQuantizationType.IQ2_S)
  def test_dequantization_iq4_xs(self): self._test_dequantization(GGMLQuantizationType.IQ4_XS)
  def test_dequantization_iq1_m(self): self._test_dequantization(GGMLQuantizationType.IQ1_M)
  def test_dequantization_mxfp4(self): self._test_dequantization(GGMLQuantizationType.MXFP4)
  @unittest.skipUnless(dtypes.bfloat16 in supported_dtypes, "Backend must support bfloat16")
  def test_dequantization_bf16(self): self._test_dequantization(GGMLQuantizationType.BF16)
  def test_dequantization_mxfp4_old(self):
    def encode(nibbles, E):
      packed = [(low & 0xF) | ((high & 0xF) << 4) for low, high in zip(nibbles[:16], nibbles[16:])]
      return np.array([E] + packed, dtype=np.uint8)

    def decode(code, E):
      sign = -1.0 if (code & 0b1000) else 1.0
      exp = (code >> 1) & 0b11
      mant = code & 0b1
      val = 2 * ((1.0 + 0.5 * mant) * np.exp2(exp - 1) if exp else 0.5 * mant)
      scale = np.exp2(E - 128) if E >= 2 else np.exp2(-127 if E == 1 else -128)
      return sign * val * scale

    blocks, expected = [], []
    rng = np.random.default_rng(42)
    for _ in range(4):
      E = rng.integers(0, 256)
      codes = rng.integers(0, 16, size=32, dtype=np.uint8)
      blocks.append(encode(codes, E))
      expected.extend(decode(c, E) for c in codes)
    tensor = Tensor(np.concatenate(blocks))
    out = ggml_data_to_tensor(tensor, len(expected), GGMLQuantizationType.MXFP4.value)
    np.testing.assert_equal(out.numpy(), expected)

  def test_dequantization_mxfp4_block(self):
    # https://gist.github.com/Ananta-Ranganathan/3317b6ed51a3b033e9c2564fafb4e043
    # used the above script to download the first block of blk.0.attn_k_b.weight from
    # https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/blob/main/GLM-4.7-Flash-MXFP4_MOE.gguf
    # and compute the canonical expected dequantized output with the GGUF PY implementation
    block = np.array([0x7a, 0x29, 0xab, 0x61, 0x10, 0x21, 0x02, 0x4a,
                    0x15, 0xca, 0x05, 0x01, 0x9b, 0x39, 0x0b, 0x0b, 0x1c], dtype=np.uint8)
    expected = np.array([-0.01562500, -0.04687500, 0.01562500, 0.00000000,
                        0.01562500,  0.03125000, -0.03125000, 0.09375000,
                        -0.03125000,  0.09375000, 0.01562500, -0.04687500,
                        -0.01562500, -0.04687500, -0.04687500, -0.06250000,
                        0.03125000, -0.03125000, 0.12500000,  0.01562500,
                        0.03125000,  0.00000000, 0.06250000,  0.01562500,
                        -0.06250000,  0.00000000, 0.00000000, -0.01562500,
                        0.04687500,  0.00000000, 0.00000000,  0.01562500], dtype=np.float32)
    out = ggml_data_to_tensor(Tensor(block), 32, GGMLQuantizationType.MXFP4.value)
    np.testing.assert_equal(out.numpy(), expected)

  def test_dequantization_q1_0(self):
    # Q1_0: 2 bytes fp16 scale + 16 bytes (128 1-bit values)
    block = np.frombuffer(np.float16(2.0).tobytes() + np.packbits(np.random.choice([0, 1], size=128)).tobytes(), dtype=np.uint8).copy()
    expected = np.float16(2.0) * (np.unpackbits(block[2:], bitorder="little").astype(np.int8) * 2 - 1)
    # TODO: replace 41 with GGMLQuantizationType.Q1_0.value on next gguf-py release
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 128, 41).numpy().flatten(), expected)

  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

  @staticmethod
  def _build_gguf(tensors, kvs):
    # [header] [kv_data] [tensor_infos] [padding] [tensor_data_blob]
    buf = bytearray()
    # Header: magic "GGUF" + version=3 + n_tensors + n_kv
    buf += struct.pack("<4siqq", b"GGUF", 3, len(tensors), len(kvs))
    # KV entries: [key_len: uint64][key bytes][type: int32][value]
    for k, v in kvs:
      kb = k.encode()
      if isinstance(v, str): buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<i", 8) + struct.pack("<Q", len(v)) + v.encode()
      else: buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<i", 4) + struct.pack("<I", v)
    data_off = 0
    # Tensor infos: [name_len][name][ndims][dims reversed][qtype][offset_into_data_blob]
    for name, dims, qtype, data in tensors:
      nb = name.encode()
      buf += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
      for d in reversed(dims): buf += struct.pack("<Q", d)
      buf += struct.pack("<i", qtype) + struct.pack("<Q", data_off)
      data_off += len(data)
    buf += b"\x00" * ((32 - len(buf) % 32) % 32)
    for _, _, _, data in tensors: buf += data
    return bytes(buf)

  def test_multi_part_load(self):
    with tempfile.TemporaryDirectory() as d:
      d = pathlib.Path(d)
      a, b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), np.array([5.0, 6.0], dtype=np.float32)
      (d / "test-00001-of-00002.gguf").write_bytes(self._build_gguf([("a", (4,), 0, a.tobytes())], [("split.count", 2), ("split.no", 0)]))
      (d / "test-00002-of-00002.gguf").write_bytes(self._build_gguf([("b", (2,), 0, b.tobytes())], [("split.count", 2), ("split.no", 1)]))
      kv, ts = gguf_load(d / "test-00001-of-00002.gguf")
      self.assertEqual(kv["split.count"], 2)
      np.testing.assert_equal(ts["a"].numpy(), a)
      np.testing.assert_equal(ts["b"].numpy(), b)

      # missing part 2
      (d / "test-00002-of-00002.gguf").unlink()
      with self.assertRaises(FileNotFoundError):
        gguf_load(d / "test-00001-of-00002.gguf")

  def _test_dequantization(self, qtype: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n_el, n_bytes = ggml_test_block_count * block_size, ggml_test_block_count * type_size

    try:
      q_data = quantize((np.random.random((n_el,)).astype(np.float32) * 100 - 50), qtype)
    except NotImplementedError:
      q_data = np.random.default_rng(42).integers(0, 256, size=n_bytes, dtype=np.uint8)
    ref = dequantize(q_data, qtype)

    q_tensor = Tensor(q_data)
    dq_tensor = ggml_data_to_tensor(q_tensor, n_el, qtype.value).reshape(n_el)

    np.testing.assert_equal(dq_tensor.numpy(), ref)

  def _test_gguf_load(self, url: str):
    fp = fetch(url)
    model_size = os.stat(fp).st_size
    gguf_tensor = Tensor.empty(model_size, dtype=dtypes.uint8, device=f"disk:{fp}").to(Device.DEFAULT)
    kv_data, tensors = gguf_load(gguf_tensor)

    reader = GGUFReader(fp)

    for rt in reader.tensors:
      ref = dequantize(rt.data, rt.tensor_type)
      np.testing.assert_equal(tensors[rt.name].numpy(), ref.reshape(tensors[rt.name].shape))

    for k, f in reader.fields.items():
      if k.startswith("GGUF."): continue  # skip file header keys (version, tensor_count, kv_count)
      def read_val(i, parts=f.parts, is_str=(f.types[-1] == GGUFValueType.STRING)):
        return bytes(parts[i]).decode("utf-8") if is_str else parts[i][0].item()
      if f.types[0] == GGUFValueType.ARRAY:
        self.assertEqual(kv_data[k], [read_val(i) for i in f.data])
      else:
        self.assertEqual(kv_data[k], read_val(-1))

class TestGGUFGEMV(unittest.TestCase):
  def _test_gguf_gemv(self, qtype: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    rows, cols = (1024, 512) if qtype == GGMLQuantizationType.BF16 else (8192, 2048)
    n_blocks = rows * cols // block_size
    rng = np.random.default_rng(42)
    if qtype == GGMLQuantizationType.BF16:
      q_data = (rng.standard_normal(rows * cols).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    else:
      # generate random quantized blocks with valid fp16 scale fields (random bytes can produce NaN scales)
      q_data = rng.integers(0, 256, size=n_blocks * type_size, dtype=np.uint8).reshape(n_blocks, type_size)
      scales = np.float16(rng.standard_normal(n_blocks * 4)).view(np.uint8).reshape(n_blocks, -1)
      if qtype in (GGMLQuantizationType.Q5_0, GGMLQuantizationType.Q8_0,
                   GGMLQuantizationType.IQ2_XXS, GGMLQuantizationType.IQ2_XS,
                   GGMLQuantizationType.IQ3_XXS, GGMLQuantizationType.IQ4_NL,
                   GGMLQuantizationType.IQ1_S, GGMLQuantizationType.IQ2_S,
                   GGMLQuantizationType.IQ3_S, GGMLQuantizationType.IQ4_XS): q_data[:, :2] = scales[:, :2]  # d at offset 0
      elif qtype in (GGMLQuantizationType.Q5_1, GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K):
        q_data[:, :4] = scales[:, :4]  # d, m/dmin at offset 0
      elif qtype == GGMLQuantizationType.Q2_K: q_data[:, -4:] = scales[:, :4]  # d, dmin at end
      elif qtype in (GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q3_K): q_data[:, -2:] = scales[:, :2]  # d at end
      elif qtype == GGMLQuantizationType.IQ1_M:
        s = np.float16(rng.standard_normal(n_blocks)).view(np.uint16)
        sc = q_data[:, -8:].copy().view(np.uint16).reshape(n_blocks, 4)
        sc &= np.uint16(0x0FFF)
        sc[:, 0] |= (s & np.uint16(0x000F)) << 12
        sc[:, 1] |= (s & np.uint16(0x00F0)) << 8
        sc[:, 2] |= (s & np.uint16(0x0F00)) << 4
        sc[:, 3] |= (s & np.uint16(0xF000))
        q_data[:, -8:] = sc.reshape(n_blocks, -1).view(np.uint8)
      elif qtype == GGMLQuantizationType.MXFP4: q_data[:, 0] = rng.integers(120, 136, size=n_blocks, dtype=np.uint8) # constrain byte0
      q_data = q_data.flatten()
    ref = dequantize(q_data, qtype).reshape(rows, cols)

    # build a minimal gguf in memory: header + 1 tensor info + aligned data
    buf = bytearray()
    buf += struct.pack("<4siqq", b"GGUF", 3, 1, 0)              # magic, version, n_tensors, n_kv
    buf += struct.pack("<Q", 6) + b"weight"                      # tensor name
    buf += struct.pack("<I", 2)                                  # ndims
    buf += struct.pack("<QQ", cols, rows)                        # dims (gguf stores reversed)
    buf += struct.pack("<i", qtype.value)
    buf += struct.pack("<Q", 0)                                  # offset
    buf += b"\x00" * ((32 - len(buf) % 32) % 32)                # pad to alignment=32
    buf += q_data.tobytes()

    _, tensors = gguf_load(Tensor(np.frombuffer(buf, dtype=np.uint8)).to(None))

    x = rng.standard_normal(cols).astype(np.float32)
    with np.errstate(all='ignore'):
      np.testing.assert_allclose((tensors["weight"] @ Tensor(x)).numpy(), ref @ x, atol=1e-2, rtol=1e-2)
    if qtype == GGMLQuantizationType.BF16 or dtypes.half in supported_dtypes: np.testing.assert_equal(tensors["weight"].numpy(), ref)
    assert np.isfinite(ref).all() and np.isfinite(tensors["weight"].numpy()).all(), f"{qtype.name} has NaN/Inf"

  def test_gguf_gemv_q8_0(self): self._test_gguf_gemv(GGMLQuantizationType.Q8_0)
  def test_gguf_gemv_q5_0(self): self._test_gguf_gemv(GGMLQuantizationType.Q5_0)
  def test_gguf_gemv_q5_1(self): self._test_gguf_gemv(GGMLQuantizationType.Q5_1)
  def test_gguf_gemv_q2_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q2_K)
  def test_gguf_gemv_q3_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q3_K)
  def test_gguf_gemv_q4_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q4_K)
  def test_gguf_gemv_q5_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q5_K)
  def test_gguf_gemv_q6_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q6_K)
  def test_gguf_gemv_iq2_xxs(self): self._test_gguf_gemv(GGMLQuantizationType.IQ2_XXS)
  def test_gguf_gemv_iq2_xs(self): self._test_gguf_gemv(GGMLQuantizationType.IQ2_XS)
  def test_gguf_gemv_iq3_xxs(self): self._test_gguf_gemv(GGMLQuantizationType.IQ3_XXS)
  def test_gguf_gemv_iq1_s(self): self._test_gguf_gemv(GGMLQuantizationType.IQ1_S)
  def test_gguf_gemv_iq4_nl(self): self._test_gguf_gemv(GGMLQuantizationType.IQ4_NL)
  def test_gguf_gemv_iq3_s(self): self._test_gguf_gemv(GGMLQuantizationType.IQ3_S)
  def test_gguf_gemv_iq2_s(self): self._test_gguf_gemv(GGMLQuantizationType.IQ2_S)
  def test_gguf_gemv_iq4_xs(self): self._test_gguf_gemv(GGMLQuantizationType.IQ4_XS)
  def test_gguf_gemv_iq1_m(self): self._test_gguf_gemv(GGMLQuantizationType.IQ1_M)
  def test_gguf_gemv_mxfp4(self): self._test_gguf_gemv(GGMLQuantizationType.MXFP4)
  @unittest.skipUnless(dtypes.bfloat16 in supported_dtypes, "Backend must support bfloat16")
  def test_gguf_gemv_bf16(self): self._test_gguf_gemv(GGMLQuantizationType.BF16)

class TestGGUFGC(unittest.TestCase):
  def test_gguf_load_no_tensor_leak(self):
    """gguf_load must not retain references to the input tensor after returning."""
    fp = fetch("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q8_0.gguf?download=true")
    t = Tensor.empty(os.stat(fp).st_size, dtype=dtypes.uint8, device=f"disk:{fp}").to(Device.DEFAULT).realize()
    with disable_gc():
      ref_before = sys.getrefcount(t)
      kv_data, tensors = gguf_load(t)
      self.assertEqual(sys.getrefcount(t), ref_before, "gguf_load leaked a reference to the input tensor")

if __name__ == '__main__':
  unittest.main()
