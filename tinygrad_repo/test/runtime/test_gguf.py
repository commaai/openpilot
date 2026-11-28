import gc, os, struct, unittest, tempfile, pathlib, sys, weakref
from tinygrad import dtypes, Tensor, fetch, Device
from tinygrad.helpers import disable_gc, Context
from tinygrad.llm.gguf import _ggml_iq_grid, _ggml_iq_signs, ggml_data_to_tensor, gguf_parse, gguf_load
from tinygrad.runtime.autogen import ggml_common as _ggml
import numpy as np
from gguf import GGUFReader, GGUFValueType, GGMLQuantizationType, GGML_QUANT_SIZES, dequantize, quantize
from gguf.quants import IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS

ggml_test_block_count = 4
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

class TestGGUFTables(unittest.TestCase):
  def test_iq_tables_not_retained(self):
    for make_table in (lambda: _ggml_iq_signs(Device.DEFAULT), lambda: _ggml_iq_grid(Device.DEFAULT, _ggml.iq3s_grid, (512, 4))):
      table = make_table().realize()
      ref = weakref.ref(table)
      del table
      gc.collect()
      self.assertIsNone(ref())

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
  def test_load_sample_q6_k(self): self._test_gguf_load("https://huggingface.co/Isotr0py/test-gguf-sample/resolve/main/Quant_Q6_K_1024.gguf?download=true")

  def test_dequantization_q8_0_hardcoded(self):
    # Q8_0: 2 bytes float16 scale + 32 bytes int8 values, dequant = scale * values
    block = np.frombuffer(np.float16(2.0).tobytes() + np.arange(1, 33, dtype=np.int8).tobytes(), dtype=np.uint8).copy()
    expected = np.arange(1, 33, dtype=np.float32) * 2.0
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 32, GGMLQuantizationType.Q8_0.value).numpy().flatten(), expected)

  def test_dequantization_iq1_s_hardcoded(self):
    # IQ1_S: d + qs[32] + qh[16]. qs=qh=0 -> grid[0]=all -1, scale=1, delta=+0.125 -> -0.875
    block = np.frombuffer(np.float16(1.0).tobytes() + bytes(48), dtype=np.uint8).copy()
    expected = np.full(256, -0.875, dtype=np.float32)
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 256, 19).numpy().flatten(), expected)

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
  def test_dequantization_nvfp4(self): self._test_dequantization(GGMLQuantizationType.NVFP4)
  @unittest.skipUnless(dtypes.bfloat16 in supported_dtypes, "Backend must support bfloat16")
  def test_dequantization_bf16(self): self._test_dequantization(GGMLQuantizationType.BF16)
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

  def test_parse(self):
    f32, f16 = np.arange(8, dtype=np.float32).reshape(2, 4), np.arange(16, dtype=np.float16).reshape(2, 8)
    q8 = (np.float16(2.0).tobytes() + np.arange(32, dtype=np.int8).tobytes())*2
    tensors = [("f32", f32.shape, 0, f32.tobytes()), ("f16", f16.shape, 1, f16.tobytes()), ("q8", (2, 32), 8, q8)]
    expected = {"f32": f32, "f16": f16, "q8": np.tile(np.arange(32, dtype=np.float32)*2, (2, 1))}
    # Neither following tensors nor trailing file padding belong in the packed views.
    blob = self._build_gguf(tensors, [("general.name", "test")]) + bytes(32)
    with tempfile.TemporaryDirectory() as d:
      (path := pathlib.Path(d) / "test.gguf").write_bytes(blob)
      for source in (path, str(path), Tensor(path), Tensor(blob, device="CPU")):
        with self.subTest(source=repr(source)):
          kv, entries = gguf_parse(source)
          self.assertEqual(kv, {"general.name": "test"})
          self.assertEqual(set(entries), set(expected))
          for name, shape, typ, packed in tensors:
            data, parsed_shape, parsed_typ = entries[name]
            self.assertEqual((parsed_shape, parsed_typ), (shape, typ))
            self.assertEqual((data.shape, data.dtype), ((len(packed),), dtypes.uint8))
            self.assertEqual(data.device, source.device if isinstance(source, Tensor) else f"DISK:{path.resolve()}")
            self.assertEqual(data.data().tobytes(), packed)
          loaded_kv, decoded = gguf_load(source)
          self.assertEqual(loaded_kv, kv)
          for name, value in expected.items(): np.testing.assert_equal(decoded[name].numpy(), value)

  def test_load_weight_scale(self):
    from gguf import GGUFWriter
    from tinygrad.helpers import get_child
    from tinygrad.llm.model import Transformer
    block = {'attn_norm': (8,), 'attn_q': (8, 8), 'attn_k': (8, 8), 'attn_v': (8, 8), 'attn_output': (8, 8), 'ffn_norm': (8,),
             'ffn_gate_inp': (4, 8), 'ffn_gate_exps': (4, 8, 8), 'ffn_up_exps': (4, 8, 8), 'ffn_down_exps': (4, 8, 8)}
    shapes = {'token_embd.weight': (32, 8), 'output_norm.weight': (8,), **{f'blk.0.{k}.weight': s for k, s in block.items()}}
    rng = np.random.default_rng(0)
    weights = {k:rng.normal(size=s).astype(np.float32) for k,s in shapes.items()}
    scales = {f'blk.0.ffn_{p}_exps.scale': np.array([1, 2, 3, 4], np.float32) * (i+1) * 1e-6 for i, p in enumerate(('gate', 'up', 'down'))}
    scales['blk.0.attn_output.scale'] = np.array([5e-6], np.float32)
    with tempfile.TemporaryDirectory() as folder:
      writer = GGUFWriter(path:=pathlib.Path(folder)/'model.gguf', 'deepseek2')
      for key,value in {'context_length':16, 'embedding_length':8, 'expert_feed_forward_length':8, 'block_count':1, 'expert_count':4,
                        'expert_used_count':2, 'attention.head_count':4, 'attention.head_count_kv':4, 'attention.key_length':2,
                        'rope.dimension_count':2}.items():
        writer.add_uint32('deepseek2.'+key, value)
      writer.add_float32('deepseek2.rope.freq_base', 10000)
      writer.add_float32('deepseek2.attention.layer_norm_rms_epsilon', 1e-5)
      writer.add_array('tokenizer.ggml.tokens', [str(i) for i in range(32)])
      for name,value in {**weights, **scales}.items(): writer.add_tensor(name, value)
      writer.write_header_to_file()
      writer.write_kv_data_to_file()
      writer.write_tensors_to_file()
      writer.close()
      with Context(DEV="CPU"):
        model = Transformer.from_gguf(path)[0]
        for k, v in scales.items(): np.testing.assert_equal(get_child(model, k.removesuffix('.scale')).scale.numpy(), v)
        self.assertIsNone(model.blk[0].attn_q.scale)

  def test_parse_empty(self):
    blob = self._build_gguf([], [("general.name", "empty")])
    for loader in (gguf_parse, gguf_load):
      self.assertEqual(loader(Tensor(blob, device="CPU")), ({"general.name": "empty"}, {}))

  def test_parse_unsupported_type(self):
    blob = self._build_gguf([("a", (256,), 1337, bytes(256))], [])
    for loader in (gguf_parse, gguf_load):
      with self.assertRaisesRegex(ValueError, "GGML type '1337' is not supported"):
        loader(Tensor(blob, device="CPU"))

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
      parsed_kv, entries = gguf_parse(d / "test-00001-of-00002.gguf")
      self.assertEqual(parsed_kv, kv)
      for i, (name, value) in enumerate((("a", a), ("b", b)), 1):
        data, shape, typ = entries[name]
        self.assertEqual((shape, typ, data.nbytes()), (value.shape, 0, value.nbytes))
        self.assertEqual(data.device, f"DISK:{(d / f'test-{i:05d}-of-00002.gguf').resolve()}")
        self.assertEqual(data.data().tobytes(), value.tobytes())
      for loader in (gguf_parse, gguf_load):
        with self.assertRaisesRegex(ValueError, "requires a path argument"):
          loader(Tensor(d / "test-00001-of-00002.gguf"))
        with self.assertRaisesRegex(ValueError, "must be loaded from the first split"):
          loader(d / "test-00002-of-00002.gguf")

      # missing part 2
      (d / "test-00002-of-00002.gguf").unlink()
      for loader in (gguf_parse, gguf_load):
        with self.assertRaises(FileNotFoundError): loader(d / "test-00001-of-00002.gguf")

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

    # Check every value, dequantizing both implementations in bounded, block-aligned chunks.
    # Realize in batches so same-shape dequants fuse into few kernels; .contiguous() gives each chunk its own
    # store, since realizing a bare shrink would compute the whole parent tensor (exceeds WEBGPU buffer limits).
    # LRU=0: the many distinct chunk sizes would otherwise accumulate in the allocator cache (100s of MB of RSS).
    pending: list[tuple[Tensor, np.ndarray]] = []
    pending_bytes = 0
    def flush():
      nonlocal pending_bytes
      if not pending: return
      Tensor.realize(*[c for c, _ in pending])
      for c, ref in pending: np.testing.assert_equal(c.numpy(), ref)
      pending.clear()
      pending_bytes = 0
    with Context(LRU=0), open(fp, 'rb') as f:  # read raw bytes directly: the reader's memmap faults the whole file into RAM
      for rt in reader.tensors:
        t = tensors[rt.name].flatten()
        block_size, type_size = GGML_QUANT_SIZES[rt.tensor_type]
        chunk = (8 << 20) // t.dtype.itemsize
        for start in range(0, t.numel(), chunk):
          if pending_bytes >= (64 << 20): flush()
          end = min(start+chunk, t.numel())
          f.seek(rt.data_offset + start//block_size*type_size)
          ref = dequantize(np.frombuffer(f.read((end-start)//block_size*type_size), dtype=np.uint8), rt.tensor_type)
          pending.append((t[start:end].contiguous(), ref.reshape(-1)))
          pending_bytes += (end-start) * t.dtype.itemsize
    flush()

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
    rows, cols = 16, 512
    n_blocks = rows * cols // block_size
    rng = np.random.default_rng(42)
    if qtype == GGMLQuantizationType.BF16:
      q_data = (rng.standard_normal(rows * cols).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    else:
      # generate random quantized blocks with valid fp16 scale fields (random bytes can produce NaN scales)
      q_data = rng.integers(0, 256, size=n_blocks * type_size, dtype=np.uint8).reshape(n_blocks, type_size)
      scales = np.float16(rng.standard_normal(n_blocks * 4)).view(np.uint8).reshape(n_blocks, -1)
      if qtype == GGMLQuantizationType.Q8_0: q_data[:, :2] = scales[:, :2]  # d at offset 0
      elif qtype == GGMLQuantizationType.Q6_K: q_data[:, -2:] = scales[:, :2]  # d at end
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
  def test_gguf_gemv_q6_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q6_K)
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
