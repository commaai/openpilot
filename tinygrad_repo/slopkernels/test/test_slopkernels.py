import ast, pathlib, subprocess, sys, unittest
from unittest.mock import patch

import numpy as np
from tinygrad import Context, Device, Tensor, UOp, dtypes
from tinygrad.helpers import DEV
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops

import slopkernels
from slopkernels.bench import _krun


def _amd_target() -> tuple[int, ...]|None:
  dev = Tensor.empty(1).device
  if not isinstance(dev, str) or dev.split(":")[0] != "AMD": return None
  t = getattr(Device[dev], "target", None)
  return t if t is not None and t[0] in (9, 11, 12) and isinstance(Device[dev].renderer, HIPRenderer) else None


def _make_q8_0(rng:np.random.Generator, rows:int, cols:int) -> tuple[Tensor, Tensor, Tensor]:
  raw_np = np.zeros((rows * cols // 32, 34), dtype=np.uint8)
  raw_np[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
  raw_np[:, 2:] = rng.integers(-60, 60, size=(rows * cols // 32, 32), dtype=np.int8).view(np.uint8)
  raw_u8 = Tensor(raw_np.reshape(-1)).realize()
  w_ref = ggml_data_to_tensor(raw_u8, rows * cols, 8).reshape(rows, cols).realize()
  return raw_u8, raw_u8.bitcast("uint16").contiguous().realize(), w_ref


class TestSlopKernelsUIR(unittest.TestCase):
  def test_narrow_api_and_purity_boundary(self):
    """slopkernels exposes only {get, generate_all}, uses pure UOp builders only, and tinygrad only calls get(name) in llm/model.py."""
    self.assertEqual(set(slopkernels.__all__), {"generate_all", "get"})
    pkg_dir = pathlib.Path(slopkernels.__file__).resolve().parent
    forbidden_names, forbidden_attrs = {"Tensor", "Buffer", "Device", "getenv", "nn"}, {"realize", "numpy", "item"}
    for p in [pkg_dir / "__init__.py", *(pkg_dir / "src").glob("*.py")]:
      tree = ast.parse(p.read_text())
      for node in ast.walk(tree):
        if isinstance(node, ast.Name): self.assertNotIn(node.id, forbidden_names, f"{p.name} must be pure UOp (found {node.id})")
        elif isinstance(node, ast.ImportFrom):
          for alias in node.names: self.assertNotIn(alias.name, forbidden_names, f"{p.name} must not import {alias.name}")
        elif isinstance(node, ast.Attribute): self.assertNotIn(node.attr, forbidden_attrs, f"{p.name} must not call .{node.attr}()")
    import tinygrad
    root = pathlib.Path(tinygrad.__file__).resolve().parent
    model_py = (root / "llm" / "model.py").resolve()
    for p in root.rglob("*.py"):
      if "slopkernels" not in (txt := p.read_text()): continue
      tree = ast.parse(txt)
      slop_imports = [n for n in ast.walk(tree)
                      if (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("slopkernels"))
                      or (isinstance(n, ast.Import) and any(a.name.startswith("slopkernels") for a in n.names))]
      if p.resolve() == model_py:
        self.assertEqual(len(slop_imports), 1)
        imp = slop_imports[0]
        assert isinstance(imp, ast.ImportFrom)
        self.assertEqual((imp.module, [a.name for a in imp.names]), ("slopkernels", ["get"]))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_load_kernel"):
          calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "get"]
          self.assertEqual(len(calls), 1)
          self.assertEqual((len(calls[0].args), len(calls[0].keywords)), (1, 0))
      else:
        self.assertEqual(slop_imports, [], f"{p} must not import slopkernels")
    tracked_uir = subprocess.run(["git", "ls-files", "*.uir"], cwd=str(root.parent), capture_output=True, text=True, check=False)
    if tracked_uir.returncode == 0:
      self.assertEqual(tracked_uir.stdout.strip(), "", f".uir files must not be tracked in git:\n{tracked_uir.stdout}")

  def test_generate_all_and_get_disk_roundtrip(self):
    """slopkernels.get(name) writes fresh color-free .uir to tinygrad/llm/kernels/ and round-trips through disk; release fallback reads disk .uir."""
    import tinygrad
    from tinygrad.llm import model as llm_model
    uir_dir = slopkernels.UIR_DIR
    self.assertEqual(uir_dir, (pathlib.Path(tinygrad.__file__).resolve().parent / "llm" / "kernels").resolve())
    sample = uir_dir / "cdna_gemm_f16.uir"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("stale garbage\n", encoding="utf-8")
    slopkernels.get.cache_clear()
    u = slopkernels.get("cdna_gemm_f16")
    self.assertIsNotNone(u)
    assert u is not None
    self.assertIs(u.op, Ops.SINK)
    self.assertTrue(any(x.is_template for x in u.toposort()))
    self.assertNotIn("stale garbage", sample.read_text(encoding="utf-8"))
    self.assertNotIn("\x1b", sample.read_text(encoding="utf-8"))
    self.assertIsNone(slopkernels.get("nonexistent_llm_kernel"))
    all_uirs = slopkernels.generate_all()
    # Release mode: slopkernels absent, tinygrad/llm/kernels/<name>.uir present -> loaded & executable directly from disk.
    has_amd = _amd_target() is not None
    llm_model._load_kernel.cache_clear()
    llm_model._get_kernel.cache_clear()
    got_hc = None
    with patch.dict(sys.modules, {"slopkernels": None}):
      u_rel = llm_model._load_kernel("cdna_gemm_f16")
      self.assertIsNotNone(u_rel)
      assert u_rel is not None
      self.assertIs(u_rel.op, Ops.SINK)
      self.assertTrue(any(x.is_template for x in u_rel.toposort()))
      self.assertIsNone(llm_model._load_kernel("nonexistent_llm_kernel"))
      if has_amd:
        dev = Tensor.empty(1).device
        rng = np.random.default_rng(7)
        mix = Tensor(rng.normal(0, 0.2, (2, 24)).astype(np.float32)).realize()
        flat = Tensor(rng.normal(0, 0.5, (2, 128)).astype(np.float32)).realize()
        base = Tensor(rng.normal(0, 0.2, (24,)).astype(np.float32)).realize()
        scale = Tensor(rng.uniform(0.5, 1.5, (3,)).astype(np.float32)).realize()
        got_hc = llm_model._run_kernel("amd_hc_coeffs", (2, 24), dev, mix, flat, base, scale,
                                       tokens=2, features=128, iters=2, eps=1e-6, norm_eps=1e-6)
    if has_amd:
      self.assertIsNotNone(got_hc)
      assert got_hc is not None
      np.testing.assert_allclose(got_hc.numpy()[:, 8:].reshape(2, 4, 4).sum(-1), 1.0, rtol=2e-2, atol=2e-2)
    llm_model._load_kernel.cache_clear()
    llm_model._get_kernel.cache_clear()
    self.assertEqual(len(all_uirs), len(slopkernels._BUILDERS))
    self.assertEqual(len(list(uir_dir.glob("*.uir"))), len(all_uirs))
    self.assertLess(sum(len(v) for v in all_uirs.values()), 1_000_000)
    for name, uir in all_uirs.items():
      self.assertNotIn("\x1b", uir)
      self.assertIn("addrspace=TEMPLATE", uir)
      self.assertEqual((uir_dir / f"{name}.uir").read_text(encoding="utf-8").rstrip("\n"), uir)

  def test_templated_llm_kernels_execute_exact(self):
    if (tgt := _amd_target()) is None: self.skipTest("AMD required")
    dev = Tensor.empty(1).device
    assert dev is not None
    rng = np.random.default_rng(23)
    arch_gemm = "cdna" if tgt[0] == 9 else "rdna4" if tgt[0] == 12 else "rdna3"
    g0 = Tensor([0], dtype=dtypes.uint32).realize()
    # 1. Dense GEMV
    w_d = Tensor(rng.standard_normal((64, 128)).astype(np.float16)).realize()
    x_d = Tensor(rng.standard_normal((2, 128)).astype(np.float32)).realize()
    got_d = _krun("amd_gemv_f16_f32", (2, 64), dev, w_d, x_d, tokens=2, out_features=64, in_blks=1, val_chunk=4).numpy()
    np.testing.assert_allclose(got_d, x_d.numpy() @ w_d.numpy().astype(np.float32).T, rtol=2e-3, atol=2e-3)
    # 2. Quantized GEMV (direct) & WMMA (M=16)
    w_u8, _, w_ref = _make_q8_0(rng, 64, 1280)
    w_bytes = int(w_u8.numel())
    sel0 = Tensor([0], dtype=dtypes.int32).realize()
    x_q1 = Tensor(rng.standard_normal((2, 1280)).astype(np.float32)).realize()
    ref_q1 = (x_q1 @ w_ref.T).numpy()
    got_q1 = _krun("amd_gemv_quant_f32", (2, 64), dev, w_u8, x_q1, sel0, g0, qtype=8,
                   tokens=2, x_rows=2, out_features=64, in_features=1280, raw_bytes=w_bytes,
                   group_rows=0, use_sel=0, x_share=1, ndev=1, arch=tgt[0], num_experts=0).numpy()
    np.testing.assert_allclose(got_q1, ref_q1, rtol=3e-2, atol=2e-2 * np.abs(ref_q1).max())
    x_q16 = Tensor(rng.standard_normal((16, 1280)).astype(np.float16)).realize()
    ref_q16 = (x_q16.float() @ w_ref.T).numpy()
    got_q16 = _krun(f"{arch_gemm}_gemm_f16", (16, 64), dev, w_u8, x_q16, g0, qtype=8,
                    m_blks=1, m_tiles=1, out_features=64, in_features=1280, raw_bytes=w_bytes).numpy()
    np.testing.assert_allclose(got_q16, ref_q16, rtol=2e-2, atol=2e-2 * np.abs(ref_q16).max())
    # 3. MoE decode
    mw_u8, _, mw_ref = _make_q8_0(rng, 4 * 16, 256)
    mw_ref_np = mw_ref.numpy().reshape(4, 16, 256)
    sel_m = Tensor(np.array([2, 0], dtype=np.int32)).realize()
    x_m = Tensor(rng.standard_normal((1, 256)).astype(np.float32)).realize()
    got_m = _krun("amd_gemv_quant_f32", (2, 16), dev, mw_u8, x_m, sel_m, g0, qtype=8,
                  tokens=2, x_rows=1, out_features=16, in_features=256, raw_bytes=int(mw_u8.numel()), group_rows=0,
                  use_sel=1, x_share=2, ndev=1, arch=tgt[0], num_experts=4).numpy()
    for j, eid in enumerate((2, 0)):
      np.testing.assert_allclose(got_m[j], mw_ref_np[eid] @ x_m.numpy()[0], rtol=3e-2, atol=5e-2)
    # 4. Flash attention decode + prefill
    kv_fa = Tensor(rng.standard_normal((2, 1, 2, 128, 64)).astype(np.float16)).realize()
    for group in (2, 3, 5, 6, 7):
      with self.subTest(group=group):
        shape = (1, 2 * group, 1, 64)
        q_dec = Tensor(rng.standard_normal(shape).astype(np.float16)).realize()
        got_fd = _krun("amd_flash_decode", shape, dev, q_dec, kv_fa, UOp.const(53),
                       B=1, H_KV=2, G=group, d_blks=1, n_blks=2, C=2, valid_kv_offset=0).numpy()
        ref_fd = q_dec.float().scaled_dot_product_attention(kv_fa[0, :, :, :53].float(), kv_fa[1, :, :, :53].float(), enable_gqa=True).numpy()
        np.testing.assert_allclose(got_fd, ref_fd, rtol=5e-3, atol=5e-3)
    if tgt[0] != 9:
      q_pre = Tensor(rng.standard_normal((1, 4, 32, 64)).astype(np.float16)).realize()
      got_fp = _krun(f"{arch_gemm}_flash_attn", (1, 4, 32, 64), dev, q_pre, kv_fa, UOp.const(64), UOp.const(32),
                     B=1, H=4, H_KV=2, m_blks=1, d_blks=1, n_blks=4, valid_kv_offset=0, q_start_from_end=1)
      mask = Tensor.full((32, 64), float("-inf"), dtype=dtypes.float32).triu(64 - 32 + 1)
      ref_fp = q_pre.float().scaled_dot_product_attention(kv_fa[0, :, :, :64].float(), kv_fa[1, :, :, :64].float(), attn_mask=mask, enable_gqa=True)
      np.testing.assert_allclose(got_fp.numpy(), ref_fp.numpy(), rtol=5e-3, atol=5e-3)
    # 5. Gated delta prefill (start_pos=0 resets, start_pos=3 preserves)
    gd_q = Tensor(rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32)).realize()
    gd_k = Tensor(rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32)).realize()
    gd_v = Tensor(rng.standard_normal((1, 2, 4, 8)).astype(np.float32)).realize()
    gd_b = Tensor(rng.uniform(0, 1, (1, 2, 4)).astype(np.float32)).realize()
    gd_a = Tensor(rng.uniform(0.8, 1.0, (1, 2, 4)).astype(np.float32)).realize()
    init_np = rng.normal(0, 0.2, (1, 2, 8, 32)).astype(np.float32)
    for sp in (0, 3):
      gd_s = Tensor(init_np.copy()).contiguous().realize()
      st_ref = np.zeros_like(init_np) if sp == 0 else init_np.copy()
      outs_ref = []
      for t in range(4):
        st_ref *= gd_a.numpy()[:, :, t, None, None]
        delta = (gd_v.numpy()[:, :, t] - (st_ref * gd_k.numpy()[:, :, t, None]).sum(-1)) * gd_b.numpy()[:, :, t, None]
        st_ref += delta[..., None] * gd_k.numpy()[:, :, t, None]
        outs_ref.append((st_ref * gd_q.numpy()[:, :, t, None]).sum(-1))
      got_gd = _krun("amd_gated_delta", (1, 2, 4, 8), dev, gd_q, gd_k, gd_v, gd_b, gd_a.reshape(1, 2, 4, 1), gd_s,
                     UOp.const(sp), batch=1, heads=2, tokens=4, v_tiles=2, k_blks=1, alpha_k=1, arch=tgt[0]).numpy()
      np.testing.assert_allclose(got_gd, np.stack(outs_ref, axis=2), rtol=2e-4, atol=2e-6)
      np.testing.assert_allclose(gd_s.numpy(), st_ref, rtol=2e-4, atol=2e-6)
    # 6. mHC coeffs
    hc_mix = Tensor(rng.normal(0, 0.2, (2, 24)).astype(np.float32)).realize()
    hc_flat = Tensor(rng.standard_normal((2, 128)).astype(np.float32)).realize()
    hc_base = Tensor(rng.normal(0, 0.2, (24,)).astype(np.float32)).realize()
    hc_scale = Tensor(rng.uniform(0.5, 1.5, (3,)).astype(np.float32)).realize()
    got_hc = _krun("amd_hc_coeffs", (2, 24), dev, hc_mix, hc_flat, hc_base, hc_scale,
                   tokens=2, features=128, iters=2, eps=1e-6, norm_eps=1e-6).numpy()
    np.testing.assert_allclose(got_hc[:, 8:].reshape(2, 4, 4).sum(-1), 1.0, rtol=2e-2, atol=2e-2)

  def test_multi_gpu_tp(self):
    if _amd_target() is None or not DEV.interface.startswith("MOCK"): self.skipTest("MOCK AMD required for 2-GPU TP")
    devs, rng = ("AMD:0", "AMD:1"), np.random.default_rng(31)
    sel0 = Tensor([0], dtype=dtypes.int32).shard(devs, axis=None).realize()
    g0 = Tensor([0], dtype=dtypes.uint32).shard(devs, axis=None).realize()
    # 1. Column-sharded GEMV
    w_u8, _, w_ref = _make_q8_0(rng, 64, 256)
    w_col = w_u8.reshape(64, -1).shard(devs, axis=0).flatten().realize()
    x_rep = Tensor(rng.standard_normal((2, 256)).astype(np.float32)).shard(devs, axis=None).realize()
    got_col = _krun("amd_gemv_quant_f32", (2, 32), devs, w_col, x_rep, sel0, g0, axis=1, qtype=8,
                    tokens=2, x_rows=2, out_features=32, in_features=256, raw_bytes=int(w_col.uop.max_shard_shape[0]),
                    group_rows=0, use_sel=0, x_share=1, ndev=1, arch=12, num_experts=0)
    np.testing.assert_allclose(got_col.numpy(), (x_rep.to("AMD:0") @ w_ref.T).numpy(), rtol=3e-2, atol=5e-2)
    # 2. Row-sharded GEMV (split-K all-reduce)
    w_u8_r, _, w_ref_r = _make_q8_0(rng, 32, 512)
    w_row = w_u8_r.reshape(32, 2, -1).permute(1, 0, 2).contiguous().shard(devs, axis=0).flatten().realize()
    x_sh = Tensor(rng.standard_normal((2, 512)).astype(np.float32)).shard(devs, axis=1).realize()
    ref_row = (x_sh.to("AMD:0") @ w_ref_r.T).numpy()
    got_row = _krun("amd_gemv_quant_f32", (2, 32), devs, w_row, x_sh, sel0, g0, axis=0, qtype=8,
                    tokens=2, x_rows=2, out_features=32, in_features=256, raw_bytes=int(w_row.uop.max_shard_shape[0]),
                    group_rows=0, use_sel=0, x_share=1, ndev=1, arch=12, num_experts=0).reshape(2, 2, 32).sum(0)
    np.testing.assert_allclose(got_row.numpy(), ref_row, rtol=3e-2, atol=2e-2 * np.abs(ref_row).max())
    # 3. Expert-sharded MoE
    mw_u8, _, mw_ref = _make_q8_0(rng, 4 * 16, 256)
    mw_sh = mw_u8.reshape(4, -1).shard(devs, axis=0).flatten().realize()
    sel_sh = Tensor(np.array([3, 1], dtype=np.int32)).shard(devs, axis=None).realize()
    xm_sh = Tensor(rng.standard_normal((1, 256)).astype(np.float32)).shard(devs, axis=None).realize()
    got_ms = _krun("amd_gemv_quant_f32", (2, 16), devs, mw_sh, xm_sh, sel_sh, g0, axis=0, qtype=8, tokens=2, x_rows=1,
                   out_features=16, in_features=256, raw_bytes=int(mw_sh.uop.max_shard_shape[0]), group_rows=0,
                   use_sel=1, x_share=2, ndev=2, arch=12, num_experts=4).reshape(2, 2, 16).sum(0).numpy()
    for j, eid in enumerate((3, 1)):
      np.testing.assert_allclose(got_ms[j], mw_ref.numpy().reshape(4, 16, 256)[eid] @ xm_sh.to("AMD:0").numpy()[0], rtol=3e-2, atol=5e-2)
    # 4. Head-sharded flash decode
    q_fa = Tensor(rng.standard_normal((1, 4, 1, 64)).astype(np.float16)).realize()
    kv_fa = Tensor(rng.standard_normal((2, 1, 2, 128, 64)).astype(np.float16)).realize()
    q_sh, kv_sh = q_fa.shard(devs, axis=1).realize(), kv_fa.shard(devs, axis=2).realize()
    got_fs = _krun("amd_flash_decode", (1, 2, 1, 64), devs, q_sh, kv_sh, UOp.const(41), axis=1,
                   B=1, H_KV=1, G=2, d_blks=1, n_blks=2, C=2, valid_kv_offset=0).numpy()
    ref_fs = q_fa.float().scaled_dot_product_attention(kv_fa[0, :, :, :41].float(), kv_fa[1, :, :, :41].float(), enable_gqa=True).numpy()
    np.testing.assert_allclose(got_fs, ref_fs, rtol=5e-3, atol=5e-3)

  def test_regression_guards(self):
    if (tgt := _amd_target()) is None: self.skipTest("AMD HIP renderer required")
    from tinygrad.llm.model import Linear, TransformerBlock, TransformerConfig
    from slopkernels.bench import _iq_lut, _make_packed, _packed_words
    dev, rng = Tensor.empty(1).device, np.random.default_rng(19)
    assert isinstance(dev, str)
    def _knames(t:Tensor) -> list[str]:
      with Context(ALLOW_DEVICE_USAGE=0): linear, _ = t.linear_with_externs()
      return [c.src[0].arg.name if c.src[0].arg else "" for c in linear.src if c.src[0].op is Ops.SINK]
    # 1. RDNA4 OTF FP8 IQ3_S GEMM accuracy across a 7.5x superblock scale jump (would clip >448 without sc_m scale)
    if tgt[0] == 12:
      m, n, k = 128, 2048, 512
      qtype, w_len, w_dt, _ = _packed_words(n, k, "IQ3_S")
      _, tile_u8 = _make_packed(n, k, "IQ3_S", dev)
      raw_tile = tile_u8.numpy().reshape(tile_u8.shape[0], k // 256, 110).copy()
      raw_tile[:, 0, :2], raw_tile[:, 0, 106:110] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8), 0x00
      raw_tile[:, 1, :2], raw_tile[:, 1, 106:110] = np.frombuffer(np.float16(0.075).tobytes(), dtype=np.uint8), 0xff
      w_np = np.broadcast_to(raw_tile.reshape(tile_u8.shape)[None], (n // tile_u8.shape[0], *tile_u8.shape)).reshape(-1).copy()
      w_u8, x, (lut, _) = Tensor(w_np, device=dev).realize(), Tensor(np.eye(m, k, k=256, dtype=np.float16), device=dev).realize(), _iq_lut(qtype, dev)
      out = _krun("rdna4_gemm_f16", (m, n), dev, w_u8, x, lut,
                  qtype=qtype, m_blks=m // 64, m_tiles=4, out_features=n, in_features=k, raw_bytes=w_len * w_dt.itemsize)[:, :4].numpy()
      w4 = ggml_data_to_tensor(Tensor(raw_tile[:4].reshape(-1), device=dev), 4 * k, qtype).reshape(4, k).numpy()
      np.testing.assert_allclose(out, w4[:, 256:384].T, rtol=8e-2, atol=1e-2)
    # 2. Unused PARAM slots (sel/grid) must not emit extra E/copy kernels or require ALLOW_DEVICE_USAGE
    w_u8, _, _ = _make_q8_0(rng, 64, 256)
    qlin = Linear(256, 64, bias=False)
    qlin.weight = ggml_data_to_tensor(w_u8, 64 * 256, 8).reshape(64, 256)
    qx = Tensor(rng.standard_normal((1, 256)).astype(np.float32)).realize()
    self.assertEqual(_knames(qlin(qx)), ["linear_quant"])
    # 3. Unrealized f32->f16 dense weight must fuse in pure tinygrad (no cast kernel), while realized f16 and GGUF bitcast f16 use linear_f16_gemv
    w_f32 = Tensor(rng.standard_normal((48, 256)).astype(np.float32)).realize()
    dlin_lazy, dlin_f16, dlin_gguf = Linear(256, 48, bias=False), Linear(256, 48, bias=False), Linear(256, 48, bias=False)
    dlin_lazy.weight = w_f32.cast(dtypes.half)
    dlin_f16.weight = w_f32.cast(dtypes.half).contiguous().realize()
    w_raw_u8 = Tensor(w_f32.numpy().astype(np.float16).view(np.uint8).reshape(-1)).realize()
    dlin_gguf.weight = w_raw_u8.bitcast(dtypes.half).reshape(48, 256)
    self.assertEqual(len(_knames(dlin_lazy(qx).sigmoid())), 1)
    self.assertEqual(_knames(dlin_f16(qx)), ["linear_f16_gemv"])
    self.assertEqual(_knames(dlin_gguf(qx)), ["linear_f16_gemv"])
    np.testing.assert_allclose(dlin_gguf(qx).numpy(), (qx @ dlin_f16.weight.float().T).numpy(), rtol=3e-3, atol=3e-3)
    # 4. GQA group=6 (e.g. qwen3.8:27b H=24, H_KV=4) must dispatch amd_flash_decode
    cfg = TransformerConfig(num_blocks=1, dim=256, hidden_dim=256, n_heads=24, n_kv_heads=4, norm_eps=1e-6, vocab_size=64,
                            head_dim=64, rope_theta=10000.0, rope_dim=64, v_head_dim=64, max_context=128)
    blk = TransformerBlock(cfg)
    for p in (blk.attn_q.weight, blk.attn_k.weight, blk.attn_v.weight, blk.attn_output.weight): p.realize()
    bx = Tensor(rng.standard_normal((1, 1, 256)).astype(np.float32)).realize()
    blk._init_state(bx)
    attn_names = _knames(blk._attention(bx, (UOp.variable("start_pos", 0, 127).bind(5),)))
    self.assertIn("flash_decode_partial", attn_names)
    self.assertIn("flash_decode_combine", attn_names)

  def test_where_dispatch_branches(self):
    """Exercise every Ops.WHERE branch in amd_gemv_f16_f32 and amd_gemv_quant_f32 and verify exactness."""
    if (tgt := _amd_target()) is None: self.skipTest("AMD HIP renderer required")
    dev, rng = Tensor.empty(1).device, np.random.default_rng(77)
    assert isinstance(dev, str)
    g0, sel0 = Tensor([0], dtype=dtypes.uint32).realize(), Tensor([0], dtype=dtypes.int32).realize()
    # 1. amd_gemv_f16_f32: base (tok=1 inf%256!=0), fdot2_1t (wk=1, wk>1 wr=1, non-div4 outf, wr=4), lds_mt (tok=2 inf%1024==0 outf>=1024)
    for tok, outf, inf in [(1, 64, 128), (1, 64, 256), (1, 63, 512), (1, 1024, 512), (1, 2048, 1024), (2, 1024, 1024)]:
      w = Tensor(rng.standard_normal((outf, inf)).astype(np.float16)).realize()
      x = Tensor(rng.standard_normal((tok, inf)).astype(np.float32)).realize()
      vc = 4 if inf % 128 == 0 else 1
      got = _krun("amd_gemv_f16_f32", (tok, outf), dev, w, x, tokens=tok, out_features=outf, in_blks=inf // (32 * vc), val_chunk=vc).numpy()
      ref_d = x.numpy() @ w.numpy().astype(np.float32).T
      np.testing.assert_allclose(got, ref_d, rtol=5e-3, atol=5e-3 * max(1.0, float(np.abs(ref_d).max())))
    # 2. amd_gemv_quant_f32: sdot4 (tok=4), iq4_lds (tok=1, inf=2048 & 6144 & non-div4 outf), iq4 direct fallback (tok=2 & inf=256), moe_fp8 (arch=12)
    if tgt[0] >= 11:
      w8, _, w8_ref = _make_q8_0(rng, 64, 1024)
      x4 = Tensor(rng.standard_normal((4, 1024)).astype(np.float32)).realize()
      ref4 = (x4 @ w8_ref.T).numpy()
      got4 = _krun("amd_gemv_quant_f32", (4, 64), dev, w8, x4, sel0, g0, qtype=8, tokens=4, x_rows=4, out_features=64,
                   in_features=1024, raw_bytes=int(w8.numel()), group_rows=0, use_sel=0, x_share=1, ndev=1, arch=tgt[0], num_experts=0).numpy()
      np.testing.assert_allclose(got4, ref4, rtol=4e-2, atol=3e-2 * np.abs(ref4).max())
      for tok, outf, inf in [(1, 64, 2048), (1, 62, 2048), (1, 64, 6144), (1, 64, 8192), (2, 64, 2048), (1, 64, 256)]:
        raw_np = rng.integers(0, 256, size=(outf * inf // 256, 136), dtype=np.uint8)
        raw_np[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
        raw_t = Tensor(raw_np.reshape(-1)).realize()
        w_ref = ggml_data_to_tensor(raw_t, outf * inf, 23).reshape(outf, inf).realize()
        xq = Tensor(rng.standard_normal((tok, inf)).astype(np.float32)).realize()
        ref = (xq @ w_ref.T).numpy()
        got = _krun("amd_gemv_quant_f32", (tok, outf), dev, raw_t, xq, sel0, g0, qtype=23, tokens=tok, x_rows=tok, out_features=outf,
                    in_features=inf, raw_bytes=int(raw_t.numel()), group_rows=0, use_sel=0, x_share=1, ndev=1, arch=tgt[0], num_experts=0).numpy()
        np.testing.assert_allclose(got, ref, rtol=3e-2, atol=2e-2 * np.abs(ref).max())
    if tgt[0] == 12:
      mw_u8, _, mw_ref = _make_q8_0(rng, 4 * 64, 256)
      mw_np = mw_ref.numpy().reshape(4, 64, 256)
      sel16 = Tensor(rng.integers(0, 4, size=(16,), dtype=np.int32)).realize()
      xm16 = Tensor(rng.standard_normal((16, 256)).astype(np.float32)).realize()
      got16 = _krun("amd_gemv_quant_f32", (16, 64), dev, mw_u8, xm16, sel16, g0, qtype=8, tokens=16, x_rows=16, out_features=64,
                    in_features=256, raw_bytes=int(mw_u8.numel()), group_rows=0, use_sel=1, x_share=1, ndev=1, arch=12, num_experts=4).numpy()
      ref16 = np.stack([mw_np[int(eid)] @ xm16.numpy()[i] for i, eid in enumerate(sel16.numpy())])
      np.testing.assert_allclose(got16, ref16, rtol=6e-2, atol=5e-2 * np.abs(ref16).max())
      # 3. rdna4_flash_attn qm=2 branch (M>=512, D<=128) & rdna4_gemm_f16 tok>=1024 FP8 branch (real GPU; too large for MockGPU emu)
      if not DEV.interface.startswith("MOCK"):
        kv_512 = Tensor(rng.standard_normal((2, 1, 2, 512, 64)).astype(np.float16)).realize()
        q_512 = Tensor(rng.standard_normal((1, 4, 512, 64)).astype(np.float16)).realize()
        got_qm2 = _krun("rdna4_flash_attn", (1, 4, 512, 64), dev, q_512, kv_512, UOp.const(512), UOp.const(512),
                        B=1, H=4, H_KV=2, m_blks=16, d_blks=1, n_blks=16, valid_kv_offset=0, q_start_from_end=1).numpy()
        mask_512 = Tensor.full((512, 512), float("-inf"), dtype=dtypes.float32).triu(1)
        ref_qm2 = q_512.float().scaled_dot_product_attention(kv_512[0].float(), kv_512[1].float(), attn_mask=mask_512, enable_gqa=True).numpy()
        np.testing.assert_allclose(got_qm2, ref_qm2, rtol=5e-3, atol=5e-3)
        w8g, _, w8g_ref = _make_q8_0(rng, 2048, 512)
        x1024 = Tensor(rng.standard_normal((1024, 512)).astype(np.float16)).realize()
        got1024 = _krun("rdna4_gemm_f16", (1024, 2048), dev, w8g, x1024, g0, qtype=8,
                        m_blks=16, m_tiles=4, out_features=2048, in_features=512, raw_bytes=int(w8g.numel())).numpy()
        ref1024 = (x1024.float() @ w8g_ref.T).numpy()
        np.testing.assert_allclose(got1024, ref1024, rtol=5e-2, atol=4e-2 * np.abs(ref1024).max())

if __name__ == "__main__":
  unittest.main()
