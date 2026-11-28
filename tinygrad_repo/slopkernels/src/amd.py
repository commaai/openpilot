import functools, math

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

WARP_SIZE, WMMA_M, WMMA_N, WMMA_K, DECODE_C_CAP = 32, 16, 16, 16, 64
WMMA_ARG, LOG2E = ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)
_H2T = "decltype(__builtin_amdgcn_cvt_pkrtz(0.0f, 0.0f))"

def _unbind(v:UOp) -> UOp: return v.unbind_all()[0]
def _nat(s:str, dt:DType, *args:UOp, inline:bool=True) -> UOp:
  return UOp.native(s, shape=args[0].shape, dtype=dt, fmt=True).call(*args, inline=inline)
def _lane_xor(v:UOp, m:int, inline:bool=False, dpp:bool|UOp=False) -> UOp:
  if isinstance(dpp, UOp): return dpp.where(_lane_xor(v, m, inline=inline, dpp=True), _lane_xor(v, m, inline=inline, dpp=False))
  vi = v.bitcast(dtypes.int32)
  if dpp and m in (1, 2, 4, 8):
    return _nat(f"__builtin_amdgcn_update_dpp(0, {{0}}, {0x160 | m}, 15, 15, true)", dtypes.int32, vi, inline=inline).bitcast(v.dtype)
  if dpp and m == 16:
    return _nat("__builtin_amdgcn_permlanex16({0}, {0}, 0x76543210u, 0xfedcba98u, false, false)",
                dtypes.int32, vi, inline=inline).bitcast(v.dtype)
  return _nat(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | m<<10})", dtypes.int32, vi, inline=inline).bitcast(v.dtype)
def _ds_bpermute(a:UOp, v:UOp, inline:bool=True) -> UOp:
  return _nat("__builtin_amdgcn_ds_bpermute({0}, {1})", dtypes.int32, a.cast(dtypes.int32), v.bitcast(dtypes.int32), inline=inline).bitcast(v.dtype)
def _fast_exp2(v:UOp) -> UOp: return _nat("__builtin_amdgcn_exp2f({})", dtypes.float32, v)
def _fast_log2(v:UOp) -> UOp: return _nat("__builtin_amdgcn_logf({})", dtypes.float32, v)
def _fast_fmax(a:UOp, b:UOp) -> UOp: return _nat("__builtin_fmaxf({0}, {1})", dtypes.float32, a, b)
def _cvt_pk_f16(a:UOp, b:UOp, inline:bool=True) -> UOp:
  return _nat("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {1}))", dtypes.uint32, a, b, inline=inline)
def _amd_fdot2(a:UOp, b:UOp, c:UOp) -> UOp:
  return _nat(f"__builtin_amdgcn_fdot2(__builtin_bit_cast({_H2T}, {{0}}), __builtin_bit_cast({_H2T}, {{1}}), {{2}}, false)", dtypes.float32, a, b, c)
def _amd_byte_perm(a:UOp, b:UOp, sel:UOp) -> UOp:
  return _nat("__builtin_amdgcn_perm({}, {}, {})", dtypes.uint32, *(x.cast(dtypes.uint32) for x in (a, b, sel)))
def _sb0(v:UOp) -> UOp: return _nat("(__builtin_amdgcn_sched_barrier(0), ({0}))", v.dtype, v, inline=False)
def _wmma_u4(a_ptr:UOp, b_ptr:UOp, c_ptr:UOp, tok:UOp, *deps:UOp) -> UOp:
  call = UOp.native(
    "({{ typedef __fp16 _hv8 __attribute__((__vector_size__(8 * sizeof(__fp16)))); "
    "typedef float _fv8 __attribute__((__vector_size__(8 * sizeof(float)))); "
    "*(_fv8*)({2}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(*(_hv8*)({0}), *(_hv8*)({1}), *(_fv8*)({2})); 0; }})",
    shape=(), dtype=dtypes.int32, fmt=True).call(a_ptr, b_ptr, c_ptr, inline=False)
  return tok.after(*deps)[0].store(call)
def _tree_fmax(vs:list[UOp]) -> UOp:
  while len(vs) > 1: vs = [_fast_fmax(vs[i], vs[i + 1]) for i in range(0, len(vs) - 1, 2)] + ([vs[-1]] if len(vs) % 2 else [])
  return vs[0]

def _u4_to_h8(u4:list[UOp], tmp:UOp, *deps:UOp) -> UOp:
  return tmp.after(tmp.after(*deps).store(UOp.stack(*u4).bitcast(dtypes.half)))

def warp_reduce(val:UOp, op:Ops=Ops.ADD, full_wave:bool=False, masks:tuple[int, ...]|None=None, dpp:bool|UOp=False) -> UOp:
  if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
  if val.op is Ops.REDUCE or any(u.op is Ops.REDUCE for u in val.src):
    tmp = UOp.alloc((1,), val.dtype, addrspace=AddrSpace.REG)
    val = tmp.after(tmp[0].store(val))[0].load()
  assert val.dtype in (dtypes.float32, dtypes.int32, dtypes.uint32)
  for mask in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)) if masks is None else masks:
    peer = _lane_xor(val, mask, dpp=dpp)
    val = _fast_fmax(val, peer) if op is Ops.MAX and val.dtype == dtypes.float32 else val.alu(op, peer)
  return val

def _reg(shape:tuple[int|UOp, ...], value:float, *deps:UOp) -> UOp:
  ret = UOp.alloc(shape, dtypes.float, addrspace=AddrSpace.REG)
  return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(value)))

def _amd_load(ptr:UOp, lanes:int|None=None, stream:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  if lanes is None: return ptr.load(arg="nontemporal")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord * math.prod(buf.shape[i + 1:]) for i, coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(arg="nontemporal" if stream else None)

def _live_decode_chunks(vlen:UOp, C:UOp|int) -> UOp:
  return ((vlen + 31) // 32).minimum(C.minimum(DECODE_C_CAP) if isinstance(C, UOp) else min(C, DECODE_C_CAP)).maximum(1)

# ******** dense fp16/bf16/f32 gemv (nontemporal + tiled LDS fdot2 for multi-token f16) ********

@functools.cache
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *, in_blks:UOp, out_features:UOp, tokens:UOp, val_chunk:UOp,
                         lds_mt:bool=False, fdot2_1t:bool=False) -> UOp:
  inf = in_blks * (32 * val_chunk)
  if fdot2_1t:
    n_k = inf // 256
    wk = (out_features >= 2048).where(4, (out_features >= 1024).where(2, 4)).minimum(n_k)
    wr = ((out_features >= 1024) & (out_features % 4).eq(0)).where(4, (out_features % 2).eq(0).where(2, 1))
    steps = (n_k + wk - 1) // wk
    rb = UOp.range(out_features // wr, 0, AxisType.GLOBAL)
    lane, wave_k, w_row = UOp.range(32, -1, AxisType.WARP), UOp.range(wk, 1, AxisType.LOCAL), UOp.range(wr, 2, AxisType.LOCAL)
    row, acc, s = rb * wr + w_row, _reg((1,), 0.0), UOp.range(steps, 5)
    kblk = wave_k + s * wk
    ok, base_k = kblk < n_k, kblk.minimum(n_k - 1) * 256 + lane * 8
    wv, v = _amd_load(w[row, base_k], 8).bitcast(dtypes.uint32), acc.after(s)[0].load()
    xv0, xv1 = _amd_load(x[0, base_k], 4), _amd_load(x[0, base_k + 4], 4)
    for j, (a0, a1) in enumerate(((xv0[0], xv0[1]), (xv0[2], xv0[3]), (xv1[0], xv1[1]), (xv1[2], xv1[3]))):
      v = _amd_fdot2(wv[j], _cvt_pk_f16(a0, a1), v)
    upd = acc[0].store(ok.where(v, acc.after(s)[0].load())).end(s)
    wsum = warp_reduce(acc.after(upd)[0].load(), full_wave=True)
    lds = UOp.alloc((wr, wk), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds_r, kr = lds.after(lds[w_row, wave_k.valid(lane.eq(0))].store(wsum).barrier()), UOp.range(wk, 12)
    return out[0, row.valid(wave_k.eq(0) & lane.eq(0))].store(lds_r[w_row, kr].load().reduce(kr, arg=(Ops.ADD, 0)).cast(out.dtype)).end(
      lane, wave_k, w_row, rb).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))
  if lds_mt:
    w_rows_l, ku_l, blk_k = (out_features % 16).eq(0).where(16, 8), 4, 1024
    rb, wr, lane = UOp.range(out_features // w_rows_l, 0, AxisType.GLOBAL), UOp.range(w_rows_l, 2, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
    out_row, tid, threads, acc = rb * w_rows_l + wr, wr * 32 + lane, w_rows_l * 32, _reg((tokens,), 0.0)
    x_lds, i_r = UOp.alloc((tokens, ku_l * 4, 32), dtypes.uint32, addrspace=AddrSpace.LOCAL), UOp.range(inf // blk_k, 4)
    k0, tot_p = i_r * blk_k, tokens * (blk_k // 2)
    xi = UOp.range((tot_p + threads - 1) // threads, 10)
    pid = tid + xi * threads
    ok, pid_s = pid < tot_p, pid.minimum(tot_p - 1)
    xt, xk2 = pid_s // (blk_k // 2), pid_s % (blk_k // 2)
    x_r = x_lds.after(x_lds.after(i_r)[xt.valid(ok), (xk2 // 128) * 4 + (xk2 % 4), (xk2 % 128) // 4].store(
      _cvt_pk_f16(x[xt, k0 + xk2 * 2].load().float(), x[xt, k0 + xk2 * 2 + 1].load().float())).end(xi).barrier())
    w_vs = [_amd_load(w[out_row, k0 + ku_i * 256 + lane * 8], 8, stream=True).bitcast(dtypes.uint32) for ku_i in range(ku_l)]
    t_r = UOp.range(tokens, 8)
    v = acc.after(i_r)[t_r].load()
    for ku_i in range(ku_l):
      for j in range(4): v = _amd_fdot2(w_vs[ku_i][j], x_r[t_r, ku_i * 4 + j, lane].load(), v)
    upd, t_o = acc[t_r].store(v).end(t_r).barrier().end(i_r), UOp.range(tokens, 9)
    return out[t_o, out_row.valid(lane.eq(0))].store(warp_reduce(acc.after(upd)[t_o].load(), full_wave=True).cast(out.dtype)).end(
      t_o, lane, wr, rb).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))
  vc = (inf % 256).eq(0).where(8, val_chunk) if w.dtype.itemsize == 2 else val_chunk
  ku = ((inf // (32 * vc)) % 2).eq(0).where(2, 1)
  n_blks, w_rows = inf // (32 * vc * ku), ((out_features >= 1024) & (out_features % 4).eq(0)).where(4, 1)
  rb, lane, wr = UOp.range(out_features // w_rows, 0, AxisType.GLOBAL), UOp.range(32, 2, AxisType.LOCAL), UOp.range(w_rows, 3, AxisType.LOCAL)
  out_row, acc = rb * w_rows + wr, _reg((tokens,), 0.0)
  i_r, ku_r, u, t_r = UOp.range(n_blks, 4), UOp.range(ku, 7, AxisType.UPCAST), UOp.range(vc, 6, AxisType.UPCAST), UOp.range(tokens, 8)
  col = ((i_r * ku + ku_r) * 32 + lane) * vc + u
  dot = (w[out_row, col].load(arg="nontemporal").float() * x[t_r, col].load().float()).reduce(u, ku_r, arg=(Ops.ADD, 0))
  upd, t_o = acc[t_r].store(acc.after(i_r)[t_r].load() + dot).end(t_r, i_r), UOp.range(tokens, 9)
  return out[t_o, out_row.valid(lane.eq(0))].store(warp_reduce(acc.after(upd)[t_o].load(), full_wave=True).cast(out.dtype)).end(
    t_o, rb, wr, lane).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

# ******** flash attention decode (wave_g head-split + 16-lane b128 on D=128 + b128 combine) ********

@functools.cache
def _amd_flash_attention_decode_partial(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, *, valid_kv_len:UOp, d_blks:UOp, group:UOp, waves:UOp,
                                        g_split:int|UOp=1, lanes_per_tok:UOp) -> UOp:
  vlen, (_, B, H_KV, max_n, _) = _unbind(valid_kv_len), cache_kv.shape
  D, PARTIALS = d_blks * 64, out.shape[2]
  Gw, waves_n, toks_per_w = group // g_split, waves // g_split, WARP_SIZE // lanes_per_tok
  step_stride, u32_pl = waves_n * toks_per_w, (d_blks * 32) // lanes_per_tok
  dpl, scale_l2, live_chunks = u32_pl * 2, D.cast(dtypes.float32).rsqrt() * LOG2E, _live_decode_chunks(vlen, PARTIALS)
  total_steps = (vlen + step_stride - 1) // step_stride
  block_bhkv, block_chunk = UOp.range(B * H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(waves, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  zerof, neg_inf = UOp.const(0.0, dtypes.float32), UOp.const(-1e30, dtypes.float32)
  z32, zw = UOp.const(0, dtypes.uint32), UOp.const(0, dtypes.weakint)
  wg_i, wn_i = wave % g_split, wave // g_split
  sub_t, sub_l = lane // lanes_per_tok, lane % lanes_per_tok
  w_tok_id, g_base = wn_i * toks_per_w + sub_t, wg_i * Gw

  qh_reg, h_q, w_q = UOp.alloc((Gw, u32_pl), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(Gw, 15), UOp.range(u32_pl, 16)
  qh_reg = qh_reg.after(qh_reg[h_q, w_q].store(_cvt_pk_f16(
    q[b, kv_head * group + g_base + h_q, 0, sub_l * dpl + 2 * w_q].load().float() * scale_l2,
    q[b, kv_head * group + g_base + h_q, 0, sub_l * dpl + 2 * w_q + 1].load().float() * scale_l2)).end(w_q, h_q))
  acc_reg, max_reg, sum_reg = _reg((Gw, dpl), 0.0), _reg((Gw,), -1e30), _reg((Gw,), 0.0)
  base_st, rem_st = total_steps // live_chunks, total_steps % live_chunks
  step_start = block_chunk * base_st + block_chunk.minimum(rem_st)
  step_end = step_start + base_st + (block_chunk < rem_st).where(UOp.const(1, dtypes.weakint), zw)
  full_end = step_end.minimum(vlen // step_stride)
  n_full, n_tail = (full_end - step_start).maximum(0), (step_end - full_end.maximum(step_start)).maximum(0)
  kv_u32, row_u32, bh_st = cache_kv.flatten().bitcast(dtypes.uint32), d_blks * 32, max_n * (d_blks * 32)
  v_off_u32, bh_base = B * H_KV * bh_st, (b * H_KV + kv_head) * bh_st + sub_l * u32_pl

  def load_into(kb:UOp, vb:UOp, st_idx:UOp, dep:tuple[UOp, ...], masked:bool, rid:int) -> tuple[UOp, UOp]:
    ks_i = st_idx * step_stride + w_tok_id
    vl, w_u = (ks_i < vlen) if masked else None, UOp.range(u32_pl, rid, AxisType.UPCAST)
    ks_s = vl.where(ks_i, zw) if vl is not None else ks_i
    vv = kv_u32[v_off_u32 + bh_base + ks_s * row_u32 + w_u].load()
    return (kb.after(kb.after(*dep)[w_u].store(kv_u32[bh_base + ks_s * row_u32 + w_u].load()).end(w_u)),
            vb.after(vb.after(*dep)[w_u].store(vl.where(vv, z32) if vl is not None else vv).end(w_u)))

  def comp_step(kb:UOp, vb:UOp, st_idx:UOp|None, a_in:UOp, m_in:UOp, s_in:UOp, rid:int) -> UOp:
    h_u, w_d = UOp.range(Gw, rid), UOp.range(u32_pl, rid + 1)
    d_r = _reg((1,), 0.0, h_u)
    d_v = d_r.after(d_r[0].store(_amd_fdot2(qh_reg[h_u, w_d].load(), kb[w_d].load(), d_r.after(w_d)[0].load())).end(w_d))[0].load()
    for mask in (4, 2, 1): d_v = d_v + _lane_xor(d_v, mask, inline=True)
    d_v = (lanes_per_tok > 8).where(d_v + _lane_xor(d_v, 8, inline=True), d_v)
    d_v = (lanes_per_tok > 16).where(d_v + _lane_xor(d_v, 16, inline=True), d_v)
    vl = (st_idx * step_stride + w_tok_id < vlen) if st_idx is not None else None
    sc, old_m = vl.where(d_v, neg_inf) if vl is not None else d_v, m_in[h_u].load()
    new_m = _fast_fmax(old_m, sc)
    alpha, beta = _fast_exp2(old_m - new_m), vl.where(_fast_exp2(sc - new_m), zerof) if vl is not None else _fast_exp2(sc - new_m)
    w_v = UOp.range(u32_pl, rid + 2)
    vh2 = vb[w_v].load().reshape((1,)).bitcast(dtypes.half)
    a_st = UOp.group(*(acc_reg[h_u, 2 * w_v + k].store(alpha * a_in[h_u, 2 * w_v + k].load() + beta * vh2[k].float()) for k in (0, 1))).end(w_v)
    return UOp.group(a_st, max_reg[h_u].store(new_m), sum_reg[h_u].store(alpha * s_in[h_u].load() + beta)).end(h_u)

  kb0, vb0, kb1, vb1 = (UOp.alloc((u32_pl,), dtypes.uint32, addrspace=AddrSpace.REG) for _ in range(4))
  has_full = n_full > 0
  k0_r, v0_r = load_into(kb0, vb0, has_full.where(step_start, zw), (qh_reg,), False, 35)
  it = UOp.range(n_full, 40)
  k1_r, v1_r = load_into(kb1, vb1, step_start + (it + 1).minimum((n_full - 1).maximum(0)), (it, k0_r, v0_r), False, 41)
  comp = comp_step(k0_r.after(it, k1_r, v1_r), v0_r.after(it, k1_r, v1_r), None, acc_reg.after(it), max_reg.after(it), sum_reg.after(it), 42)
  w_cp = UOp.range(u32_pl, 45, AxisType.UPCAST)
  last_done = UOp.group(comp, UOp.group(kb0[w_cp].store(k1_r.after(comp)[w_cp].load()),
                                        vb0[w_cp].store(v1_r.after(comp)[w_cp].load())).end(w_cp)).end(it)
  it_t, st_t = UOp.range(n_tail, 50), full_end.maximum(step_start)
  kbt, vbt = load_into(kb1, vb1, st_t + it_t, (it_t, last_done), True, 51)
  dt = comp_step(kbt, vbt, st_t + it_t, acc_reg.after(last_done, it_t), max_reg.after(last_done, it_t), sum_reg.after(last_done, it_t), 52).end(it_t)
  acc_f, max_f, sum_f = acc_reg.after(dt), max_reg.after(dt), sum_reg.after(dt)

  acc_lds = UOp.alloc((step_stride, group, D // 2), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  ml_lds = UOp.alloc((step_stride, group, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  h_w, w_w = UOp.range(Gw, 60), UOp.range(u32_pl, 61)
  inv_s = sum_f[h_w].load().maximum(1.0).reciprocal()
  s_acc = acc_lds[w_tok_id, g_base + h_w, sub_l * u32_pl + w_w].store(
    _cvt_pk_f16(acc_f[h_w, 2 * w_w].load() * inv_s, acc_f[h_w, 2 * w_w + 1].load() * inv_s)).end(w_w)
  b_ep = UOp.group(s_acc, ml_lds[w_tok_id.valid(sub_l.eq(0)), g_base + h_w, 0].store(max_f[h_w].load()),
                   ml_lds[w_tok_id.valid(sub_l.eq(0)), g_base + h_w, 1].store(sum_f[h_w].load())).end(h_w).barrier()
  acc_lds_r, ml_lds_r, tid, tot_gd2 = acc_lds.after(b_ep), ml_lds.after(b_ep), wave * WARP_SIZE + lane, group * (D // 2)
  fi = UOp.range((tot_gd2 + waves * WARP_SIZE - 1) // (waves * WARP_SIZE), 62)
  flat = tid + fi * (waves * WARP_SIZE)
  h_f, d2_f, mr, wi_m = (flat // (D // 2)).minimum(group - 1), flat % (D // 2), _reg((1,), -1e30, fi), UOp.range(step_stride, 63)
  md = mr[0].store(_fast_fmax(mr.after(wi_m)[0].load(), ml_lds_r[wi_m, h_f, 0].load())).end(wi_m)
  M_val = mr.after(md)[0].load()
  v0r, v1r, lr, wi_a = _reg((1,), 0.0, md), _reg((1,), 0.0, md), _reg((1,), 0.0, md), UOp.range(step_stride, 64)
  w_wi = _fast_exp2(ml_lds_r[wi_a, h_f, 0].load() - M_val) * ml_lds_r[wi_a, h_f, 1].load()
  vh2 = acc_lds_r[wi_a, h_f, d2_f].load().reshape((1,)).bitcast(dtypes.half)
  ad = UOp.group(v0r[0].store(v0r.after(wi_a)[0].load() + w_wi * vh2[0].float()),
                 v1r[0].store(v1r.after(wi_a)[0].load() + w_wi * vh2[1].float()), lr[0].store(lr.after(wi_a)[0].load() + w_wi)).end(wi_a)
  vf, q_h, lsum = flat < tot_gd2, kv_head * group + h_f, lr.after(ad)[0].load()
  inv_l = lsum.maximum(1.0).reciprocal()
  return UOp.group(
    out[b, q_h.valid(vf), block_chunk, d2_f * 2].store(v0r.after(ad)[0].load() * inv_l),
    out[b, q_h.valid(vf), block_chunk, d2_f * 2 + 1].store(v1r.after(ad)[0].load() * inv_l),
    stats[b, q_h.valid(vf & d2_f.eq(0)), block_chunk, 0].store(M_val),
    stats[b, q_h.valid(vf & d2_f.eq(0)), block_chunk, 1].store(lsum),
  ).end(fi, lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, *, valid_kv_len:UOp) -> UOp:
  vlen, (B, H, C, D) = _unbind(valid_kv_len), partial.shape
  assert isinstance(D, UOp)
  epl = ((D % 128).eq(0) & (D >= 128)).where(4, 2)
  block_bh, block_dt = UOp.range(B * H, 0, AxisType.GLOBAL), UOp.range(D // (WARP_SIZE * epl), 1, AxisType.GLOBAL)
  lane, i_u = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL), UOp.range(epl, 102, AxisType.UPCAST)
  b, h, dim, chunk = block_bh // H, block_bh % H, block_dt * (WARP_SIZE * epl) + lane * epl + i_u, UOp.range(_live_decode_chunks(vlen, C), 100)
  acc, m_r, l_r = _reg((epl,), 0.0), _reg((1,), -1e30), _reg((1,), 0.0)
  mc, lc, m_old = stats[b, h, chunk, 0].load(), stats[b, h, chunk, 1].load(), m_r.after(chunk)[0].load()
  m_new = _fast_fmax(m_old, mc)
  a_sc, b_sc = _fast_exp2(m_old - m_new), _fast_exp2(mc - m_new) * lc
  upd = UOp.group(acc[i_u].store(a_sc * acc.after(chunk)[i_u].load() + b_sc * partial[b, h, chunk, dim].load()).end(i_u),
                  m_r[0].store(m_new), l_r[0].store(a_sc * l_r.after(chunk)[0].load() + b_sc)).end(chunk)
  return o[b, h, 0, dim].store((acc.after(upd)[i_u].load() * l_r.after(upd)[0].load().reciprocal()).cast(o.dtype)).end(
    i_u, lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

# ******** flash attention prefill: unified M=32 Split-N + general pipelined WMMA ********

def _r3_unpack_p(src_u4:UOp, tmp:UOp, is_g0:UOp, rid:int, *deps:UOp) -> UOp:
  j = UOp.range(4, rid)
  my_u = src_u4[j].load()
  peer_u = _lane_xor(my_u, 16, inline=True)
  t2 = tmp.after(*deps).reshape(2, 4, 2)
  return tmp.after(UOp.group(
    t2[0, j].store(is_g0.where(my_u, peer_u).reshape((1,)).bitcast(dtypes.half)),
    t2[1, j].store(is_g0.where(peer_u, my_u).reshape((1,)).bitcast(dtypes.half))).end(j))

def _fa_load_q(q:UOp, block_bh:UOp, m_base:UOp, td_base:int|UOp, td_w:UOp, d_blks:UOp, M:int|UOp,
               lane_c:UOp, lane_g:UOp, scale_l2:UOp, rdna4:bool) -> tuple[UOp, UOp|None]:
  kq_i = UOp.range(td_w, 10)
  if rdna4:
    q_reg, p = UOp.alloc((td_w, 4), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(4, 11)
    q_c = (td_base + kq_i) * 16 + (p // 2) * 8 + lane_g * 4 + (p % 2) * 2
    q_reg = q_reg.after(q_reg[kq_i, p].store(_cvt_pk_f16(
      q[block_bh, m_base + lane_c, q_c].load().float() * scale_l2,
      q[block_bh, m_base + lane_c, q_c + 1].load().float() * scale_l2)).end(p, kq_i))
    return q_reg, UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG)
  q_u32, j = q.flatten().bitcast(dtypes.uint32), UOp.range(8, 11)
  q_reg, qv_buf = UOp.alloc((td_w, 16), dtypes.half, addrspace=AddrSpace.REG), UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
  q_off = (block_bh * M + m_base + lane_c) * (d_blks * 32) + (td_base + kq_i) * 8
  qv_r = qv_buf.after(qv_buf.after(kq_i).store(_amd_load(q_u32[q_off], 4).cat(_amd_load(q_u32[q_off + 4], 4))))
  h = qv_r[j].load().reshape((1,)).bitcast(dtypes.half)
  q_st = q_reg.reshape(td_w, 8, 2)[kq_i, j].store(_cvt_pk_f16(h[0].float() * scale_l2, h[1].float() * scale_l2).reshape((1,)).bitcast(dtypes.half))
  return q_reg.after(q_st.end(j, kq_i)), None

@functools.cache
def _amd_flash_attention_smallm(o:UOp, q:UOp, cache:UOp, *, valid_kv_len:UOp, q_start:UOp, d_blks:UOp, rdna4:bool=True) -> UOp:
  valid_kv_len, q_start, (BH, M, D), (_, B, H_KV, physical_n, _) = _unbind(valid_kv_len), _unbind(q_start), q.shape, cache.shape
  assert isinstance(D, UOp)
  waves_n, block_m_sz, bn_w = 8, 16, 16
  block_n_tot, threads, m_blks_eff = waves_n * bn_w, waves_n * WARP_SIZE, M // block_m_sz
  TD, K_CH, V_CH_tot, gqa_group = d_blks * 4, d_blks * 8, block_n_tot // 8, BH // (B * H_KV)
  SCALE_L2, zw, z32 = D.cast(dtypes.float32).rsqrt() * LOG2E, UOp.const(0, dtypes.weakint), UOp.const(0, dtypes.uint32)
  c_lo, c_hi = UOp.const(0x05040100, dtypes.uint32), UOp.const(0x07060302, dtypes.uint32)

  block_bh, block_m_raw = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range(m_blks_eff, 1, AxisType.GLOBAL)
  block_m, wn, lane = m_blks_eff - 1 - block_m_raw, UOp.range(waves_n, 2, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  tid, lane_g, lane_c, is_g0 = wn * WARP_SIZE + lane, lane // 16, lane % 16, (lane // 16).eq(0)
  kv_head, k_u32 = block_bh // gqa_group, cache.flatten().bitcast(dtypes.uint32)
  v_base_u32, m_base = B * H_KV * physical_n * (d_blks * 32), block_m * block_m_sz

  q_u, tmp_q = _fa_load_q(q, block_bh, m_base, zw, TD, d_blks, M, lane_c, lane_g, SCALE_L2, rdna4)
  k_lds_raw = UOp.alloc((block_n_tot * D,), dtypes.half, addrspace=AddrSpace.LOCAL)
  v_lds_raw = UOp.alloc((block_n_tot * D,), dtypes.half, addrspace=AddrSpace.LOCAL) if rdna4 else k_lds_raw
  P_reg_buf = UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG) if rdna4 else None
  P_half_buf = UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG) if not rdna4 else None
  P_tmp_buf = UOp.alloc((16,), dtypes.half, addrspace=AddrSpace.REG) if not rdna4 else None

  acc, m_i, l_i = _reg((TD, 8), 0.0), _reg((1,), -1e30), _reg((1,), 0.0)
  S_reg_buf = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
  n_end_r = ((q_start + (block_m + 1) * block_m_sz).minimum(valid_kv_len) + block_n_tot - 1) // block_n_tot
  n_full_r = ((q_start + block_m * block_m_sz + 1).minimum(valid_kv_len) // block_n_tot).minimum(n_end_r)
  def swz_k(kr:UOp, kk:UOp) -> UOp: return ((kr & 7) ^ (d_blks >= 2).where((kk >> 2) & 1, zw)) if rdna4 else (kr & 7)
  def swz_v(d_i:UOp, ch:UOp) -> UOp: return (ch ^ (d_i & 7) ^ ((d_i >> 3) & 7)) if rdna4 else (ch ^ ((d_i >> 1) & 1))
  k_items, v_items = block_n_tot * TD, (block_n_tot // 2) * (d_blks * 8)

  def store_k(rnd:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    k_lds_u32, su = k_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(block_n_tot, K_CH, 4), UOp.range(4, rid)
    bid = tid + su * threads
    ok, bid_s = bid < k_items, bid.minimum(k_items - 1)
    kr, kk_i = bid_s // TD, bid_s % TD
    kb = (kv_head * physical_n + (rnd * block_n_tot + kr).minimum(physical_n - 1)) * (d_blks * 32) + kk_i * 8
    v0, v1, kr_v = _amd_load(k_u32[kb], 4, stream=rdna4), _amd_load(k_u32[kb + 4], 4, stream=rdna4), kr.valid(ok)
    if rdna4:
      xk = swz_k(kr, kk_i)
      ks = [k_lds_u32[kr_v, (kk_i * 2) ^ xk].store(UOp.stack(v0[0], v0[1], v1[0], v1[1])),
            k_lds_u32[kr_v, (kk_i * 2 + 1) ^ xk].store(UOp.stack(v0[2], v0[3], v1[2], v1[3]))]
    else: ks = [k_lds_u32[kr_v, (kk_i * 2) ^ (kr & 7)].store(v0), k_lds_u32[kr_v, (kk_i * 2 + 1) ^ (kr & 7)].store(v1)]
    return UOp.group(*ks).end(su)

  def store_v(rnd:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    v_lds_u32 = v_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(D, V_CH_tot, 4) if rdna4 else \
                v_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(waves_n, D, 2, 4)
    n_ctiles, su = d_blks * 8, UOp.range(4, rid)
    tbid = tid + su * threads
    ok, tbid_s = tbid < v_items, tbid.minimum(v_items - 1)
    c0, pid = (tbid_s % n_ctiles) * 8, tbid_s // n_ctiles
    if rdna4: ga, gb, ch_a, slot_a = rnd * block_n_tot + pid * 2, rnd * block_n_tot + pid * 2 + 1, (pid * 2) // 8, ((pid * 2) % 8) // 2
    else:
      w_v, p_w = pid // 8, pid % 8
      blk4, par = p_w // 2, p_w % 2
      ga, gb, ch_a, slot_a = rnd * block_n_tot + w_v * 16 + blk4 * 4 + par, rnd * block_n_tot + w_v * 16 + blk4 * 4 + 2 + par, par, blk4
    va = _amd_load(k_u32[v_base_u32 + (kv_head * physical_n + ga.minimum(physical_n - 1)) * (d_blks * 32) + c0 // 2], 4, stream=rdna4)
    vb = _amd_load(k_u32[v_base_u32 + (kv_head * physical_n + gb.minimum(physical_n - 1)) * (d_blks * 32) + c0 // 2], 4, stream=rdna4)
    ra_b, rb_b = UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG)
    ra_r, rb_r = ra_b.after(ra_b.after(*dep, su).store(va)), rb_b.after(rb_b.after(*dep, su).store(vb))
    p = UOp.range(4, rid + 50)
    ra_p, rb_p = (ga < valid_kv_len).where(ra_r[p].load(), z32), (gb < valid_kv_len).where(rb_r[p].load(), z32)
    def _vidx(m:int) -> UOp:
      return (v_lds_u32[(c0 + p * 2 + m).valid(ok), swz_v(c0 + p * 2 + m, ch_a), slot_a] if rdna4 else
              v_lds_u32[w_v, (c0 + p * 2 + m).valid(ok), swz_v(c0 + p * 2 + m, ch_a), slot_a])
    return UOp.group(_vidx(0).store(_amd_byte_perm(rb_p, ra_p, c_lo)), _vidx(1).store(_amd_byte_perm(rb_p, ra_p, c_hi))).end(p, su)

  k0 = store_k(zw, (q_u,), 90).barrier() if rdna4 else q_u
  rnd = UOp.range(n_end_r, 100)
  k_dep = k0 if rdna4 else store_k(rnd, (k0, rnd), 101).barrier()
  K_lds = k_lds_raw.after(k_dep, rnd).reshape(block_n_tot, K_CH, 8)
  S_reg = S_reg_buf.after(S_reg_buf.after(k_dep, rnd).store(S_reg_buf.const_like(0.0)))
  k_s, k_row = UOp.range(TD, 130), wn * bn_w + lane_c
  if rdna4:
    assert tmp_q is not None
    qf = _u4_to_h8([q_u[k_s, p].load() for p in range(4)], tmp_q, k_s)
    qk_done = S_reg.store(UOp.wmma(K_lds[k_row, (k_s * 2 + lane_g) ^ swz_k(k_row, k_s)], qf, S_reg.after(k_s), *WMMA_ARG)).end(k_s)
  else:
    k_f = K_lds[k_row, (k_s * 2) ^ (k_row & 7)].cat(K_lds[k_row, (k_s * 2 + 1) ^ (k_row & 7)])
    qk_done = S_reg.store(UOp.wmma(k_f, q_u[k_s], S_reg.after(k_s), *WMMA_ARG)).end(k_s)
  v_st = store_v(rnd, (qk_done if rdna4 else qk_done.barrier(),), 102)
  S_cur = S_reg.after(qk_done)
  r_cm, e_m = UOp.range((rnd >= n_full_r).cast(dtypes.weakint), 104), UOp.range(8, 105, AxisType.UPCAST)
  k_idx = rnd * block_n_tot + wn * bn_w + (lane_g * 8 + e_m if rdna4 else e_m * 2 + lane_g)
  causal = (k_idx <= q_start + m_base + lane_c) & (k_idx < valid_kv_len)
  S_cur = S_reg.after(S_reg[e_m].store(causal.where(S_cur.after(r_cm)[e_m].load(), UOp.const(-math.inf, dtypes.float32))).end(e_m, r_cm))
  row_max = warp_reduce(_tree_fmax([S_cur[ei].load() for ei in range(8)]), Ops.MAX, masks=(16,), dpp=rdna4)
  m_prev = m_i.after(rnd, qk_done)[0].load()
  m_new = _fast_fmax(m_prev, row_max)
  m_safe = _fast_fmax(m_new, UOp.const(-1e30, dtypes.float32))
  alpha, acc_p, td_a, e_a = _fast_exp2(m_prev - m_safe), acc.after(rnd, qk_done), UOp.range(TD, 120), UOp.range(8, 121, AxisType.UPCAST)
  acc_res = acc[td_a, e_a].store(acc_p[td_a, e_a].load() * alpha).end(e_a, td_a)
  ps_r, j_p = _reg((1,), 0.0, qk_done), UOp.range(4, 122)
  p0, p1 = _fast_exp2(S_cur[2 * j_p].load() - m_safe), _fast_exp2(S_cur[2 * j_p + 1].load() - m_safe)
  pk = _cvt_pk_f16(p0, p1)
  if rdna4:
    assert P_reg_buf is not None
    h2 = pk.reshape((1,)).bitcast(dtypes.half)
    p_st = UOp.group(P_reg_buf.after(qk_done)[2 * j_p].store(h2[0]), P_reg_buf.after(qk_done)[2 * j_p + 1].store(h2[1]))
  else:
    assert P_half_buf is not None
    p_st = P_half_buf.after(qk_done)[j_p].store(pk)
  p_done = UOp.group(p_st, ps_r[0].store(ps_r.after(qk_done, j_p)[0].load() + (p0 + p1))).end(j_p)
  sm_done = UOp.group(acc_res, p_done, m_i[0].store(m_new),
                      l_i[0].store(alpha * l_i.after(rnd, qk_done)[0].load() +
                                   warp_reduce(ps_r.after(p_done)[0].load(), Ops.ADD, masks=(16,), dpp=rdna4)))
  v_ready = UOp.group(sm_done, v_st).barrier()
  td_pv = UOp.range(TD, 150)
  d_row = td_pv * 16 + lane_c
  if rdna4:
    assert P_reg_buf is not None
    k_next = store_k((rnd + 1).minimum((n_end_r - 1).maximum(0)), (v_ready,), 103)
    V_lds = v_lds_raw.after(v_ready).reshape(D, V_CH_tot, 8)
    last_d = acc[td_pv].store(
      UOp.wmma(V_lds[d_row, swz_v(d_row, wn * 2 + lane_g)], P_reg_buf.after(v_ready), acc.after(v_ready)[td_pv], *WMMA_ARG)).end(td_pv)
    df = UOp.group(last_d, k_next).barrier().end(rnd)
  else:
    assert P_half_buf is not None and P_tmp_buf is not None
    V_lds = v_lds_raw.after(v_ready).reshape(waves_n, D, 2, 8)
    p_f = _r3_unpack_p(P_half_buf.after(v_ready), P_tmp_buf, is_g0, 155, v_ready)
    v_f = V_lds[wn, d_row, swz_v(d_row, UOp.const(0, dtypes.weakint))].cat(V_lds[wn, d_row, swz_v(d_row, UOp.const(1, dtypes.weakint))])
    df = UOp.group(v_ready, acc[td_pv].store(UOp.wmma(v_f, p_f, acc.after(v_ready)[td_pv], *WMMA_ARG)).end(td_pv)).barrier().end(rnd)
  acc_f, m_f, l_f = acc.after(df), m_i.after(df), l_i.after(df)

  b_pre = df.barrier()
  l_inv_w = (l_f[0].load() > UOp.const(0.0, dtypes.float32)).where(l_f[0].load().reciprocal(), UOp.const(0.0, dtypes.float32))
  if rdna4:
    acc_lds, ml_f32, td_w, e_w = k_lds_raw.after(b_pre), v_lds_raw.after(b_pre).bitcast(dtypes.float32), UOp.range(TD, 300), UOp.range(8, 301)
    st_acc = acc_lds[(wn * 16 + lane_c) * D + td_w * 16 + lane_g * 8 + e_w].store(
      (acc_f[td_w, e_w].load() * l_inv_w).cast(dtypes.half)).end(e_w, td_w)
    b_red = UOp.group(st_acc, ml_f32[((wn * 16 + lane_c) * 2).valid(is_g0)].store(m_f[0].load()),
                      ml_f32[((wn * 16 + lane_c) * 2 + 1).valid(is_g0)].store(l_f[0].load())).barrier()
    acc_lr, ml_lr, tot_out = k_lds_raw.after(b_red), v_lds_raw.after(b_red).bitcast(dtypes.float32), 16 * D
    fi = UOp.range((tot_out + threads - 1) // threads, 310)
    flat = tid + fi * threads
    ok_o, flat_s = flat < tot_out, flat.minimum(tot_out - 1)
    r_o, c_o = flat_s // D, flat_s % D
    mr, wi_m = _reg((1,), -1e30, fi), UOp.range(waves_n, 311)
    M_safe = _fast_fmax(mr.after(mr[0].store(_fast_fmax(mr.after(wi_m)[0].load(), ml_lr[(wi_m * 16 + r_o) * 2].load())).end(wi_m))[0].load(),
                        UOp.const(-1e30, dtypes.float32))
    ovr, lsr, wi_a = _reg((1,), 0.0, M_safe), _reg((1,), 0.0, M_safe), UOp.range(waves_n, 312)
    w_i = _fast_exp2(ml_lr[(wi_a * 16 + r_o) * 2].load() - M_safe) * ml_lr[(wi_a * 16 + r_o) * 2 + 1].load()
    ad = UOp.group(ovr[0].store(ovr.after(wi_a)[0].load() + w_i * acc_lr[(wi_a * 16 + r_o) * D + c_o].load().float()),
                   lsr[0].store(lsr.after(wi_a)[0].load() + w_i)).end(wi_a)
    ls = lsr.after(ad)[0].load()
    inv_l = (ls > UOp.const(0.0, dtypes.float32)).where(ls.reciprocal(), UOp.const(0.0, dtypes.float32))
    return o[block_bh, (m_base + r_o).valid(ok_o), c_o].store((ovr.after(ad)[0].load() * inv_l).cast(o.dtype)).end(
      fi, lane, wn, block_m_raw, block_bh).sink(arg=KernelInfo(name="rdna4_flash_attn_smallm", opts_to_apply=()))
  # RDNA3: reuse single 32 KB kv_lds (upper half ml_f32, lower half 16 KB D//2 per half!)
  ml_base = (block_n_tot * D) // 4
  ml_f32 = k_lds_raw.after(b_pre).bitcast(dtypes.float32)
  st_ml = UOp.group(ml_f32[(ml_base + (wn * 16 + lane_c) * 2).valid(is_g0)].store(m_f[0].load()),
                    ml_f32[(ml_base + (wn * 16 + lane_c) * 2 + 1).valid(is_g0)].store(l_f[0].load())).barrier()
  ml_r, dh = ml_f32.after(st_ml), UOp.range(2, 289)
  o_h16, td_s, e_w = k_lds_raw.after(st_ml, dh), UOp.range(TD // 2, 290), UOp.range(8, 291)
  st_o = o_h16[wn * 16 * (D // 2) + lane_c * (D // 2) + td_s * 16 + e_w * 2 + lane_g].store(
    (acc_f[dh * (TD // 2) + td_s, e_w].load() * l_inv_w).cast(dtypes.half)).end(e_w, td_s).barrier()
  o_r, ri, ci = k_lds_raw.after(st_o), UOp.range(2, 310), UOp.range((D // 2) // 32, 315)
  r_out, c_loc = wn * 2 + ri, lane * ((D // 2) // 32) + ci
  mr, wk_m = _reg((1,), -1e30, ri, ci), UOp.range(waves_n, 316)
  M_safe = _fast_fmax(mr.after(mr[0].store(_fast_fmax(mr.after(wk_m)[0].load(), ml_r[ml_base + (wk_m * 16 + r_out) * 2].load())).end(wk_m))[0].load(),
                      UOp.const(-1e30, dtypes.float32))
  ovr, lsr, wk_a = _reg((1,), 0.0, M_safe), _reg((1,), 0.0, M_safe), UOp.range(waves_n, 317)
  w_i = _fast_exp2(ml_r[ml_base + (wk_a * 16 + r_out) * 2].load() - M_safe) * ml_r[ml_base + (wk_a * 16 + r_out) * 2 + 1].load()
  ad = UOp.group(ovr[0].store(ovr.after(wk_a)[0].load() + w_i * o_r[wk_a * 16 * (D // 2) + r_out * (D // 2) + c_loc].load().float()),
                 lsr[0].store(lsr.after(wk_a)[0].load() + w_i)).end(wk_a)
  ls = lsr.after(ad)[0].load()
  w_inv = (ls > UOp.const(0.0, dtypes.float32)).where(ls.reciprocal(), UOp.const(0.0, dtypes.float32))
  ep = o[block_bh, m_base + r_out, dh * (D // 2) + c_loc].store((ovr.after(ad)[0].load() * w_inv).cast(o.dtype)).end(ci, ri).barrier().end(dh)
  return ep.end(lane, wn, block_m_raw, block_bh).sink(arg=KernelInfo(name="rdna3_flash_attn_smallm", opts_to_apply=()))

@functools.cache
def _rdna4_fa_kernel(o:UOp, q:UOp, cache:UOp, *, valid_kv_len:UOp, q_start:UOp, d_blks:UOp, qm:int=1, ds2:bool=False) -> UOp:
  # Unified RDNA4 FlashAttention kernel (qm=2 D<=128, qm=1 D=192..256, ds2=True D>=320 or D=256 GQA%6==0):
  # 100% K/V pipelined (global_load_tr_b128 for V), 2 barriers/tile, <=251 VGPRs, 0 spills across all shapes.
  valid_kv_len, q_start, (BH, M, D), (_, B, H_KV, physical_n, _) = _unbind(valid_kv_len), _unbind(q_start), q.shape, cache.shape
  assert isinstance(BH, UOp) and isinstance(M, UOp) and isinstance(D, UOp) and isinstance(B, UOp) and isinstance(H_KV, UOp)
  zf, tn, waves_d = UOp.const(0.0, dtypes.float32), 1 if ds2 else 2, 2 if ds2 else 1
  gqa_group, m_qm_tot = BH // (B * H_KV), M // (qm * 16)
  gqa_w4 = (gqa_group % 4).eq(0).where(4, (gqa_group % 3).eq(0).where(3, (gqa_group % 2).eq(0).where(2, 1)))
  if ds2:
    is_g6 = (d_blks <= 4) & (gqa_group % 6).eq(0)
    gqa_w = is_g6.where(6, gqa_w4)
    max_wph = (is_g6.where(12, 8) // (gqa_w * 2)).maximum(1)
  else:
    gqa_w = (gqa_group % 8).eq(0).where(8, gqa_w4)
    max_wph = (8 // gqa_w).maximum(1)
  n_hg = BH // gqa_w
  def _wph_ok(k:int) -> UOp:
    c = (max_wph >= k) & (m_qm_tot % k).eq(0)
    return (c & (n_hg * (m_qm_tot // k) >= 64)) if qm == 2 else c
  w_per_head = _wph_ok(8).where(8, _wph_ok(4).where(4, _wph_ok(2).where(2, 1)))
  total_waves, block_m_sz, block_n_sz = gqa_w * w_per_head * waves_d, w_per_head * qm * WMMA_M, tn * WMMA_N
  threads, m_blks_eff, TD_w = total_waves * WARP_SIZE, M // block_m_sz, (d_blks * 4) // waves_d
  KSTR_8, VSTR_8, DG, KGRP = d_blks * 8 + 1, 3 if ds2 else 5, d_blks * 8, 2 if ds2 else 4
  KITEMS, NTR, LGRP = block_n_sz * DG, KGRP * DG, threads // 8
  KP, VP = (KITEMS + threads - 1) // threads, (NTR + LGRP - 1) // LGRP
  KVP, SCALE_L2 = KP.maximum(VP), D.cast(dtypes.float32).rsqrt() * LOG2E

  ax_hg, ax_m = (1, 0) if qm == 2 else (0, 1)
  block_hg, block_m_raw = UOp.range(n_hg, ax_hg, AxisType.GLOBAL), UOp.range(m_blks_eff, ax_m, AxisType.GLOBAL)
  block_m = m_blks_eff - 1 - block_m_raw
  wave_tot, lane = UOp.range(total_waves, 2, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  wave, wd = (wave_tot // 2, wave_tot % 2) if ds2 else (wave_tot, UOp.const(0, dtypes.weakint))
  tid = wave_tot * WARP_SIZE + lane
  lane_g, lane_c, vj, lgi = lane // 16, lane % 16, tid & 7, tid >> 3
  g_idx, qbi = wave % gqa_w, wave // gqa_w
  block_bh, kv_head = block_hg * gqa_w + g_idx, block_hg // (gqa_group // gqa_w)
  m_base, td_base = block_m * block_m_sz + qbi * qm * 16, wd * TD_w

  q_u32, k_u32 = q.flatten().bitcast(dtypes.uint32), cache.flatten().bitcast(dtypes.uint32)
  v_flat, v_base_h = cache.flatten(), B * H_KV * physical_n * D
  q_reg_u32, qv_b = UOp.alloc((qm, TD_w, 4), dtypes.uint32, addrspace=AddrSpace.REG), UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG)
  qi_q, kq_i, p_q = UOp.range(qm, 9), UOp.range(TD_w, 10), UOp.range(4, 11)
  qv_r = qv_b.after(qv_b.after(qi_q, kq_i).store(
    _amd_load(q_u32[(block_bh * M + m_base + qi_q * 16 + lane_c) * (d_blks * 32) + (td_base + kq_i) * 8 + lane_g * 4], 4)))
  hq = qv_r[p_q].load().reshape((1,)).bitcast(dtypes.half)
  q_reg = q_reg_u32.after(q_reg_u32[qi_q, kq_i, p_q].store(
    _cvt_pk_f16(hq[0].float() * SCALE_L2, hq[1].float() * SCALE_L2, inline=False)).end(p_q, kq_i, qi_q))

  sK = UOp.alloc((block_n_sz * KSTR_8 * 8,), dtypes.half, addrspace=AddrSpace.LOCAL)
  sV = UOp.alloc((D * VSTR_8 * 8,), dtypes.half, addrspace=AddrSpace.LOCAL)
  sX = UOp.alloc((total_waves * WARP_SIZE * 8,), dtypes.float32, addrspace=AddrSpace.LOCAL) if ds2 else None
  acc, m_i, l_i = _reg((qm, TD_w, 8), 0.0), _reg((qm,), -1e30), _reg((qm,), 0.0)
  al_reg, S_reg_buf = UOp.alloc((qm,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((qm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG)
  P_reg_u32 = UOp.alloc((qm, tn, 4), dtypes.uint32, addrspace=AddrSpace.REG)
  f_buf_u32 = UOp.alloc((2 if qm == 2 else 4, 4), dtypes.uint32, addrspace=AddrSpace.REG)
  kv_reg, tok = UOp.alloc((KVP, 4), dtypes.uint32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)

  kv_max, one_h2 = (valid_kv_len - 1).maximum(0), UOp.const(0x3c003c00, dtypes.uint32)
  n_end_all = ((q_start + (block_m + 1) * block_m_sz).minimum(valid_kv_len) + block_n_sz - 1) // block_n_sz
  n_full = ((q_start + block_m * block_m_sz + 1).minimum(valid_kv_len) // block_n_sz).minimum(n_end_all)

  p_k0 = UOp.range(KP, 20)
  i_k0 = tid + p_k0 * threads
  ok_k0, i_k0s = i_k0 < KITEMS, i_k0.minimum(KITEMS - 1)
  kv_u0 = _amd_load(k_u32[(kv_head * physical_n + (i_k0s // DG).minimum(kv_max)) * (d_blks * 32) + (i_k0s % DG) * 4], 4)
  sK_u32_0 = sK.after(q_reg).bitcast(dtypes.uint32).reshape(block_n_sz, KSTR_8, 4)
  k0_b = sK_u32_0[(i_k0s // DG).valid(ok_k0), i_k0s % DG].store(kv_u0).end(p_k0).barrier()

  ti = UOp.range(n_end_all, 100)
  k0 = ti * block_n_sz
  K_lds_u32 = sK.after(k0_b, ti).bitcast(dtypes.uint32).reshape(block_n_sz, KSTR_8, 4)

  # Prefetch V[ti] -> kv_reg before QK
  r_v = UOp.range(VP, 102)
  u_v = (lgi + r_v * LGRP).minimum(NTR - 1)
  v_ptr = v_flat[v_base_h + (kv_head * physical_n + (k0 + (u_v % KGRP) * 8 + vj).minimum(kv_max)) * D + (u_v // KGRP) * 8]
  vr_call = UOp.native(
    "({{ typedef __fp16 _hv8 __attribute__((__vector_size__(8 * sizeof(__fp16)))); "
    "*(_hv8*)({0}) = __builtin_amdgcn_global_load_tr_b128_v8f16((_hv8*)({1})); 0; }})",
    shape=(), dtype=dtypes.int32, fmt=True).call(kv_reg.after(k0_b, ti)[r_v, 0], v_ptr, inline=False)
  vr_ld0 = tok.after(k0_b, ti)[0].store(vr_call).end(r_v)
  vr_ld = m_i[0].store(_sb0(m_i.after(k0_b, ti, vr_ld0)[0].load()))

  def _lds_wmma_loop(n_it:UOp, n_f:int, rid:int, dep:UOp, ld_fn, wmma_fn) -> UOp:
    r_i = UOp.range(n_it, rid)
    st = UOp.group(*(f_buf_u32.after(dep, r_i)[j].store(ld_fn(r_i, j)) for j in range(n_f)))
    sb = UOp.group(st, f_buf_u32[0, 0].store(_sb0(f_buf_u32.after(st)[0, 0].load())))
    fr, wt = f_buf_u32.after(sb), sb
    return wmma_fn(r_i, fr, wt).end(r_i)

  S_reg = S_reg_buf.after(S_reg_buf.after(k0_b, ti, vr_ld).store(S_reg_buf.const_like(0.0)))
  if ds2:
    n_b4, n_r2 = TD_w // 4, (TD_w % 4) // 2
    def _qk_ds2(n_it:UOp, un:int, base:int|UOp, rid:int, dep:UOp) -> UOp:
      def _w(r_i, fr, wt):
        for j in range(un): wt = _wmma_u4(fr[j, 0], q_reg[0, base + r_i * un + j, 0], S_reg.after(dep, wt)[0, 0, 0], tok, wt)
        return wt
      return _lds_wmma_loop(n_it, un, rid, dep, lambda r_i, j: K_lds_u32.after(r_i)[lane_c, (td_base + base + r_i * un + j) * 2 + lane_g].load(), _w)
    qk_d4 = _qk_ds2(n_b4, 4, 0, 111, S_reg)
    qk_done = UOp.group(qk_d4, _qk_ds2(n_r2, 2, n_b4 * 4, 112, qk_d4))
  elif qm == 2:
    def _qk_qm2(r_i, fr, wt):
      for tni in range(2):
        for qi in range(2): wt = _wmma_u4(fr[tni, 0], q_reg[qi, r_i, 0], S_reg.after(wt)[qi, tni, 0], tok, wt)
      return wt
    qk_done = _lds_wmma_loop(TD_w, 2, 111, S_reg, lambda r_i, tni: K_lds_u32[tni * 16 + lane_c, r_i * 2 + lane_g].load(), _qk_qm2)
  else:
    def _qk_qm1(r_i, fr, wt):
      for j in range(2):
        for tni in range(2): wt = _wmma_u4(fr[j * 2 + tni, 0], q_reg[0, r_i * 2 + j, 0], S_reg.after(wt)[0, tni, 0], tok, wt)
      return wt
    qk_done = _lds_wmma_loop(TD_w // 2, 4, 111, S_reg,
                             lambda r_i, k: K_lds_u32[(k % 2) * 16 + lane_c, (r_i * 2 + k // 2) * 2 + lane_g].load(), _qk_qm1)

  vr_r = kv_reg.after(qk_done, vr_ld)
  sV_u32 = sV.after(qk_done, vr_ld).bitcast(dtypes.uint32).reshape(D, VSTR_8, 4)
  su_n = 2 if qm == 2 else 4
  r_vs, st_vs = UOp.range((VP + su_n - 1) // su_n, 103), []
  for su in range(su_n):
    r_i = r_vs * su_n + su
    u_vs = lgi + r_i * LGRP
    ok_vs, u_vss = (r_i < VP) & (u_vs < NTR), u_vs.minimum(NTR - 1)
    st_vs.append(sV_u32[((u_vss // KGRP) * 8 + vj).valid(ok_vs), u_vss % KGRP].store(vr_r[r_i.minimum(KVP - 1)].load()))
  st_v = UOp.group(*st_vs).end(r_vs)
  if ds2:
    assert sX is not None
    e_w1 = UOp.range(8, 116, AxisType.UPCAST)
    st_sx = sX.after(qk_done)[(wave_tot * WARP_SIZE + lane) * 8 + e_w1].store(S_reg.after(qk_done)[0, 0, e_w1].load()).end(e_w1)
    b_mid = UOp.group(st_v, st_sx).barrier()
    e_w2, peer_w = UOp.range(8, 117, AxisType.UPCAST), wave_tot ^ UOp.const(1, dtypes.weakint)
    s_dep = S_reg[0, 0, e_w2].store(S_reg.after(b_mid)[0, 0, e_w2].load() + sX.after(b_mid)[(peer_w * WARP_SIZE + lane) * 8 + e_w2].load()).end(e_w2)
    S_cur = S_reg.after(s_dep)
  else:
    b_mid = s_dep = st_v.barrier()
    S_cur = S_reg.after(qk_done, b_mid)

  r_cm, tn_m, e_m = UOp.range((ti >= n_full).cast(dtypes.weakint), 118), UOp.range(tn, 119), UOp.range(8, 120, AxisType.UPCAST)
  k_idx = k0 + tn_m * 16 + lane_g * 8 + e_m
  S_cur = S_reg.after(UOp.group(*(S_reg[qi, tn_m, e_m].store(
    ((k_idx <= q_start + m_base + qi * 16 + lane_c) & (k_idx < valid_kv_len)).where(
      S_cur.after(r_cm)[qi, tn_m, e_m].load(), UOp.const(-math.inf, dtypes.float32))) for qi in range(qm))).end(e_m, tn_m, r_cm))

  need_g = UOp.const(False, dtypes.bool)
  for qi in range(qm):
    smax = _tree_fmax([S_cur[qi, tni, e].load() for tni in range(tn) for e in range(8)])
    need_g = need_g | (smax > (m_i.after(ti, s_dep)[qi].load() + UOp.const(14.0, dtypes.float32)))
  r_grow = UOp.range(_nat("(__builtin_amdgcn_ballot_w32({0}) != 0u)", dtypes.bool, need_g).cast(dtypes.weakint), 121)
  g_sts = []
  for qi in range(qm):
    m_p = m_i.after(ti, s_dep, r_grow)[qi].load()
    sm_i = _tree_fmax([S_cur.after(r_grow)[qi, tni, e].load() for tni in range(tn) for e in range(8)])
    rm = _fast_fmax(sm_i, _lane_xor(sm_i, 16, inline=False, dpp=True))
    m_n = _fast_fmax(_fast_fmax(m_p, rm), UOp.const(-1e30, dtypes.float32))
    al_s = al_reg.after(r_grow)[qi].store(_fast_exp2(m_p - m_n))
    al = al_reg.after(al_s)[qi].load()
    td_a, e_a = UOp.range(TD_w, 122 + qi * 2), UOp.range(8, 123 + qi * 2, AxisType.UPCAST)
    g_sts += [
      acc[qi, td_a, e_a].store(acc.after(ti, s_dep, al_s)[qi, td_a, e_a].load() * al).end(e_a, td_a),
      l_i.after(al_s)[qi].store(l_i.after(ti, s_dep, al_s)[qi].load() * al),
      m_i.after(al_s)[qi].store(m_n),
    ]
  acc_res = UOp.group(*g_sts).end(r_grow)

  k0_next, p_k = (ti + 1) * block_n_sz, UOp.range(KP, 101)
  i_k = (tid + p_k * threads).minimum(KITEMS - 1)
  kv_u = _amd_load(k_u32[(kv_head * physical_n + (k0_next + i_k // DG).minimum(kv_max)) * (d_blks * 32) + (i_k % DG) * 4], 4)
  kr_ld0 = kv_reg.after(b_mid, acc_res)[p_k].store(kv_u).end(p_k)
  kr_ld = UOp.group(kr_ld0, l_i[0].store(_sb0(l_i.after(acc_res, kr_ld0)[0].load())))

  qi_p, tni_p, j_p = UOp.range(qm, 130), UOp.range(tn, 131), UOp.range(4, 132)
  m_cur = m_i.after(acc_res, kr_ld)[qi_p].load()
  pk = _cvt_pk_f16(_fast_exp2(S_cur[qi_p, tni_p, 2 * j_p].load() - m_cur),
                   _fast_exp2(S_cur[qi_p, tni_p, 2 * j_p + 1].load() - m_cur), inline=False)
  p_done = UOp.group(P_reg_u32.after(acc_res, kr_ld)[qi_p, tni_p, j_p].store(pk),
                     l_i[qi_p].store(_amd_fdot2(pk, one_h2, l_i.after(acc_res, kr_ld, qi_p, tni_p, j_p)[qi_p].load()))).end(j_p, tni_p, qi_p)

  V_lds_u32, p_frag = sV.after(b_mid, p_done).bitcast(dtypes.uint32).reshape(D, VSTR_8, 4), P_reg_u32.after(p_done)
  if ds2:
    def _pv_ds2(n_it:UOp, un:int, base:int|UOp, rid:int, dep:UOp) -> UOp:
      def _w(r_i, fr, wt):
        for j in range(un): wt = _wmma_u4(fr[j, 0], p_frag[0, 0, 0], acc.after(dep, r_i, wt)[0, base + r_i * un + j, 0], tok, wt)
        return wt
      return _lds_wmma_loop(n_it, un, rid, dep, lambda r_i, j: V_lds_u32.after(r_i)[(td_base + base + r_i * un + j) * 16 + lane_c, lane_g].load(), _w)
    pv_d4 = _pv_ds2(n_b4, 4, 0, 125, p_done)
    pv_done = UOp.group(pv_d4, _pv_ds2(n_r2, 2, n_b4 * 4, 126, pv_d4))
  else:
    pv_t, un_v = p_done, 2 if qm == 2 else 4
    for tni in range(2):
      def _pv_lo(r_i, fr, wt, tni=tni, pv_t=pv_t):
        for j in range(un_v):
          for qi in range(qm): wt = _wmma_u4(fr[j, 0], p_frag[qi, tni, 0], acc.after(pv_t, r_i, wt)[qi, r_i * un_v + j, 0], tok, wt)
        return wt
      pv_t = _lds_wmma_loop(TD_w // un_v, un_v, 126 + tni, pv_t,
                            lambda r_i, j, tni=tni: V_lds_u32[(r_i * un_v + j) * 16 + lane_c, tni * 2 + lane_g].load(), _pv_lo)
    pv_done = pv_t

  kr_r = kv_reg.after(pv_done, kr_ld)
  sK_u32 = sK.after(pv_done, kr_ld).bitcast(dtypes.uint32).reshape(block_n_sz, KSTR_8, 4)
  p_ks, st_ks = UOp.range((KP + su_n - 1) // su_n, 104), []
  for su in range(su_n):
    p_i = p_ks * su_n + su
    i_ks0 = tid + p_i * threads
    ok_ks, i_kss = (p_i < KP) & (i_ks0 < KITEMS), i_ks0.minimum(KITEMS - 1)
    st_ks.append(sK_u32[(i_kss // DG).valid(ok_ks), i_kss % DG].store(kr_r[p_i.minimum(KVP - 1)].load()))
  loop_done = UOp.group(*st_ks).end(p_ks).barrier().end(ti)

  acc_f, l_f = acc.after(loop_done), l_i.after(loop_done)
  qi_o, td_o, e_o = UOp.range(qm, 299), UOp.range(TD_w, 300), UOp.range(8, 301, AxisType.UPCAST)
  l_tot = l_f[qi_o].load() + _lane_xor(l_f[qi_o].load(), 16, inline=False, dpp=True)
  l_inv = (l_tot > zf).where(l_tot.reciprocal(), zf)
  return o[block_bh, m_base + qi_o * 16 + lane_c, (td_base + td_o) * 16 + lane_g * 8 + e_o].store(
    (acc_f[qi_o, td_o, e_o].load() * l_inv).cast(o.dtype)).end(
    e_o, td_o, qi_o, lane, wave_tot, block_hg, block_m_raw).sink(arg=KernelInfo(name="rdna4_flash_attn", opts_to_apply=()))

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, *, valid_kv_len:UOp, q_start:UOp, d_blks:UOp, rdna4:bool=False) -> UOp:
  if rdna4:
    BH, M, B, H_KV = q.shape[0], q.shape[1], cache.shape[1], cache.shape[2]
    assert isinstance(BH, UOp) and isinstance(M, UOp) and isinstance(B, UOp) and isinstance(H_KV, UOp)
    gqa_group = BH // (B * H_KV)
    use_dsplit = (d_blks >= 5) | ((d_blks >= 3) & (gqa_group % 6).eq(0))
    use_qm2 = (d_blks <= 2) & (M >= 512)
    return UOp(Ops.WHERE, src=(use_dsplit,
      _rdna4_fa_kernel(o, q, cache, valid_kv_len=valid_kv_len, q_start=q_start, d_blks=d_blks, ds2=True),
      UOp(Ops.WHERE, src=(use_qm2,
        _rdna4_fa_kernel(o, q, cache, valid_kv_len=valid_kv_len, q_start=q_start, d_blks=d_blks, qm=2),
        _rdna4_fa_kernel(o, q, cache, valid_kv_len=valid_kv_len, q_start=q_start, d_blks=d_blks, qm=1)))))
  valid_kv_len, q_start, (BH, M, D), (_, B, H_KV, physical_n, _) = _unbind(valid_kv_len), _unbind(q_start), q.shape, cache.shape
  assert isinstance(BH, UOp) and isinstance(M, UOp) and isinstance(D, UOp) and isinstance(B, UOp) and isinstance(H_KV, UOp)
  gqa_group = BH // (B * H_KV)
  waves_d = ((d_blks >= 4) & (d_blks % 2).eq(0)).where(2, 1)
  use_wd, gqa_base = waves_d > 1, (gqa_group % 4).eq(0).where(4, (gqa_group % 2).eq(0).where(2, 1))
  gqa_w = use_wd.where((gqa_group % 2).eq(0).where(2, 1), gqa_base)
  w_per_head = use_wd.where((d_blks >= 6).where(1, 2), UOp.const(2, dtypes.weakint))
  waves_m, bn_tiles = gqa_w * w_per_head, 2
  total_waves, block_m_sz, block_n_sz = waves_m * waves_d, w_per_head * WMMA_M, bn_tiles * WMMA_N
  threads, m_blks_eff, TN, TD, K_CH, V_CH = total_waves * WARP_SIZE, M // block_m_sz, bn_tiles, d_blks * 4, d_blks * 8, bn_tiles * 2
  TD_w, SCALE_L2, zw = TD // waves_d, D.cast(dtypes.float32).rsqrt() * LOG2E, UOp.const(0, dtypes.weakint)

  block_hg, block_m_raw = UOp.range(BH // gqa_w, 0, AxisType.GLOBAL), UOp.range(m_blks_eff, 1, AxisType.GLOBAL)
  block_m = m_blks_eff - 1 - block_m_raw
  wave_tot, lane = UOp.range(total_waves, 2, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  wave, wd, tid, lane_g, lane_c = wave_tot // waves_d, wave_tot % waves_d, wave_tot * WARP_SIZE + lane, lane // 16, lane % 16
  is_g0, block_bh, wave_m, kv_head = lane_g.eq(0), block_hg * gqa_w + (wave // w_per_head), wave % w_per_head, block_hg // (gqa_group // gqa_w)
  k_u32, v_base_u32 = cache.flatten().bitcast(dtypes.uint32), B * H_KV * physical_n * (d_blks * 32)
  m_base, td_base = block_m * block_m_sz + wave_m * 16, wd * TD_w
  q_reg, _ = _fa_load_q(q, block_bh, m_base, td_base, TD_w, d_blks, M, lane_c, lane_g, SCALE_L2, False)

  k_lds_raw = UOp.alloc((block_n_sz * D,), dtypes.half, addrspace=AddrSpace.LOCAL)
  v_lds_raw = UOp.alloc((block_n_sz * D,), dtypes.half, addrspace=AddrSpace.LOCAL)
  s_lds = UOp.alloc((waves_m, waves_d, WARP_SIZE, TN, 8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  acc, m_i, l_i, S_reg_buf = _reg((TD_w, 8), 0.0), _reg((1,), -1e30), _reg((1,), 0.0), UOp.alloc((TN, 8), dtypes.float32, addrspace=AddrSpace.REG)
  P_half_buf = UOp.alloc((TN, 4), dtypes.uint32, addrspace=AddrSpace.REG)
  P_tmp_buf = UOp.alloc((16,), dtypes.half, addrspace=AddrSpace.REG)

  n_end_all = ((q_start + (block_m + 1) * block_m_sz).minimum(valid_kv_len) + block_n_sz - 1) // block_n_sz
  n_full = ((q_start + block_m * block_m_sz + 1).minimum(valid_kv_len) // block_n_sz).minimum(n_end_all)
  def swz_v_ch(d_i:UOp, ch:int|UOp) -> UOp: return ch ^ ((d_i >> 1) & 3)
  k_items, v_items, sub_u = block_n_sz * TD, (block_n_sz // 2) * (d_blks * 8), 2

  def store_k(tile_idx:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    k_lds_u32 = k_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(block_n_sz, K_CH, 4)
    ki, ks = UOp.range((k_items + threads * sub_u - 1) // (threads * sub_u), rid), []
    for su in range(sub_u):
      bid = tid + (ki * sub_u + su) * threads
      ok, bid_s = bid < k_items, bid.minimum(k_items - 1)
      kr, kk_i = bid_s // TD, bid_s % TD
      kb = (kv_head * physical_n + tile_idx * block_n_sz + kr) * (d_blks * 32) + kk_i * 8
      v0, v1, kr_v = _amd_load(k_u32[kb], 4), _amd_load(k_u32[kb + 4], 4), kr.valid(ok)
      ks += [k_lds_u32[kr_v, (kk_i * 2) ^ (kr & 7)].store(v0), k_lds_u32[kr_v, (kk_i * 2 + 1) ^ (kr & 7)].store(v1)]
    return UOp.group(*ks).end(ki)

  def store_v(tile_idx:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    v_lds_u32, z32 = v_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(D, V_CH, 4), UOp.const(0, dtypes.uint32)
    c_lo, c_hi, n_ctiles = UOp.const(0x05040100, dtypes.uint32), UOp.const(0x07060302, dtypes.uint32), d_blks * 8
    ti, vs = UOp.range((v_items + threads * sub_u - 1) // (threads * sub_u), rid), []
    for su in range(sub_u):
      tbid = tid + (ti * sub_u + su) * threads
      ok, tbid_s = tbid < v_items, tbid.minimum(v_items - 1)
      c0, pid = (tbid_s % n_ctiles) * 8, tbid_s // n_ctiles
      blk4, par = pid // 2, pid % 2
      ga, gb, ch_a, slot_a = tile_idx * block_n_sz + blk4 * 4 + par, tile_idx * block_n_sz + blk4 * 4 + 2 + par, (blk4 // 4) * 2 + par, blk4 % 4
      va = _amd_load(k_u32[v_base_u32 + (kv_head * physical_n + ga.minimum(physical_n - 1)) * (d_blks * 32) + c0 // 2], 4)
      vb = _amd_load(k_u32[v_base_u32 + (kv_head * physical_n + gb.minimum(physical_n - 1)) * (d_blks * 32) + c0 // 2], 4)
      ra_b, rb_b = UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG)
      ra_r, rb_r = ra_b.after(ra_b.after(*dep, ti).store(va)), rb_b.after(rb_b.after(*dep, ti).store(vb))
      p = UOp.range(4, rid + 50 + su)
      ra_p, rb_p = (ga < valid_kv_len).where(ra_r[p].load(), z32), (gb < valid_kv_len).where(rb_r[p].load(), z32)
      vs.append(UOp.group(
        v_lds_u32[(c0 + p * 2).valid(ok), swz_v_ch(c0 + p * 2, ch_a), slot_a].store(_amd_byte_perm(rb_p, ra_p, c_lo)),
        v_lds_u32[(c0 + p * 2 + 1).valid(ok), swz_v_ch(c0 + p * 2 + 1, ch_a), slot_a].store(_amd_byte_perm(rb_p, ra_p, c_hi))).end(p))
    return UOp.group(*vs).end(ti)

  k0 = store_k(zw, (q_reg,), 20).barrier()
  it = UOp.range(n_end_all, 100)
  K_lds = k_lds_raw.after(k0, it).reshape(block_n_sz, K_CH, 8)
  S_reg = S_reg_buf.after(S_reg_buf.after(it).store(S_reg_buf.const_like(0.0)))
  k_s = UOp.range(TD_w, 130)
  kk = td_base + k_s
  qk_done = UOp.group(*(S_reg[tni].store(UOp.wmma(
    K_lds[tni * 16 + lane_c, (kk * 2) ^ ((tni * 16 + lane_c) & 7)].cat(K_lds[tni * 16 + lane_c, (kk * 2 + 1) ^ ((tni * 16 + lane_c) & 7)]),
    q_reg[k_s], S_reg.after(k_s)[tni], *WMMA_ARG)) for tni in range(TN))).end(k_s)

  r_wd, e_r = UOp.range(use_wd.cast(dtypes.weakint), 138), UOp.range(8, 140, AxisType.UPCAST)
  s_st_b = UOp.group(*(s_lds.after(qk_done, r_wd)[wave.valid(use_wd), wd, lane, tn_i, e_r].store(
    S_reg.after(qk_done, r_wd)[tn_i, e_r].load()) for tn_i in range(TN))).end(e_r).barrier()
  e_r2, wd_r = UOp.range(8, 142, AxisType.UPCAST), UOp.range(waves_d, 143)
  qk_all = UOp.group(qk_done, UOp.group(*(S_reg[tn_i, e_r2].store(
    s_lds.after(s_st_b)[wave, wd_r, lane, tn_i, e_r2].load().reduce(wd_r, arg=(Ops.ADD, 0))) for tn_i in range(TN))).end(e_r2, r_wd))
  S_cur = S_reg.after(qk_all)
  v_st = store_v(it, (qk_all,), 101)
  r_cm, tn_m, e_m = UOp.range((it >= n_full).cast(dtypes.weakint), 103), UOp.range(TN, 104), UOp.range(8, 105, AxisType.UPCAST)
  k_idx = it * block_n_sz + tn_m * 16 + e_m * 2 + lane_g
  causal = (k_idx <= q_start + m_base + lane_c) & (k_idx < valid_kv_len)
  S_cur = S_reg.after(S_reg[tn_m, e_m].store(
    causal.where(S_cur.after(r_cm)[tn_m, e_m].load(), UOp.const(-math.inf, dtypes.float32))).end(e_m, tn_m, r_cm))

  row_max = warp_reduce(_tree_fmax([S_cur[ti, ei].load() for ti in range(TN) for ei in range(8)]), Ops.MAX, masks=(16,))
  m_prev = m_i.after(it, qk_all)[0].load()
  row_grow = row_max > (m_prev + UOp.const(12.0, dtypes.float32))
  r_grow = UOp.range(_nat("(__builtin_amdgcn_ballot_w32({0}) != 0u)", dtypes.bool, row_grow).cast(dtypes.weakint), 119)
  m_new = row_grow.where(_fast_fmax(_fast_fmax(m_prev, row_max), UOp.const(-1e30, dtypes.float32)), m_prev)
  alpha, acc_p = _fast_exp2(m_prev - m_new), acc.after(it, qk_all, r_grow)
  td_a, e_a = UOp.range(TD_w, 120), UOp.range(8, 121, AxisType.UPCAST)
  acc_res = UOp.group(
    acc[td_a, e_a].store(acc_p[td_a, e_a].load() * alpha).end(e_a, td_a),
    m_i[0].store(m_new), l_i[0].store(l_i.after(it, qk_all, r_grow)[0].load() * alpha)).end(r_grow)
  m_cur = m_i.after(acc_res)[0].load()
  p_sum_r, tn_p, j_p = _reg((1,), 0.0, acc_res), UOp.range(TN, 110), UOp.range(4, 122)
  p0, p1 = _fast_exp2(S_cur[tn_p, 2 * j_p].load() - m_cur), _fast_exp2(S_cur[tn_p, 2 * j_p + 1].load() - m_cur)
  p_done = UOp.group(P_half_buf.after(acc_res)[tn_p, j_p].store(_cvt_pk_f16(p0, p1)),
                     p_sum_r[0].store(p_sum_r.after(acc_res, tn_p, j_p)[0].load() + (p0 + p1))).end(j_p, tn_p)
  sm_done = UOp.group(acc_res, p_done,
                      l_i[0].store(l_i.after(acc_res)[0].load() + warp_reduce(p_sum_r.after(p_done)[0].load(), Ops.ADD, masks=(16,))))
  v_ready = UOp.group(sm_done, v_st).barrier()
  k_next_par = store_k((it + 1).minimum((n_end_all - 1).maximum(0)), (v_ready,), 112)
  a_cur, V_lds = acc.after(v_ready), v_lds_raw.after(v_ready).reshape(D, V_CH, 8)
  tni, td_pv = UOp.range(TN, 114), UOp.range(TD_w, 115)
  d_row = (td_base + td_pv) * 16 + lane_c
  p_f = _r3_unpack_p(P_half_buf.after(v_ready)[tni], P_tmp_buf, is_g0, 116, a_cur, tni)
  v_f = V_lds[d_row, swz_v_ch(d_row, tni * 2)].cat(V_lds[d_row, swz_v_ch(d_row, tni * 2 + 1)])
  a_cur = acc.after(acc[td_pv].store(UOp.wmma(v_f, p_f, a_cur.after(tni)[td_pv], *WMMA_ARG)).end(td_pv, tni))
  done = UOp.group(a_cur, k_next_par).barrier().end(it)
  acc_f, l_f = acc.after(done), l_i.after(done)
  l_inv = (l_f[0].load() > UOp.const(0.0, dtypes.float32)).where(l_f[0].load().reciprocal(), UOp.const(0.0, dtypes.float32))
  td_o, e_o = UOp.range(TD_w, 300), UOp.range(8, 301, AxisType.UPCAST)
  return o[block_bh, m_base + lane_c, (td_base + td_o) * 16 + e_o * 2 + lane_g].store(
    (acc_f[td_o, e_o].load() * l_inv).cast(o.dtype)).end(
    e_o, td_o, lane, wave_tot, block_m_raw, block_hg).sink(arg=KernelInfo(name="rdna3_flash_attn", opts_to_apply=()))

# ******** gated delta net: unified RDNA4/RDNA3 WMMA Chunked-WY + zero-LDS inline-kq decode ********

@functools.cache
def _gated_delta_wmma_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, *,
                             k_blks:UOp, alpha_k:UOp, start_pos:UOp, rdna4:bool|UOp=True) -> UOp:
  batch, heads, tokens, value_dim = core.shape
  assert isinstance(value_dim, UOp)
  is_kda = alpha_k > 1
  D, BH, k_tiles, v_tiles = k_blks * 32, batch * heads, k_blks * 2, value_dim // 4
  v_wgs, n_vsub, k_split = (v_tiles // 8).maximum(1), 2, (k_tiles >= 8).where(4, 2)
  kt_w, waves = k_tiles // k_split, n_vsub * k_split
  threads, sup_ch = waves * WARP_SIZE, is_kda.where((k_blks >= 8).where(1, 2), (k_blks >= 8).where(2, 4))
  sup_tok = sup_ch * 16

  core_f, v_f = core.reshape(BH, tokens, value_dim), v.reshape(BH, tokens, value_dim)
  q_f, k_f = q.reshape(BH, tokens, D), k.reshape(BH, tokens, D)
  beta_f, alpha_f, state_f = beta.reshape(BH, tokens), alpha.reshape(BH, tokens, alpha_k), state.reshape(BH, value_dim, D)
  wg, wave, lane = UOp.range(BH * v_wgs, 0, AxisType.GLOBAL), UOp.range(waves, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  bh, v_wg, tid, lane_g, lane_c = wg // v_wgs, wg % v_wgs, wave * WARP_SIZE + lane, lane // 16, lane % 16
  v_sub, kw, w_ch, w_role = wave // k_split, wave % k_split, wave % sup_ch, wave // sup_ch
  v_col, is_g0 = (v_wg * n_vsub + v_sub) * 16 + lane_c, lane_g.eq(0)
  z0, o1 = UOp.const(0.0, dtypes.float32), UOp.const(1.0, dtypes.float32)
  c_lo, c_hi = UOp.const(0x05040100, dtypes.uint32), UOp.const(0x07060302, dtypes.uint32)
  def _sel(r4:UOp, r3:UOp) -> UOp: return r4 if rdna4 is True else (r3 if rdna4 is False else UOp(Ops.WHERE, src=(rdna4, r4, r3)))
  def swz(row:UOp, ch8:UOp) -> UOp: return ch8 ^ (row & 7)
  def row_of(e:int|UOp) -> UOp: return _sel(lane_g * 8 + e, e * 2 + lane_g)

  n_skq = is_kda.where(3, 2)
  skq = UOp.alloc((n_skq * sup_tok * D,), dtypes.half, addrspace=AddrSpace.LOCAL)
  skt = UOp.alloc((sup_ch * D * 16,), dtypes.half, addrspace=AddrSpace.LOCAL)
  s_T, s_P = UOp.alloc((sup_ch * 256,), dtypes.half, addrspace=AddrSpace.LOCAL), UOp.alloc((sup_ch * 256,), dtypes.half, addrspace=AddrSpace.LOCAL)
  s_M = UOp.alloc((sup_ch * 256 + n_vsub * 3 * 2 * 32 * 4,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  s_dd, s_b = UOp.alloc((n_vsub * 128,), dtypes.float32, addrspace=AddrSpace.LOCAL), UOp.alloc((sup_tok,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  s_gvec = UOp.alloc((3, sup_ch, D), dtypes.float32, addrspace=AddrSpace.LOCAL)
  s_L, s_g = UOp.alloc((sup_tok,), dtypes.float32, addrspace=AddrSpace.LOCAL), UOp.alloc((sup_tok,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  s_reg = UOp.alloc((kt_w, 8), dtypes.float32, addrspace=AddrSpace.REG)
  buf_a, buf_b = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
  buf_h8, buf_h16 = UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG), UOp.alloc((16,), dtypes.half, addrspace=AddrSpace.REG)

  def pk_h8_st(dst:UOp, dep:tuple[UOp, ...], fn, rid:int) -> UOp:
    p = UOp.range(4, rid)
    u = fn(p).reshape((1,)).bitcast(dtypes.half)
    return UOp.group(dst.after(*dep)[2 * p].store(u[0]), dst.after(*dep)[2 * p + 1].store(u[1])).end(p)
  def f8_to_frag_sts(fn_e, dep:tuple[UOp, ...], rid:int, d8:UOp=buf_h8, d16:UOp=buf_h16) -> tuple[UOp, UOp, UOp, UOp]:
    st8 = pk_h8_st(d8, dep, lambda p: _cvt_pk_f16(fn_e(2 * p), fn_e(2 * p + 1), inline=False), rid)
    p = UOp.range(4, rid + 400)
    my_u = _cvt_pk_f16(fn_e(2 * p), fn_e(2 * p + 1), inline=False)
    peer_u = _lane_xor(my_u, 16)
    u_e, u_o = is_g0.where(my_u, peer_u), is_g0.where(peer_u, my_u)
    h_lo, h_hi = _amd_byte_perm(u_o, u_e, c_lo).reshape((1,)).bitcast(dtypes.half), _amd_byte_perm(u_o, u_e, c_hi).reshape((1,)).bitcast(dtypes.half)
    st16 = UOp.group(d16.after(*dep)[4 * p].store(h_lo[0]), d16.after(*dep)[4 * p + 1].store(h_lo[1]),
                     d16.after(*dep)[4 * p + 2].store(h_hi[0]), d16.after(*dep)[4 * p + 3].store(h_hi[1])).end(p)
    return st8, d8.after(st8), st16, d16.after(st16)
  def f8_to_frag_fn(fn_e, dep:tuple[UOp, ...], rid:int, d8:UOp=buf_h8, d16:UOp=buf_h16) -> tuple[UOp, UOp]:
    _, f8, _, f16 = f8_to_frag_sts(fn_e, dep, rid, d8, d16)
    return f8, f16
  def f8_to_frag(f_slice:UOp, dep:UOp, rid:int, d8:UOp=buf_h8, d16:UOp=buf_h16) -> tuple[UOp, UOp]:
    return f8_to_frag_fn(lambda e: f_slice[e].load(), (dep,), rid, d8, d16)
  def pk_xor1(src:UOp, dep:UOp, rid:int) -> UOp:
    pk, e = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(8, rid)
    return pk.after(pk.after(dep)[e].store(_cvt_pk_f16(src[e].load(), _lane_xor(src[e].load(), 1), inline=False)).end(e))
  def lds_frag(arr:UOp, *idx) -> tuple[UOp, UOp]: return arr[(*idx, lane_g)], arr[(*idx, 0)].cat(arr[(*idx, 1)])
  def wmma_st(dst:UOp, a:tuple[UOp, UOp], b:tuple[UOp, UOp], acc_in:UOp) -> UOp:
    return _sel(dst.store(UOp.wmma(a[0], b[0], acc_in, *WMMA_ARG)), dst.store(UOp.wmma(a[1], b[1], acc_in, *WMMA_ARG)))

  j_z, e_z = UOp.range(kt_w, 8), UOp.range(8, 308, AxisType.UPCAST)
  r_ld, j_i = UOp.range(start_pos.ne(0).cast(dtypes.weakint), 9), UOp.range(kt_w, 10)
  s_zeroed, k0_s = s_reg.after(s_reg[j_z, e_z].store(z0).end(e_z, j_z)), (kw * kt_w + j_i) * 16 + lane_g * 8
  sv0, sv1 = _amd_load(state_f[bh, v_col, k0_s], 4), _amd_load(state_f[bh, v_col, k0_s + 4], 4)
  s_ld_r4 = s_reg.after(s_zeroed, r_ld)[j_i].store(sv0.cat(sv1))
  sv_r, i_s = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG), UOp.range(4, 11)
  sv_r = sv_r.after(sv_r.after(s_zeroed, r_ld).store(sv0.cat(sv1)))
  s_ld_r3 = UOp.group(
    s_reg.after(s_zeroed, r_ld)[j_i, i_s].store(is_g0.where(sv_r[2 * i_s].load(), _lane_xor(sv_r[2 * i_s + 1].load(), 16))),
    s_reg.after(s_zeroed, r_ld)[j_i, i_s + 4].store(is_g0.where(_lane_xor(sv_r[2 * i_s].load(), 16), sv_r[2 * i_s + 1].load()))).end(i_s)
  s_cur_init = s_reg.after(s_zeroed, _sel(s_ld_r4, s_ld_r3).end(j_i, r_ld))

  sup_r = UOp.range((tokens + sup_tok - 1) // sup_tok, 15)
  t_base, s_cur_super = sup_r * sup_tok, s_cur_init.after(sup_r)
  n_ch, n_ch8 = ((tokens - t_base + 15) // 16).minimum(sup_ch), D // 8

  dk4_cnt, total_pairs = D // 4, (sup_tok // 2) * (D // 4)
  it_kq = UOp.range((total_pairs + threads - 1) // threads, 21)
  pid = tid + it_kq * threads
  p_ok, pid_s = pid < total_pairs, pid.minimum(total_pairs - 1)
  rp_all, dk4 = pid_s // dk4_cnt, pid_s % dk4_cnt
  c_loc_s, rp_loc, dk, ta_l, tb_l = rp_all // 8, rp_all % 8, dk4 * 4, rp_all * 2, rp_all * 2 + 1
  ta_g, tb_g = t_base + ta_l, t_base + tb_l
  def ld_pk(src:UOp, tg:UOp) -> tuple[UOp, UOp]:
    r, vl = _amd_load(src[bh, tg.minimum(tokens - 1), dk], 4), tg < tokens
    return _cvt_pk_f16(vl.where(r[0], z0), vl.where(r[1], z0), inline=False), _cvt_pk_f16(vl.where(r[2], z0), vl.where(r[3], z0), inline=False)
  ka, kb, qa, qb = ld_pk(k_f, ta_g), ld_pk(k_f, tb_g), ld_pk(q_f, ta_g), ld_pk(q_f, tb_g)
  skq_u32 = skq.after(sup_r).bitcast(dtypes.uint32).reshape(n_skq, sup_tok, n_ch8, 4)
  skq_p = skq_u32.reshape(n_skq, sup_tok, n_ch8, 2, 2)
  skt_u32 = skt.after(sup_r).bitcast(dtypes.uint32).reshape(sup_ch, D, 8)
  ch8, sub2, rp_s = dk4 // 2, dk4 % 2, rp_loc ^ ((dk4 & 1) * 4)
  ca, cb, i_kt = swz(ta_l, ch8), swz(tb_l, ch8), UOp.range(2, 213)
  ka_i, kb_i = i_kt.eq(0).where(ka[0], ka[1]), i_kt.eq(0).where(kb[0], kb[1])
  kq_st = UOp.group(
    *(skq_p[m, tl.valid(p_ok), c, sub2].store(UOp.stack(p[0], p[1]))
      for m, tl, c, p in ((0, ta_l, ca, ka), (0, tb_l, cb, kb), (1, ta_l, ca, qa), (1, tb_l, cb, qb))),
    UOp.group(skt_u32[c_loc_s.valid(p_ok), dk + 2 * i_kt, rp_s].store(_amd_byte_perm(kb_i, ka_i, c_lo)),
              skt_u32[c_loc_s.valid(p_ok), dk + 2 * i_kt + 1, rp_s].store(_amd_byte_perm(kb_i, ka_i, c_hi))).end(i_kt),
  ).end(it_kq)
  t_ab_l = w_ch * 16 + lane_c
  t_ab_g = t_base + t_ab_l
  v_ab = t_ab_g < tokens
  a_val = v_ab.where(alpha_f[bh, t_ab_g.minimum(tokens - 1), 0].load(), o1)
  b_val = v_ab.where(beta_f[bh, t_ab_g.minimum(tokens - 1)].load(), z0)
  la_reg, sh_i = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), UOp.range(4, 214)
  la_0 = la_reg.after(la_reg.after(sup_r)[0].store(_fast_log2(a_val.maximum(UOp.const(1e-30, dtypes.float32)))))
  d_sh = UOp.const(1, dtypes.int32) << sh_i.cast(dtypes.int32)
  lc32, lv = lane_c.cast(dtypes.int32), la_0.after(sh_i)[0].load()
  la_fin = la_reg.after(la_reg[0].store(
    (lc32 >= d_sh).where(lv + _ds_bpermute((lc32 - d_sh).maximum(0) << 2, lv, inline=False), lv)).end(sh_i))[0].load()
  ab_en = is_g0 & w_role.eq(0)
  b1_gdn = UOp.group(kq_st, s_L.after(sup_r)[t_ab_l.valid(ab_en)].store(la_fin), s_g.after(sup_r)[t_ab_l.valid(ab_en)].store(_fast_exp2(la_fin)),
                     s_b.after(sup_r)[t_ab_l.valid(ab_en)].store(b_val))

  tot_k = sup_ch * D
  it_k = UOp.range((tot_k + threads - 1) // threads, 21)
  kid = tid + it_k * threads
  k_ok, kid_s = kid < tot_k, kid.minimum(tot_k - 1)
  c_loc_k, k_col = kid_s // D, kid_s % D
  L_buf, l_acc, t_r = UOp.alloc((16,), dtypes.float32, addrspace=AddrSpace.REG), _reg((1,), 0.0, sup_r, it_k), UOp.range(16, 212)
  tg = t_base + c_loc_k * 16 + t_r
  a_t = (tg < tokens).where(_amd_load(alpha_f[bh, tg.minimum(tokens - 1), k_col % alpha_k]), o1)
  l_nxt = l_acc.after(t_r)[0].load() + _fast_log2(a_t.maximum(UOp.const(1e-30, dtypes.float32)))
  L_r = L_buf.after(UOp.group(l_acc[0].store(l_nxt), L_buf.after(sup_r, it_k)[t_r].store(l_nxt)).end(t_r))
  L_ref, L_last = L_r[7].load(), L_r[15].load()
  gv_st = UOp.group(*(s_gvec.after(sup_r)[idx, c_loc_k.valid(k_ok), k_col].store(_fast_exp2(val))
                      for idx, val in ((0, L_ref), (1, L_last - L_ref), (2, L_last))))
  ch8_k, slot4, w_en, rp = k_col // 8, (k_col % 8) // 2, k_ok & (k_col & 1).eq(0), UOp.range(8, 211)
  c20, cn20 = UOp.const(20.0, dtypes.float32), UOp.const(-20.0, dtypes.float32)
  def pk_pair(v_me:UOp) -> UOp: return _cvt_pk_f16(v_me, _lane_xor(v_me, 1), inline=False)
  rp_sk, ki_ab, ab = rp ^ (((k_col >> 2) & 1) * 4), UOp.alloc((2,), dtypes.float32, addrspace=AddrSpace.REG), UOp.range(2, 215)
  t_lk = c_loc_k * 16 + rp * 2 + ab
  t_gk = t_base + t_lk
  vl_k = t_gk < tokens
  ks_v, qs_v = _amd_load(k_f[bh, t_gk.minimum(tokens - 1), k_col]), _amd_load(q_f[bh, t_gk.minimum(tokens - 1), k_col])
  l_d = (L_r[rp * 2 + ab].load() - L_ref).maximum(cn20).minimum(c20)
  up_v, ki_v, c_k = _fast_exp2(l_d), vl_k.where(ks_v * _fast_exp2(-l_d), z0), swz(t_lk, ch8_k)
  ab_done = UOp.group(
    skq_u32[0, t_lk.valid(w_en), c_k, slot4].store(pk_pair(vl_k.where(ks_v * up_v, z0))),
    skq_u32[1, t_lk.valid(w_en), c_k, slot4].store(pk_pair(vl_k.where(qs_v * up_v, z0))),
    skq_u32[2, t_lk.valid(w_en), c_k, slot4].store(pk_pair(ki_v)),
    ki_ab.after(sup_r, it_k, rp)[ab].store(ki_v),
  ).end(ab)
  ki_r = ki_ab.after(ab_done)
  rp_st = UOp.group(ab_done, skt_u32[c_loc_k.valid(k_ok), k_col, rp_sk].store(_cvt_pk_f16(ki_r[0].load(), ki_r[1].load(), inline=False))).end(rp)
  b1_kda = UOp.group(UOp.group(gv_st, rp_st).end(it_k), s_b.after(sup_r)[t_ab_l.valid(ab_en)].store(b_val))
  b1 = UOp(Ops.WHERE, src=(is_kda, b1_kda, b1_gdn)).barrier()

  skq_8, skt_8, sb_r = skq.after(b1).reshape(n_skq, sup_ch, 16, n_ch8, 8), skt.after(b1).reshape(sup_ch, D, 2, 8), s_b.after(b1).reshape(sup_ch, 16)
  sL_r, sg_r, sgv_r = s_L.after(b1).reshape(sup_ch, 16), s_g.after(b1).reshape(sup_ch, 16), s_gvec.after(b1)
  def skq_frag(m_idx:UOp|int, c_idx:UOp, kt_idx:UOp) -> tuple[UOp, UOp]:
    return (skq_8[m_idx, c_idx, lane_c, swz(lane_c, kt_idx * 2 + lane_g)],
            skq_8[m_idx, c_idx, lane_c, swz(lane_c, kt_idx * 2)].cat(skq_8[m_idx, c_idx, lane_c, swz(lane_c, kt_idx * 2 + 1)]))

  k_rhs_idx, sM_1d, sP_u32, has_2roles = is_kda.where(2, 0), s_M.after(b1), s_P.after(b1).bitcast(dtypes.uint32), waves >= (sup_ch * 2)
  ga0, gb0, jg = buf_a.after(b1).store(buf_a.const_like(0.0)), buf_b.after(b1).store(buf_b.const_like(0.0)), UOp.range(k_tiles, 22)
  kf_rhs = skq_frag(k_rhs_idx, w_ch, jg)
  gram_lhs = skq_frag(has_2roles.where(w_role.minimum(1), UOp.const(0, dtypes.weakint)), w_ch, jg)
  gram_a = wmma_st(buf_a, gram_lhs, kf_rhs, buf_a.after(ga0, jg))
  r_g1 = UOp.range(has_2roles.logical_not().cast(dtypes.weakint), 220)
  gram_done = UOp.group(gram_a, wmma_st(buf_b, skq_frag(1, w_ch, jg), kf_rhs, buf_b.after(gb0, jg, r_g1)).end(r_g1)).end(jg)
  gva, gvb, pv_reg, e_g = buf_a.after(gram_done), buf_b.after(gram_done), UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG), UOp.range(8, 221)
  tr_g = row_of(e_g)
  g_rat = is_kda.where(o1, (lane_c <= tr_g).where(_fast_exp2((sL_r[w_ch, tr_g].load() - sL_r[w_ch, lane_c].load()).minimum(z0)), z0))
  a_st = sM_1d[((w_ch * 16 + tr_g) * 16 + lane_c).valid(w_role.eq(0))].store(
    (lane_c < tr_g).where(sb_r[w_ch, tr_g].load() * g_rat * gva[e_g].load(), z0))
  pv_st = pv_reg.after(gram_done)[e_g].store((lane_c <= tr_g).where(g_rat * has_2roles.where(gva[e_g].load(), gvb[e_g].load()), z0))
  ag_done = UOp.group(a_st, pv_st).end(e_g)
  pk_r, p_en, e_p = pk_xor1(pv_reg.after(ag_done), ag_done, 222), (lane_c & 1).eq(0) & has_2roles.where(w_role.eq(1), w_role.eq(0)), UOp.range(8, 223)
  b2 = UOp.group(ag_done, sP_u32[((w_ch * 16 + row_of(e_p)) * 8 + (lane_c // 2)).valid(p_en)].store(pk_r[e_p].load()).end(e_p)).barrier()

  sA_1d, sT_u32, r_fsub1 = s_M.after(b2), s_T.after(b1).bitcast(dtypes.uint32), UOp.range(w_role.eq(0).cast(dtypes.weakint), 23)
  is_hi8, c8 = lane_c >= 8, lane_c & 7
  rb8, tc8 = is_hi8.where(UOp.const(8, dtypes.weakint), UOp.const(0, dtypes.weakint)), UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
  tc8 = tc8.after(tc8.after(b2, r_fsub1).store(tc8.const_like(0.0)))
  r_fs, p_fs = UOp.range(8, 231), UOp.range(8, 232)
  v_fs = _reg((1,), 0.0, tc8, r_fs)
  v_fs_d = v_fs[0].store(v_fs.after(p_fs)[0].load() + (p_fs < r_fs).where(
    sA_1d[(w_ch * 16 + rb8 + r_fs) * 16 + rb8 + p_fs].load() * tc8.after(r_fs)[p_fs].load(), z0)).end(p_fs)
  tc8_fin = tc8.after(tc8[r_fs].store(c8.eq(r_fs).where(o1, z0) - v_fs.after(v_fs_d)[0].load()).end(r_fs))
  in_diag8, e_d = (is_g0 & (lane_c < 8)) | (lane_g.eq(1) & is_hi8), UOp.range(8, 233)
  t8_r = buf_b.after(buf_b.after(tc8_fin)[e_d].store(in_diag8.where(tc8_fin[e_d].load(), z0)).end(e_d))
  t8_pk_r, e_t8 = pk_xor1(t8_r, t8_r, 234), UOp.range(8, 235)
  t8_st = sT_u32[((w_ch * 16 + lane_g * 8 + e_t8) * 8 + (lane_c // 2)).valid((lane_c & 1).eq(0))].store(t8_pk_r[e_t8].load()).end(e_t8)
  t8_h8 = buf_h8.after(pk_h8_st(buf_h8, (t8_r,), lambda p: _cvt_pk_f16(t8_r[2 * p].load(), t8_r[2 * p + 1].load(), inline=False), 236))
  a10_en, a10_b8 = is_hi8 & is_g0, UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG)
  a10_r4 = a10_b8.after(pk_h8_st(a10_b8, (t8_h8,), lambda p: _cvt_pk_f16(
    a10_en.where(-sA_1d[(w_ch * 16 + lane_c) * 16 + 2 * p].load(), z0),
    a10_en.where(-sA_1d[(w_ch * 16 + lane_c) * 16 + 2 * p + 1].load(), z0), inline=False), 237))
  h0 = UOp.const(0.0, dtypes.half)
  def pk_r3_lo(dst:UOp, dep:UOp, fn, rid:int) -> UOp:
    p, i_hi = UOp.range(4, rid), UOp.range(8, rid + 1)
    u = fn(p).reshape((1,)).bitcast(dtypes.half)
    return dst.after(UOp.group(UOp.group(dst.after(dep)[2 * p].store(u[0]), dst.after(dep)[2 * p + 1].store(u[1])).end(p),
                               dst.after(dep)[8 + i_hi].store(h0).end(i_hi)))
  t8_h16 = pk_r3_lo(buf_h16, tc8_fin, lambda p: _cvt_pk_f16(
    (lane_c < 8).where(tc8_fin[2 * p].load(), z0), (lane_c < 8).where(tc8_fin[2 * p + 1].load(), z0), inline=False), 338)
  a10_r3 = pk_r3_lo(UOp.alloc((16,), dtypes.half, addrspace=AddrSpace.REG), t8_h16, lambda p: _cvt_pk_f16(
    is_hi8.where(-sA_1d[(w_ch * 16 + lane_c) * 16 + 2 * p].load(), z0),
    is_hi8.where(-sA_1d[(w_ch * 16 + lane_c) * 16 + 2 * p + 1].load(), z0), inline=False), 340)
  b_done = wmma_st(buf_a, (a10_r4, a10_r3), (t8_h8, t8_h16),
                   buf_a.after(buf_a.after(_sel(a10_r4.src[1], a10_r3.src[1])).store(buf_a.const_like(0.0))))
  bh8_buf, bh16_buf = UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG), UOp.alloc((16,), dtypes.half, addrspace=AddrSpace.REG)
  bh8_st, _, bh16_st, _ = f8_to_frag_sts(lambda e: buf_a.after(b_done)[e].load(), (b_done,), 242, bh8_buf, bh16_buf)
  b_t8, r_fsub2 = UOp.group(t8_st, _sel(bh8_st, bh16_st)).end(r_fsub1).barrier(), UOp.range(w_role.eq(0).cast(dtypes.weakint), 29)
  sT_mid = s_T.after(b_t8, r_fsub2).reshape(sup_ch, 16, 2, 8)
  t10_init = buf_a.after(buf_a.after(b_t8, r_fsub2).store(buf_a.const_like(0.0)))
  t10_done = wmma_st(buf_a, lds_frag(sT_mid, w_ch, lane_c), (bh8_buf.after(b_t8, r_fsub2), bh16_buf.after(b_t8, r_fsub2)), t10_init)
  t10_pk_r, sT_u32_2 = pk_xor1(buf_a.after(t10_done), t10_done, 241), s_T.after(t10_done).bitcast(dtypes.uint32)
  t10_r4 = UOp.group(*(sT_u32_2[((w_ch * 16 + 8 + e) * 8 + lane_c // 2).valid(
    lane_g.eq(1) & (lane_c < 8) & (lane_c & 1).eq(0))].store(t10_pk_r[e].load()) for e in range(8)))
  t10_r3 = UOp.group(*(sT_u32_2[((w_ch * 16 + e * 2 + lane_g) * 8 + lane_c // 2).valid(
    (lane_c < 8) & (lane_c & 1).eq(0))].store(t10_pk_r[e].load()) for e in range(4, 8)))
  b3 = _sel(t10_r4, t10_r3).end(r_fsub2).barrier()
  sT_8, sP_8 = s_T.after(b3).reshape(sup_ch, 16, 2, 8), s_P.after(b3).reshape(sup_ch, 16, 2, 8)

  ch = UOp.range(n_ch, 24)
  t0_g, s_cur = t_base + ch * 16, s_cur_super.after(b3, ch)
  v_tmp, r_vp, e_vp = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG), UOp.range(kw.eq(0).cast(dtypes.weakint), 31), UOp.range(8, 243)
  tg_v = t0_g + row_of(e_vp)
  v_tmp_r = v_tmp.after(v_tmp.after(b3, ch, r_vp)[e_vp].store(
    (tg_v < tokens).where(_amd_load(v_f[bh, tg_v.minimum(tokens - 1), v_col]), z0)).end(e_vp, r_vp))
  x_init, qs_init, j_s = buf_a.after(b3, ch).store(buf_a.const_like(0.0)), buf_b.after(b3, ch).store(buf_b.const_like(0.0)), UOp.range(kt_w, 25)
  sb_h = f8_to_frag_fn(lambda e: s_cur[j_s, e].load() * is_kda.where(sgv_r[0, ch, (kw * kt_w + j_s) * 16 + row_of(e)].load(), o1), (j_s,), 244)
  xq_done = UOp.group(wmma_st(buf_a, skq_frag(0, ch, kw * kt_w + j_s), sb_h, buf_a.after(x_init, j_s)),
                      wmma_st(buf_b, skq_frag(1, ch, kw * kt_w + j_s), sb_h, buf_b.after(qs_init, j_s))).end(j_s)
  x_p, qs_p, s_xq_1d, n_peers = buf_a.after(xq_done), buf_b.after(xq_done), s_M.after(b3, ch), k_split - 1
  xq_b = UOp.group(*(s_xq_1d[((((v_sub * n_peers + (kw - 1).maximum(0)) * 2 + m) * 32 + lane) * 4 + p).valid(kw.ne(0))].store(
    _cvt_pk_f16(src[2 * p].load(), src[2 * p + 1].load(), inline=False).bitcast(dtypes.float32))
    for m, src in ((0, x_p), (1, qs_p)) for p in range(4))).barrier()
  s_xq_h, r_k0 = s_M.after(xq_b).bitcast(dtypes.half), UOp.range(kw.eq(0).cast(dtypes.weakint), 27)
  pk_r, e_rd = UOp.range(n_peers, 270), UOp.range(8, 271)
  red_p = UOp.group(*(buf[e_rd].store(
    buf.after(xq_b, r_k0, pk_r)[e_rd].load() + s_xq_h[(((v_sub * n_peers + pk_r) * 2 + m) * 32 + lane) * 8 + e_rd].load().float())
    for m, buf in ((0, buf_a), (1, buf_b)))).end(e_rd, pk_r)
  x_tot, qs_tot = buf_a.after(red_p), buf_b.after(red_p)
  y_ready = f8_to_frag_fn(lambda e: sb_r[ch, row_of(e)].load() * (v_tmp_r.after(xq_b, r_k0)[e].load() -
    is_kda.where(o1, sg_r[ch, row_of(e)].load()) * x_tot[e].load()), (red_p,), 246)
  d_done = wmma_st(buf_a, lds_frag(sT_8, ch, lane_c), y_ready,
                   buf_a.after(buf_a.after(_sel(y_ready[0].src[1], y_ready[1].src[1])).store(buf_a.const_like(0.0))))
  d_v, e_os = buf_a.after(d_done), UOp.range(8, 247)
  o_scale = buf_b.after(d_done)[e_os].store(is_kda.where(o1, sg_r[ch, row_of(e_os)].load()) * qs_tot[e_os].load()).end(e_os)
  def dd_of(e:UOp) -> UOp:
    return is_kda.where(o1, _fast_exp2((sL_r[ch, 15].load() - sL_r[ch, row_of(e)].load()).minimum(z0))) * d_v[e].load()
  p_dd = UOp.range(4, 248)
  dd_r4 = s_dd.after(o_scale)[(v_sub * 32 + lane) * 4 + p_dd].store(
    _cvt_pk_f16(dd_of(2 * p_dd), dd_of(2 * p_dd + 1), inline=False).bitcast(dtypes.float32)).end(p_dd)
  p_dd3 = UOp.range(4, 258)
  my_u = _cvt_pk_f16(dd_of(2 * p_dd3), dd_of(2 * p_dd3 + 1), inline=False)
  peer_u, en = _lane_xor(my_u, 16), (p_dd3 < 2).where(is_g0, lane_g.eq(1))
  u_e, u_o = is_g0.where(my_u, peer_u), is_g0.where(peer_u, my_u)
  dd_r3 = UOp.group(*(s_dd.after(o_scale)[((v_sub * 16 + lane_c) * 8 + 2 * p_dd3 + m).valid(en)].store(
    _amd_byte_perm(u_o, u_e, c_hi if m else c_lo).bitcast(dtypes.float32)) for m in (0, 1))).end(p_dd3)
  o_v = buf_b.after(wmma_st(buf_b, lds_frag(sP_8, ch, lane_c), f8_to_frag(d_v, o_scale, 249), buf_b.after(o_scale)))
  e_co = UOp.range(8, 250)
  b_dd = UOp.group(_sel(dd_r4, dd_r3), core_f[bh, (t0_g + row_of(e_co)).valid((t0_g + row_of(e_co)) < tokens), v_col].store(
    o_v[e_co].load()).end(e_co)).end(r_k0).barrier()
  dd_frag = (s_dd.after(b_dd).bitcast(dtypes.half).reshape(-1, 8)[v_sub * 32 + lane],
             s_dd.after(b_dd).bitcast(dtypes.half).reshape(n_vsub, 16, 16)[v_sub, lane_c])

  j_u = UOp.range(kt_w, 28)
  k_ubase = (kw * kt_w + j_u) * 16
  k_row_u = k_ubase + lane_c
  k_col_r = (k_row_u >> 2) & 1
  skt_u = (skt_8[ch, k_row_u, lane_g ^ k_col_r], skt_8[ch, k_row_u, k_col_r].cat(skt_8[ch, k_row_u, 1 ^ k_col_r]))
  e_su = UOp.range(8, 251)
  du_v = buf_a.after(wmma_st(buf_a, skt_u, dd_frag, buf_a.after(buf_a.after(b_dd, j_u).store(buf_a.const_like(0.0)))))
  s_u_kda = s_reg[j_u, e_su].store(s_cur.after(b_dd, j_u)[j_u, e_su].load() * sgv_r[2, ch, k_ubase + row_of(e_su)].load() +
                                   du_v[e_su].load() * sgv_r[1, ch, k_ubase + row_of(e_su)].load()).end(e_su, j_u)
  s_sc = s_reg[j_u, e_su].store(s_cur.after(b_dd, j_u)[j_u, e_su].load() * sg_r[ch, 15].load()).end(e_su)
  s_u_gdn = wmma_st(s_reg[j_u], skt_u, dd_frag, s_reg.after(s_sc)[j_u]).end(j_u)
  s_u_done = UOp(Ops.WHERE, src=(is_kda, s_u_kda, s_u_gdn))

  s_fin, j_o = s_reg.after(s_u_done.barrier().end(ch).end(sup_r)), UOp.range(kt_w, 30)
  k0_o, e_so, i_so = (kw * kt_w + j_o) * 16 + lane_g * 8, UOp.range(8, 252), UOp.range(4, 253)
  st_r4 = state_f[bh, v_col, k0_o + e_so].store(s_fin[j_o, e_so].load()).end(e_so)
  lo_v, hi_v = s_fin[j_o, i_so].load(), s_fin[j_o, 4 + i_so].load()
  lo_p, hi_p = _lane_xor(lo_v, 16), _lane_xor(hi_v, 16)
  st_r3 = UOp.group(state_f[bh, v_col, k0_o + 2 * i_so].store(is_g0.where(lo_v, hi_p)),
                    state_f[bh, v_col, k0_o + 2 * i_so + 1].store(is_g0.where(lo_p, hi_v))).end(i_so)
  return _sel(st_r4, st_r3).end(j_o, lane, wave, wg).sink(arg=KernelInfo(name="gated_delta_wmma", opts_to_apply=()))

@functools.cache
def _gated_delta_scalar_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, *, k_blks:UOp, alpha_k:UOp, start_pos:UOp,
                               rdna4:bool|UOp=False) -> UOp:
  batch, heads, tokens, value_dim = core.shape
  v_tiles, key_dim, BH = value_dim // 4, k_blks * 32, batch * heads
  core_f, v_f = core.reshape(BH, tokens, value_dim), v.reshape(BH, tokens, value_dim)
  q_f, k_f = q.reshape(BH, tokens, key_dim), k.reshape(BH, tokens, key_dim)
  beta_f, alpha_f, state_f = beta.reshape(BH, tokens), alpha.reshape(BH, tokens, alpha_k), state.reshape(BH, value_dim, key_dim)
  wg, lane = UOp.range(BH * v_tiles, 0, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  bh_sc, row_base, z0 = wg // v_tiles, (wg % v_tiles) * 4, UOp.const(0.0, dtypes.float32)
  ci_u, r_init = UOp.range(k_blks, 604, AxisType.UPCAST), UOp.range(4, 603, AxisType.UPCAST)
  tok_r, r_out = UOp.range(tokens, 602), UOp.range(4, 605, AxisType.UPCAST)
  col_sc, cur_sc = lane * k_blks + ci_u, UOp.alloc((4, k_blks), dtypes.float32, addrspace=AddrSpace.REG)
  cur_sc = cur_sc.after(cur_sc[r_init, ci_u].store(
    start_pos.eq(0).where(z0, state_f[bh_sc, row_base + r_init, col_sc].load().float())).end(ci_u, r_init))
  kv_s, qv_s = k_f[bh_sc, tok_r, col_sc].load(), q_f[bh_sc, tok_r, col_sc].load()
  bv_s, av_s = beta_f[bh_sc, tok_r].load(), alpha_f[bh_sc, tok_r, col_sc % alpha_k].load()
  kqv_s = warp_reduce((kv_s * qv_s).reduce(ci_u, arg=(Ops.ADD, 0)), full_wave=True, dpp=rdna4)
  sc_stores: list[UOp] = []
  for r_i in range(4):
    dec = cur_sc.after(tok_r)[r_i, ci_u].load() * av_s
    delta = (v_f[bh_sc, tok_r, row_base + r_i].load() - warp_reduce((dec * kv_s).reduce(ci_u, arg=(Ops.ADD, 0)), full_wave=True, dpp=rdna4)) * bv_s
    q_red = warp_reduce((dec * qv_s).reduce(ci_u, arg=(Ops.ADD, 0)), full_wave=True, dpp=rdna4)
    sc_stores += [core_f[bh_sc, tok_r, (row_base + r_i).valid(lane.eq(0))].store(q_red + delta * kqv_s),
                  cur_sc[r_i, ci_u].store(dec + delta * kv_s).end(ci_u)]
  return state_f[bh_sc, row_base + r_out, col_sc].store(cur_sc.after(UOp.group(*sc_stores).end(tok_r))[r_out, ci_u].load().cast(state.dtype)).end(
    ci_u, r_out, lane, wg).sink(arg=KernelInfo(name="gated_delta_scalar", opts_to_apply=()))

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, *,
                                k_blks:UOp, alpha_k:UOp, start_pos:UOp, arch:str|UOp="rdna4") -> UOp:
  _, _, tokens, value_dim = core.shape
  assert isinstance(value_dim, UOp)
  v_tiles = value_dim // 4
  rdna4: bool|UOp = arch.eq(12) if isinstance(arch, UOp) else (arch == "rdna4")
  k_sc = _gated_delta_scalar_kernel(core, q, k, v, beta, alpha, state, k_blks=k_blks, alpha_k=alpha_k, start_pos=start_pos, rdna4=rdna4)
  if arch == "amd": return k_sc
  can_wmma = (tokens >= 16) & (v_tiles % 8).eq(0) & (k_blks >= 2) & (k_blks % 2).eq(0) & (k_blks <= 8) & (alpha_k.eq(1) | alpha_k.eq(k_blks * 32))
  if isinstance(arch, UOp): can_wmma = (arch >= 11) & can_wmma
  k_wmma = _gated_delta_wmma_kernel(core, q, k, v, beta, alpha, state, k_blks=k_blks, alpha_k=alpha_k, start_pos=start_pos, rdna4=rdna4)
  return UOp(Ops.WHERE, src=(can_wmma, k_wmma, k_sc))

# ******** fused mHC coefficients (fuses flat.square().sum(-1) directly) ********

def _hc_coeffs_kernel(out:UOp, mix:UOp, flat:UOp, base:UOp, scale:UOp, count:int, iters:UOp, eps:UOp, norm_eps:UOp) -> UOp:
  assert count == 4, "the swizzles assume 4 streams"
  features, waves = flat.shape[1], 4
  assert isinstance(features, UOp)
  token, lane, wave = UOp.range(out.shape[0], 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), UOp.range(waves, 1, AxisType.LOCAL)
  tid, n_v4 = wave * 32 + lane, features // 4
  it_s, acc = UOp.range((n_v4 + waves * 32 - 1) // (waves * 32), 3), _reg((1,), 0.0)
  idx4 = tid + it_s * (waves * 32)
  v4 = _amd_load(flat[token, idx4.minimum(n_v4 - 1) * 4], 4)
  upd = acc[0].store(acc.after(it_s)[0].load() + (idx4 < n_v4).where(v4[0] * v4[0] + v4[1] * v4[1] + v4[2] * v4[2] + v4[3] * v4[3], 0.0)).end(it_s)
  lds = UOp.alloc((waves,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds_r, wk = lds.after(lds[wave.valid(lane.eq(0))].store(warp_reduce(acc.after(upd)[0].load(), full_wave=True)).barrier()), UOp.range(waves, 6)
  rms = (lds_r[wk].load().reduce(wk, arg=(Ops.ADD, 0)) * (1.0 / features.cast(dtypes.float32)) + norm_eps).rsqrt()
  is_comb = lane < 16
  idx, which = is_comb.where(lane + 8, (lane - 16).minimum(7).maximum(0)), is_comb.where(2, (lane < 20).where(0, 1))
  sc_v = which.eq(0).where(scale[0].load().float(), which.eq(1).where(scale[1].load().float(), scale[2].load().float()))
  v = mix[token, idx].load() * rms * sc_v + base[idx].load().float()
  sig = (1.0 + _fast_exp2(v * -LOG2E)).reciprocal()
  prepost = which.eq(0).where(sig + eps, sig * 2.0)
  def rowsum(x:UOp, op=lambda a, b: a + b) -> UOp: return op(y := op(x, _lane_xor(x, 1)), _lane_xor(y, 2))
  def colsum(x:UOp) -> UOp: return (y := x + _lane_xor(x, 4)) + _lane_xor(y, 8)
  e = _fast_exp2((v - rowsum(v, _fast_fmax)) * LOG2E)
  reg, it = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), UOp.range(2 * iters - 1, 4)
  reg = reg.after(reg[0].store(e * rowsum(e).reciprocal() + eps))
  cv = reg.after(it)[0].load()
  c = reg.after(reg[0].store(cv * ((it % 2).eq(0).where(colsum(cv), rowsum(cv)) + eps).reciprocal()).end(it))[0].load()
  return out[token, is_comb.where(lane + 8, lane - 16).valid(wave.eq(0) & (lane < 24))].store(is_comb.where(c, prepost)).end(lane, wave, token).sink(
    arg=KernelInfo(name="hc_coeffs", opts_to_apply=()))
