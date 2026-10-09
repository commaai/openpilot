from __future__ import annotations
import functools, math
from typing import Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve
from tinygrad.renderer.cstyle import HIPRenderer

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)
GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 10, 11, 12, 13, 14
IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS = 17, 18, 20, 21, 22, 23
QUANT_SIZES = {Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XS: 74,
               IQ3_XXS: 98, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136}  # bytes per 256 weights
HALFWORD_QUANTS = (Q6_K, Q2_K, Q3_K, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S)
QUANT_NAMES = {Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6", IQ4_XS: "iq4_xs",
               IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ4_NL: "iq4_nl", IQ3_S: "iq3_s", IQ2_S: "iq2_s"}

def _unbind(v:int|UOp) -> int|UOp: return v.unbind_all()[0] if isinstance(v, UOp) else v

@functools.cache
def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if getenv("DISABLE_AMD_KERNELS"): return False
  # Wave32 and WMMA kernels support RDNA3/4. CDNA's wave64/MFMA is not supported.
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  # @function contexts set ALLOW_DEVICE_USAGE=0 (scheduling must not open devices); the device is always open here
  with Context(ALLOW_DEVICE_USAGE=1):
    return (t:=getattr(Device[device], "target", None)) is not None and t[0] in (11, 12) and isinstance(Device[device].renderer, HIPRenderer)

@functools.cache
def _wmma_rdna4(device:str|tuple[str, ...]) -> bool:
  if isinstance(device, tuple): device = device[0]
  with Context(ALLOW_DEVICE_USAGE=1): return getattr(Device[device], "target")[0] == 12

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=
      (f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

def _reg(shape:tuple[int, ...], value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.alloc(shape, dtypes.float, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

# ******** quant linear: q8-activation kernels over packed ggml weights ********

class Linear(nn.Linear):
  ggml_type:int|None = None
  use_custom_quant = True
  shard_axis:int|None = None
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
  def set_quantized(self, decoded:Tensor):
    if self.in_features % GGML_BLOCK_SIZE: return
    packed_sizes = {typ: decoded.numel() // 256 * type_size for typ,type_size in QUANT_SIZES.items()}
    graph = decoded.uop.toposort()
    raw = next((u for u in graph if u.op in (Ops.SHRINK, Ops.BUFFER, Ops.UNSHARD) and u.dtype == dtypes.uint8 and
                prod(u.shape) in packed_sizes.values()), None)
    if raw is None: return
    # Only unwrap storage/order-preserving views, then require the exact dequantization expression.
    # This rejects subsequent arithmetic and permutations, including RoPE's concatenated query weights.
    # a permute keeps the order if the axes not of size 1 on a device stay sorted (gguf_shard moves the devices)
    def unwrapped(u:UOp) -> UOp:
      while u.op in (Ops.RESHAPE, Ops.STAGE) or (u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype)) or \
        (u.op is Ops.PERMUTE and (kept:=[a for a in u.marg if u.src[0].shard_shape[a] != 1]) == sorted(kept)):
        u = u.src[0]
      return u
    # Several formats have the same byte count (Q3_K/IQ3_S and Q4_K/IQ4_NL). Match the expression, not just the size.
    for ggml_type, size in packed_sizes.items():
      if size != prod(raw.shape): continue
      expected = ggml_data_to_tensor(Tensor(raw).flatten(), self.in_features * self.out_features, ggml_type)
      if unwrapped(decoded.uop).key == unwrapped(expected.uop).key: break
    else: return
    # Some blocks are only halfword-aligned; keep all formats as zero-copy views of the GGUF storage.
    word_dtype = dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32
    raw_offset = raw.contiguous_view_offset()
    if raw_offset is None or raw_offset % word_dtype.itemsize or raw.buf_uop.dtype != dtypes.uint8: return
    self.ggml_type = ggml_type
    self.shard_axis = decoded.uop.axis
    self.weight = Tensor(raw).flatten().bitcast(word_dtype).contiguous()
  def __call__(self, x:Tensor) -> Tensor:
    supported = self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized(self.weight)
      if self.ggml_type is None:
        # tiny dense fp16 matmul (e.g. the ssm beta/alpha head rows): single fp16 gemv kernel instead of a
        # generic matmul schedule, and realize the densely packed weight once if it is still a lazy ggml view
        if self.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and self.out_features <= 2048 \
          and self.in_features % (WARP_SIZE*4) == 0 and self.weight.uop.axis is None:
          numel, max_shape = x.numel(), x.max_shape
          if isinstance(numel, int) or prod(max_shape) // self.in_features <= 32:
            out = f16_gemv(self, x if isinstance(numel, int) else x.pad_to(max_shape))
            return out if isinstance(numel, int) else out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
        self.use_custom_quant = supported = False  # not a supported quant format
    if self.ggml_type in QUANT_SIZES and supported:
      if isinstance(x.numel(), int): return q8_linear(self, x)
      # symbolic token count: pad to the max chunk size so the kernels see static shapes, garbage rows are sliced off
      out = q8_linear(self, x.pad_to(x.max_shape))
      return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
    return super().__call__(x)

def _amd_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=(a, b, c), arg=("__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)", dtypes.int32))

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=tuple(x.cast(dtypes.uint32) for x in (a, b, selectors)), arg=("__builtin_amdgcn_perm({}, {}, {})", dtypes.uint32))

def _amd_load(ptr:UOp, lanes:int|None=None, stream:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  # nontemporal scalar load: streamed weights must not evict the activations/KV cache from L2
  if lanes is None: return ptr.load(arg="nontemporal")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(arg="nontemporal" if stream else None)

def _load_byte(raw:UOp, base:UOp, offset:int|UOp) -> UOp:
  size = raw.dtype.itemsize
  return (raw[base + offset//size].cast(dtypes.uint32) >> ((offset%size)*8)) & 255
def _load_u32(raw:UOp, base:UOp, offset:int|UOp, stream:bool=False) -> UOp:
  # Halfword-aligned formats cannot load u32 directly.
  lo, hi = (raw[base+offset//2+i] for i in range(2))
  if stream: lo, hi = _amd_load(lo), _amd_load(hi)
  return lo.cast(dtypes.uint32) | (hi.cast(dtypes.uint32) << 16)

def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

def _iq4_bytes(packed:UOp, shift:int|UOp) -> UOp:
  # the non-linear iq4nl table as a byte lookup: 3 byte_perms beat any arithmetic/select-tree form (~60% decode)
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _amd_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _amd_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _amd_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def _q5_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  # scales/mins (6-bit each) live in block bytes 4-15: three words total, same for the whole super-block's lanes
  w1, w2, w3 = _amd_load(raw[base+1]), _amd_load(raw[base+2]), _amd_load(raw[base+3])
  sb = (subgroup & 3) * 8  # byte within word
  byte1, byte2, byte3 = (w1 >> sb) & 255, (w2 >> sb) & 255, (w3 >> sb) & 255
  scale = (subgroup < 4).where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4))
  minimum = (subgroup < 4).where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4))
  d, dmin = (raw[base] & 0xffff).cast(dtypes.uint16), (raw[base] >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

def _iq4_scale(raw:UOp, base:UOp, subgroup:UOp) -> UOp:
  header, low = raw[base].load(), raw[base+1].load()
  scale = ((low >> (4*subgroup)) & 15) | (((header >> (16+2*subgroup)) & 3) << 4)
  return _half(header) * (scale.cast(dtypes.int32)-32).float()

def iq4_half_lut(device:str|tuple[str, ...]|None) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  return Tensor.const(tuple(x for j in range(16) for i in range(16) for x in (kvalues_iq4nl[i], kvalues_iq4nl[j])),
                      dtypes.float16).to(device, force=True).bitcast(dtypes.uint32)

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  value = x.reshape(tokens, groups, 32)[token, group, lane].load().float()
  # Quantize each input once, then pack four neighboring lanes into one word.
  # Keep divisions intact: reciprocal multiplication can move half-precision inputs across rounding ties.
  d = UOp(Ops.CUSTOM, src=(warp_reduce(value.abs(), maximum=True, full_wave=True),), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  rounded = UOp(Ops.CUSTOM, src=(value, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float))
  quant = rounded.clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= UOp(Ops.CUSTOM, src=(word,), arg=(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | offset<<10})", dtypes.uint32))
  stores = (q[token, group, (lane//4).valid((lane%4).eq(0))].store(word),
            scale[token, group.valid(lane.eq(0))].store(d),
            xsum[token, group, (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant.float())))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  groups = in_features//Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  xsum = Tensor.empty(tokens, groups, 2, dtype=dtypes.float32, device=x.device)
  q, scale, xsum = Tensor.custom_kernel(q, scale, xsum, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:3]
  return q, scale, xsum

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str) -> UOp:
  chunks = out.shape[2]
  # One wave per output/chunk; group neighboring rows to amortize workgroup scheduling.
  rows = math.gcd(out_features, 4)
  row = UOp.range(out.shape[0]*out_features//rows, 0, AxisType.GLOBAL)
  wave = UOp.range(rows, 3, AxisType.LOCAL)
  token_output = row*rows+wave
  chunk, lane = UOp.range(chunks, 1, AxisType.GLOBAL), UOp.range(32, 2, AxisType.LOCAL)
  token, output = token_output // out_features, token_output % out_features
  group = lane+chunk*32
  value = (group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
  total = warp_reduce(value, full_wave=True)
  return out[token, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)).end(row, wave, chunk, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

def _iq_grid(device:str|tuple[str, ...]|None, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common as ggml
  grid, words = {IQ2_XS: (ggml.iq2xs_grid, 2), IQ2_S: (ggml.iq2s_grid, 2),
                 IQ3_XXS: (ggml.iq3xxs_grid, 1), IQ3_S: (ggml.iq3s_grid, 1)}[ggml_type]
  return Tensor.const(tuple((v >> (32*i)) & 0xffffffff for v in grid for i in range(words)),
                      dtypes.uint32).to(device, force=True)

def _iq_even_signs(signs:UOp) -> UOp:
  parity = signs ^ (signs >> 4)
  parity ^= parity >> 2
  parity ^= parity >> 1
  return signs | ((parity & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  mask = sum(((signs >> i) & 1) * (255 << (8*i)) for i in range(4))
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  return (word ^ mask) + (mask & 0x01010101)

def _quant_word(raw:UOp, base:UOp, subgroup:UOp, i:int|UOp, ggml_type:int, grid:UOp|None) -> UOp:
  # Four packed weight bytes, shared by integer-dot decode and FP16 WMMA prefill.
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset, stream=ggml_type == Q6_K)
  if ggml_type in (Q4_K, Q5_K):
    offset = (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8
    weights = (_amd_load(raw[base+offset], 8)[i] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
    if ggml_type == Q5_K: weights |= ((_amd_load(raw[base+4], 8)[i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
    return weights
  if ggml_type == Q6_K:
    low = word((subgroup//4)*64 + (subgroup%2)*32 + i*4) >> ((subgroup%4//2)*4).cast(dtypes.uint32)
    high = word(128 + (subgroup//4)*32 + i*4) >> ((subgroup%4)*2).cast(dtypes.uint32)
    return (low & 0x0f0f0f0f) | ((high & 0x03030303) << 4)
  if ggml_type in (Q2_K, Q3_K):
    offset = (16 if ggml_type == Q2_K else 32) + (subgroup//4)*32 + i*4
    weights = (word(offset) >> ((subgroup%4)*2)) & 0x03030303
    if ggml_type == Q3_K: weights |= ((word(i*4) >> subgroup) & 0x01010101) << 2
    return weights
  if ggml_type in (IQ4_NL, IQ4_XS):
    packed = word(2+(i%4)*4) if ggml_type == IQ4_NL else _amd_load(raw[base+2+subgroup*4+i%4])
    return _iq4_bytes(packed, 4*(i//4))
  assert grid is not None
  grid_words = 2 if ggml_type in (IQ2_XS, IQ2_S) else 1
  if ggml_type in (IQ3_S, IQ3_XXS):
    index = byte(2+subgroup*8+i)
    if ggml_type == IQ3_S:
      index |= ((byte(66+subgroup) >> i) & 1) << 8
      signs = byte(74+subgroup*4+i//2)
    else: signs = _iq_even_signs((word(66+subgroup*4) >> (7*(i//2))) & 127)
  elif ggml_type == IQ2_XS:
    packed = raw[base+1+subgroup*4+i//2].cast(dtypes.uint32)
    index, signs = packed & 511, _iq_even_signs(packed >> 9)
  else:
    index = byte(2+subgroup*4+i//2) | (((byte(66+subgroup) >> (2*(i//2))) & 3) << 8)
    signs = byte(34+subgroup*4+i//2)
  return _iq_signed_word(grid[index*grid_words+i%grid_words], signs >> (4*(i%2)))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp,
                         out_features:int, in_features:int, ggml_type:int) -> UOp:
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group//8, group%8
    base = (output*in_features//256 + block) * (QUANT_SIZES[ggml_type]//raw.dtype.itemsize)
    if ggml_type == IQ4_NL: base = (output*in_features//32 + group)*9
    def byte(offset): return _load_byte(raw, base, offset)
    def word(offset): return _load_u32(raw, base, offset)
    xwords = _amd_load(xq[token, group, 0], 8)
    # One accumulator per scale group: 32 weights for Q4/Q5/IQ4_XS, two groups of 16 otherwise.
    dots = [UOp.const(0, dtypes.int32)] * (1 if ggml_type in (Q4_K, Q5_K, IQ4_XS) else 2)
    for i in range(8):
      weights = _quant_word(raw, base, subgroup, i, ggml_type, grids[0] if grids else None)
      acc = i//(8//len(dots))
      dots[acc] = _amd_dp4a(weights, xwords[i], dots[acc])
    if ggml_type in (Q4_K, Q5_K):
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      total = dots[0].float()*d*scale - (xs[token, group, 0].load()+xs[token, group, 1].load())*dmin*minimum
    elif ggml_type == IQ4_XS:
      total = dots[0].float() * _iq4_scale(raw, base, subgroup)
    elif ggml_type == Q6_K:
      # Subtract the quant offset via the activation sums instead of unpacking signed bytes.
      scales = [(raw[base+96+subgroup] >> (h*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for h in range(2)]
      total = sum((dots[h].float()-32*xs[token, group, h].load())*scales[h] for h in range(2))
      return total * xd[token, group] * _half(raw[base+104])
    elif ggml_type in (Q2_K, Q3_K):
      total = UOp.const(0, dtypes.float32)
      for half in range(2):
        j = subgroup*2+half
        if ggml_type == Q2_K:
          scale = byte(j)
          total += dots[half].float() * (scale & 15).float() * _half(raw[base+40]) - \
                   xs[token, group, half] * (scale >> 4).float() * _half(raw[base+41])
        else:
          scale = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
          total += (dots[half].float() - 4*xs[token, group, half]) * (scale.cast(dtypes.int32)-32).float() * _half(raw[base+54])
    elif ggml_type in (IQ2_XS, IQ2_S):
      scales = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
      total = sum(dots[h].float() * (((scales >> (h*4)) & 15).float()+0.5) for h in range(2)) * 0.25 * _half(raw[base])
    else:
      total = (dots[0]+dots[1]).float() * _half(raw[base])
      if ggml_type == IQ3_S: total *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      if ggml_type == IQ3_XXS: total *= ((word(66+subgroup*4) >> 28).float()+0.5)*0.5
    return total * xd[token, group]
  return _decode_linear(out, out_features, in_features//32, group_dot, "linear_"+QUANT_NAMES[ggml_type])

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int):
  if out_features % (16*output_tiles): output_tiles = 1
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  token_block, output_block = UOp.range(out.shape[0]//token_tile, 0), UOp.range(out_features//(16*output_tiles*output_waves), 1)
  # lane is a hardware WARP range (like the flash kernel): the fragment math stays visible without being
  # range-split into nested loops, which would scramble the WMMA fragment layout
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  col, half = lane % 16, lane // 16
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(token_block*token_tile + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + half*8 + i for i in range(8)) for tile in range(token_tile//16))
  return output_waves, token_block, output_block, lane, wave, half, outputs, inputs, tokens

def _wmma_stores(out, outputs, tokens, accs, update, half, lane, wave, output_waves, rdna4):
  # RDNA4 owns eight consecutive rows per half-wave, matching tokens directly.
  if rdna4:
    return [out[token, output].store(acc.after(update)[i].load()) for output,output_accs in zip(outputs, accs)
            for tile_tokens,acc in zip(tokens, output_accs) for i,token in enumerate(tile_tokens)]
  # the accumulator fragment halves are exchanged between lane pairs (l, l^16) through LDS (a ds_swizzle without CUSTOM)
  flat_accs = [acc for output_accs in accs for acc in output_accs]
  lds = UOp.alloc((output_waves, 32, len(flat_accs)*8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  stores = [lds[wave, lane, a*8+i].store(acc.after(update)[i].load()) for a,acc in enumerate(flat_accs) for i in range(8)]
  lds = lds.after(*stores)
  def values(ai:int) -> tuple[UOp, ...]:
    own = tuple(lds[wave, lane, ai*8+i].load() for i in range(8))
    peer = tuple(lds[wave, lane ^ 16, ai*8+i].load() for i in range(8))
    low = half.eq(0)
    return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
  tt = len(tokens)
  return [out[token, output].store(value) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
          for tile,(tile_tokens,_acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, values(ot*tt+tile))]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, rdna4):
  x = x.reshape(out.shape[0], in_features)
  output_waves, token_block, output_block, lane, wave, physical_half, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  # IQ4_NL blocks are 32 wide: a device can hold part of a 256 weight row group
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  accs = tuple(tuple(UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.LOOP)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  # gfx12 moves k2 from the element index to lane bit 4: each lane supplies two groups of four.
  ks = tuple(i%4 + physical_half*4 + (i//4)*8 for i in range(8)) if rdna4 else tuple(range(16))
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in ks))
                   for input_token in input_tokens)
    for output_tile,output in enumerate(outputs):
      bfrag = UOp.stack(*dequant(output*output_words + block*type_words, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, *WMMA_ARG)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)
  stores = _wmma_stores(out, outputs, tokens, accs, update, physical_half, lane, wave, output_waves, rdna4)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int, rdna4:bool=False) -> UOp:
  token_tile, output_tiles = (64, 1 if out_features <= 1024 else 2) if out.shape[0] % 64 == 0 else \
    (32 if out.shape[0] % 32 == 0 else 16, 2)
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  word_indices = (layout[5], layout[5]+2) if rdna4 else tuple(range(4))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
    words = tuple((raw[qs_base+i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      (((raw[base+4+half*4+i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if ggml_type == Q5_K else 0) for i in word_indices)
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//4,
                            layout, dequant, f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma", rdna4)

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int, rdna4:bool=False) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    out_features <= 6144 else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  output_waves, _, _, lane, wave, half, _, _, _ = layout
  word_indices = (half, half+2) if rdna4 else tuple(range(4))
  local_lut = UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*32+lane, 256//(32*output_waves)
  lut = local_lut.after(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    scale = _iq4_scale(raw, base, subgroup)
    pairs = tuple(lut[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                  for word in word_indices for byte in range(4))
    return tuple((_half((pair >> (half*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in pairs)
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[IQ4_XS]//4, layout, dequant, "linear_iq4_xs_f16_wmma", rdna4)

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp,
                                  out_features:int, in_features:int, ggml_type:int, rdna4:bool=False) -> UOp:
  token_tile = 64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16
  layout = _wmma_layout(out, out_features, token_tile, 2)
  output_waves, _, _, lane, wave, physical_half, _, _, _ = layout
  word_indices = (physical_half, physical_half+2) if rdna4 else tuple(range(4))
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*32+lane, output_waves*32
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    if ggml_type == IQ4_NL: base += subgroup*9  # eight independent 18-byte blocks per 256 weights
    def byte(offset): return _load_byte(raw, base, offset)
    zero, minimum = 0, UOp.const(0, dtypes.float)
    if ggml_type == Q2_K:
      sc = byte(subgroup*2+half)
      scale, minimum = _half(raw[base+40])*(sc & 15).float(), _half(raw[base+41])*(sc >> 4).float()
    elif ggml_type == Q3_K:
      j = subgroup*2+half
      sc = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
      scale, zero = _half(raw[base+54])*(sc.cast(dtypes.int32)-32).float(), 4
    elif ggml_type == Q6_K:
      scale, zero = _half(raw[base+104])*byte(192+subgroup*2+half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
    else:
      scale = _half(raw[base])
      if ggml_type in (IQ2_XS, IQ2_S):
        sc = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
        scale *= (((sc >> (half*4)) & 15).float()+0.5)*0.25
      elif ggml_type == IQ3_S: scale *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      elif ggml_type == IQ3_XXS: scale *= ((_load_u32(raw, base, 66+subgroup*4) >> 28).float()+0.5)*0.5
    words = tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in word_indices)
    return tuple((((word >> (b*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()-zero)*scale-minimum).cast(dtypes.half)
                 for word in words for b in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//raw.dtype.itemsize,
                            layout, dequant, f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", rdna4)

def q8_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in QUANT_SIZES
  tokens = int(x.numel()) // layer.in_features
  out_features, in_features = layer.out_features, int(x.uop.shard_shape[-1])
  splits = layer.in_features // in_features
  assert (layer.shard_axis == 1) == (splits > 1), f"input features split over {splits} devices, weight split on axis {layer.shard_axis}"
  out_shape:tuple[int, ...] = (splits*tokens, out_features)
  fxn:Callable[..., UOp]
  extra = (_iq_grid(x.device, layer.ggml_type),) if layer.ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  # the kernels write the output features of their own device
  local_out = out_features // len(dev) if isinstance(dev:=layer.weight.device, tuple) and layer.shard_axis == 0 else out_features
  if tokens % 16 == 0 and local_out % 16 == 0:
    if layer.ggml_type == IQ4_XS:
      fxn, extra = _iq4_linear_f16_wmma_kernel, (iq4_half_lut(x.device),)
    else:
      fxn = functools.partial(_q5_linear_f16_wmma_kernel if layer.ggml_type in (Q4_K, Q5_K) else _quant_linear_f16_wmma_kernel,
                              ggml_type=layer.ggml_type)
    fxn = functools.partial(fxn, rdna4=_wmma_rdna4(x.device))
    srcs = (x.cast(dtypes.float16).contiguous(), *extra)
  else:
    fxn = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
    srcs = (*q8_quantize(x, tokens, in_features), *extra)
    out_shape += ((in_features+1023)//1024,)
  out = Tensor.empty(out_shape, dtype=dtypes.float32, device=x.device, axis=None if layer.shard_axis is None else 1-layer.shard_axis)
  result = Tensor.custom_kernel(out, layer.weight, *srcs, fxn=functools.partial(fxn, out_features=local_out, in_features=in_features))[0]
  if len(result.shape) == 3: result = result.sum(-1)
  # row parallel: every device wrote the partial sum of its input features, the sum over the devices is the allreduce
  if splits > 1: result = result.reshape(splits, tokens, out_features).sum(0)
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias

# ******** tiny dense fp16 gemv ********

@functools.cache
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int) -> UOp:
  bias: UOp|None = rest[0] if rest else None
  # one block per (token, output row), 32 lanes accumulate 4-wide chunks of the row
  lanes, val_chunk = WARP_SIZE, 4
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL)
  per = in_features // (lanes * val_chunk)
  assert per * lanes * val_chunk == in_features
  w = w.reshape((out_features, per, lanes*val_chunk))
  x = x.reshape((tokens, per, lanes*val_chunk))
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    for j in range(val_chunk):
      acc = acc + w[out_row, i, lane*val_chunk + j].load().float() * x[token, i, lane*val_chunk + j].load().float()
  total = warp_reduce(acc, full_wave=True)
  if bias is not None: total = total + bias[out_row].load().float()
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

def _view_back(t:Tensor) -> Tensor:
  # Widening half to float is exact; preserve casts that round or change the values.
  uop = t.uop
  while uop.op is Ops.CAST and uop.dtype == dtypes.float32 and uop.src[0].dtype in (dtypes.half, dtypes.bfloat16): uop = uop.src[0]
  return Tensor(uop).reshape(t.shape)

def f16_gemv(layer:Linear, x:Tensor) -> Tensor:
  tokens = prod(x.shape[:-1])
  assert isinstance(tokens, int)
  weight = _view_back(layer.weight)
  x = x.contiguous()
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_amd_f16_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features, tokens=tokens)
  srcs = (out, weight.reshape(-1), x.reshape(tokens, layer.in_features)) + (() if layer.bias is None else (_view_back(layer.bias),))
  return Tensor.custom_kernel(*srcs, fxn=fxn)[0].reshape(*x.shape[:-1], layer.out_features)

# ******** flash attention on the KV cache ********

def _vec_load(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  if lanes == 1: return (ptr.load().float(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i].float() for i in range(lanes))

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, valid_kv_len, max_kv_len, block_n, waves=4):
  valid_kv_len = _unbind(valid_kv_len)
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DPL, WAVES, PARTIALS = H // H_KV, block_n, D // WARP_SIZE, waves, out.shape[2]
  assert CHUNK % WAVES == 0
  SEC = CHUNK // WAVES  # keys each wave scans independently
  total_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  live_chunks = min(total_chunks, PARTIALS) if isinstance(total_chunks, int) else total_chunks.minimum(PARTIALS)
  block_bhkv, block_chunk = UOp.range(B*H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  # per-lane query fragments for every GQA head, kept packed in registers; unpacked at use
  qf = tuple(_vec_load(q[b, kv_head*G+h, 0, lane*DPL], DPL) for h in range(G))
  zerof = UOp.const(0, dtypes.float)
  # Each block scans every PARTIALS-th chunk, keeping an online softmax across rounds.
  chunk_round = UOp.range((total_chunks-1-block_chunk)//PARTIALS+1, 4, AxisType.LOOP)
  chunk_id = block_chunk + chunk_round*PARTIALS
  valids: list[UOp] = []
  scores: list[list[UOp]] = [[zerof]*G for _ in range(SEC)]
  vfrags: list[tuple[UOp, ...]] = [()]*SEC
  for j in range(SEC):
    key = chunk_id*CHUNK + wave*SEC + j
    valid = key < valid_kv_len
    valids.append(valid)
    kfrag = _vec_load(cache_kv[0, b, kv_head, key, lane*DPL], DPL)
    # V is prefetched in the score pass so both streams are in flight together
    vfrags[j] = tuple(valid.where(v, zerof) for v in _vec_load(cache_kv[1, b, kv_head, key, lane*DPL], DPL))
    for h in range(G):
      s = warp_reduce(sum((qf[h][i]*kfrag[i] for i in range(DPL)), UOp.const(0, dtypes.float)), full_wave=True) * (1/math.sqrt(D))
      scores[j][h] = valid.where(s, UOp.const(-1e30, dtypes.float))
  # A finite initial max keeps fully masked waves from computing exp(-inf - -inf).
  acc_reg, max_reg, sum_reg = _reg((G, DPL), 0), _reg((G,), -1e30), _reg((G,), 0)
  prev_acc, prev_max, prev_sum = acc_reg.after(chunk_round), max_reg.after(chunk_round), sum_reg.after(chunk_round)
  row_max = [functools.reduce(UOp.maximum, (scores[j][h] for j in range(SEC)), prev_max[h].load()) for h in range(G)]
  # Rescale the previous rounds to the new max, then accumulate this round's keys.
  alpha = [((prev_max[h].load()-row_max[h])*LOG2E).exp2() for h in range(G)]
  accs = [[alpha[h]*prev_acc[h, i].load() for i in range(DPL)] for h in range(G)]
  row_sums = [alpha[h]*prev_sum[h].load() for h in range(G)]
  for j in range(SEC):
    for h in range(G):
      beta = valids[j].where(((scores[j][h]-row_max[h])*LOG2E).exp2(), zerof)
      accs[h] = [a + beta*v for a, v in zip(accs[h], vfrags[j])]
      row_sums[h] = row_sums[h] + beta
  update = UOp.group(acc_reg.store(UOp.stack(*(x for acc in accs for x in acc)).reshape(G, DPL)),
                     max_reg.store(UOp.stack(*row_max)), sum_reg.store(UOp.stack(*row_sums))).end(chunk_round)
  acc_reg, max_reg, sum_reg = acc_reg.after(update), max_reg.after(update), sum_reg.after(update)
  # exchange across the block's waves through LDS (fp16 halves LDS so more blocks fit per CU)
  # Matching cache/LDS strides can reuse a loop-local cache index outside the loop. Pad that layout.
  acc_lds = UOp.alloc((WAVES, G, D + (LDS_PAD if G == SEC else 0)), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :, :D]
  ml_lds = UOp.alloc((WAVES, G, 2), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_acc = acc_lds.reshape(WAVES, G, WARP_SIZE, DPL)
  # Normalize before fp16 to avoid overflow. Nonempty waves have sum >= 1; empty waves keep their zero accumulator.
  stores = [lds_acc[wave, h, lane].store((acc_reg[h].load() / sum_reg[h].load().maximum(1)).cast(dtypes.half)) for h in range(G)]
  # NOTE: duplicate stores of the same value from every lane are harmless here
  stores += [ml_lds[wave, h, i].store(x) for h in range(G) for i, x in enumerate((max_reg[h].load(), sum_reg[h].load()))]
  acc_lds, ml_lds = acc_lds.after(*stores), ml_lds.after(*stores)
  tid = wave*WARP_SIZE + lane
  final_stores:list[UOp] = []
  for i in range(-(-G*D//(WAVES*WARP_SIZE))):
    flat = tid + i*WAVES*WARP_SIZE
    h, d = flat // D, flat % D
    M = functools.reduce(UOp.maximum, (ml_lds[w, h, 0].load() for w in range(WAVES)))
    # LDS holds normalized values; restore each wave's sum before combining.
    val = sum((((ml_lds[w, h, 0].load()-M)*LOG2E).exp2() * ml_lds[w, h, 1].load() * acc_lds[w, h, d].load().float()
               for w in range(WAVES)), zerof)
    oidx = out[b, kv_head*G + h, block_chunk, d]
    if G*D % (WAVES*WARP_SIZE): oidx = out[b, (kv_head*G + h).valid(flat < G*D), block_chunk, d]
    final_stores.append(oidx.store(val))
  hstat = tid
  M = functools.reduce(UOp.maximum, (ml_lds[w, hstat, 0].load() for w in range(WAVES)))
  L = sum((((ml_lds[w, hstat, 0].load()-M)*LOG2E).exp2() * ml_lds[w, hstat, 1].load() for w in range(WAVES)), zerof)
  q_head = (kv_head*G + hstat).valid(hstat < G) if WAVES*WARP_SIZE > G else kv_head*G + hstat
  final_stores += [stats[b, q_head, block_chunk, 0].store(M), stats[b, q_head, block_chunk, 1].store(L)]
  return UOp.group(*final_stores).end(lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, live:int|UOp) -> UOp:
  # one wave per (batch, head, 64-dim tile): every lane redundantly weights its chunks; no cross-lane traffic
  live = _unbind(live)
  B, H, C, D = cast(tuple[int, int, int, int], partial.shape)
  DT = 64 if D % 64 == 0 else WARP_SIZE  # dims per block
  assert D % DT == 0
  block_bh, block_dt = UOp.range(B*H, 0, AxisType.GLOBAL), UOp.range(D//DT, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  b, h = block_bh // H, block_bh % H
  NPD = DT // WARP_SIZE  # output dims per lane
  dims = tuple(block_dt*DT + lane*NPD + i for i in range(NPD))
  chunk = UOp.range(live, 100, AxisType.LOOP)
  def iloop(ph, val): return ph.store(ph.const_like(val))
  chunk_max = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  chunk_max_i = chunk_max.after(iloop(chunk_max, -math.inf))
  update0 = chunk_max_i.store(chunk_max_i.after(chunk).maximum(stats[b, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max_i.after(update0)
  chunk2 = UOp.range(live, 101, AxisType.LOOP)
  acc = UOp.alloc((NPD,), dtypes.float, addrspace=AddrSpace.REG)
  weight_sum = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  acc_i, weight_sum_i = acc.after(iloop(acc, 0)), weight_sum.after(iloop(weight_sum, 0))
  w = ((stats[b, h, chunk2, 0].load()-chunk_max)*LOG2E).exp2()
  update1 = UOp.group(*[acc_i[i].store(acc_i.after(chunk2)[i].load() + w*partial[b, h, chunk2, d].load()) for i, d in enumerate(dims)],
                      weight_sum_i[0].store(weight_sum_i.after(chunk2)[0].load() + w*stats[b, h, chunk2, 1].load())).end(chunk2)
  acc, weight_sum = acc_i.after(update1), weight_sum_i.after(update1)
  inv = 1 / weight_sum[0].load()
  return UOp.group(*[o[b, h, 0, d].store(acc[i].load() * inv) for i, d in enumerate(dims)]) \
    .end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int) -> Tensor:
  # Carry length bindings even when the cache was populated independently of this call.
  if isinstance(valid_kv_len, UOp): cache_kv = Tensor(cache_kv.uop.after(valid_kv_len))
  B, H, D = cache_kv.shape[1], q.shape[1], cache_kv.shape[4]
  chunks, axis = min(48, max_kv_len // 64), q.uop.axis
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device, axis=axis)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device, axis=axis)
  waves, group = 16, H // cache_kv.shape[2]
  while waves * group * ((D+LDS_PAD)*2 + 8) > 65536: waves //= 2
  assert waves > 0, "attention head group exceeds shared memory capacity"
  fxn = functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=64, waves=waves)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv, fxn=fxn)[:2]
  live = (valid_kv_len+63)//64
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  out = Tensor.empty(B, H, 1, D, dtype="float32", device=q.device, axis=axis)
  fxn = functools.partial(_amd_flash_decode_combine, live=live)
  return Tensor.custom_kernel(out, partial, stats, fxn=fxn)[0]

def _wmma_fragment(fragment:UOp, lane:UOp, rdna4:bool) -> UOp:
  return fragment.reshape(2, 2, 4)[:, lane//16, :].reshape(8) if rdna4 else fragment

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, valid_kv_len:int|UOp, q_start:int|UOp|None=None, rdna4:bool=False) -> UOp:
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start) if q_start is not None else None
  BH, M, D = q.shape
  _, B, H_KV, physical_n, cache_dim = cache.shape
  k, v = cache[0].reshape(B*H_KV, physical_n, cache_dim), cache[1].reshape(B*H_KV, physical_n, cache_dim)
  assert k.shape == v.shape and BH % k.shape[0] == 0 and k.shape[2] == D
  gqa_group = BH // k.shape[0]
  if isinstance(M, int): assert M % BLOCK_M == 0
  assert isinstance(D, int) and D % WMMA_K == 0 and D % LANES_PER_WAVE_N == 0
  TM, TN, TD, SCALE = BLOCK_M//(WAVES_M*LANES_PER_WAVE_M), BLOCK_N//LANES_PER_WAVE_N, D//(WAVES_N*LANES_PER_WAVE_N), 1/math.sqrt(D)
  # query row 0 sits at sequence position q_base (the queries may be padded beyond valid_kv_len - q_base rows)
  q_base = valid_kv_len - M if q_start is None else q_start
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)
  kv_head = block_bh // gqa_group
  q, o = (x.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m] for x in (q, o))
  k, v = k[kv_head], v[kv_head]
  wave_m, wave_n, lane = UOp.range(WAVES_M, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, lane_m, lane_n = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane, lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  Q_ELEMS_PER_THREAD, KV_ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK, BLOCK_N * D // THREADS_PER_BLOCK
  QP_lds = UOp.alloc((BLOCK_M, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)
  KV_lds = UOp.alloc((BLOCK_N, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :D]
  acc, m_i, l_i = _reg((TM, TD), 0), _reg((TM,), -math.inf), _reg((TM,), 0)
  n_tile = UOp.range(((q_base + (block_m + 1) * BLOCK_M).minimum(valid_kv_len) + BLOCK_N - 1) // BLOCK_N, 100, AxisType.LOOP)
  Q_lds = QP_lds[:, :D]
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
  load_k = UOp.range(KV_ELEMS_PER_THREAD, 90)
  kval = k.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_k].float()
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_k].store(kval).end(load_k)
  Q_lds, KV_lds_k = Q_lds.after(Q_store, K_store), KV_lds.after(Q_store, K_store)
  S_reg = _reg((TM, TN), 0, n_tile)
  k_qk, tm1, tn1 = UOp.range(D//WMMA_K, 101, AxisType.LOOP), UOp.range(TM//WMMA_ACC, 200), UOp.range(TN, 201)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
  # All waves must finish reading Q/K before their shared memory is reused for P/V.
  q_frag, k_frag = (_wmma_fragment(f, lane, rdna4) for f in (q_frag, k_frag))
  qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk).barrier()
  S_reg = S_reg.after(qk_done, S_reg.store(S_reg * SCALE))
  rm, rn = UOp.range(TM, 250), UOp.range(TN, 251)
  q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + (lane_m*TM + rm if rdna4 else rm*LANES_PER_WAVE_M + lane_m)
  k_idx = n_tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
  causal = (k_idx <= q_idx) & (k_idx < valid_kv_len)
  S_reg = S_reg.after(S_reg[rm, rn].store(causal.where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))).end(rm, rn))
  m_ij, rm2 = _reg((TM,), -math.inf, n_tile), UOp.range(TN, 261, AxisType.LOOP)
  m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
  ri_w = UOp.range(TM, 270)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], maximum=True)).end(ri_w))
  tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
  S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
  p_local, ri_ws = _reg((TM,), 0, n_tile), UOp.range(TM, 295)
  p_sum = p_local.after(p_local[ri_ws].store(sum((warp_reduce(S_reg[ri_ws, rn]) for rn in range(TN)), S_reg.const_like(0))).end(ri_ws))
  P_lds = QP_lds.flatten()[:WAVES_N * BLOCK_M * BLOCK_N].reshape(WAVES_N, BLOCK_M, BLOCK_N)
  # gfx11 distributes even/odd rows between half-waves; gfx12 distributes the low/high eight rows.
  row_shape = (LANES_PER_WAVE_M, TM) if rdna4 else (TM, LANES_PER_WAVE_M)
  row_lane, row_elem = (2, 3) if rdna4 else (3, 2)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, *row_shape, TN, LANES_PER_WAVE_N).permute(1, 0, row_lane, 5, row_elem, 4) \
    .reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))
  beta_i, ri4, rj4 = UOp.alloc((TM,), dtypes.float, addrspace=AddrSpace.REG), UOp.range(TM, 330), UOp.range(TD, 331)
  m_new = m_i[ri4].maximum(m_ij[ri4])
  alpha_val, beta_val = ((m_i[ri4] - m_new) * LOG2E).exp2(), ((m_ij[ri4] - m_new) * LOG2E).exp2()
  correction = UOp.group(acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
                         l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
                         m_i[ri4].store(m_new), beta_i[ri4].store(beta_val)).end(ri4)
  acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  V_lds = UOp.alloc((D, BLOCK_N + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_copy, load_v = V_lds.after(qk_done).permute(1, 0), UOp.range(KV_ELEMS_PER_THREAD, 390)
  v_pos = n_tile*BLOCK_N + (tid*KV_ELEMS_PER_THREAD + load_v)//D
  vval = (v_pos < valid_kv_len).where(v.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_v].float(), 0)
  V_store = V_copy.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_v].store(vval).end(load_v)
  P_lds, V_lds = P_lds.after(P_store, V_store), V_lds.after(P_store, V_store)
  pv_acc = _reg((TM, TD), 0, n_tile)
  k_pv, tm2, tn2 = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.LOOP), UOp.range(TM//WMMA_ACC, 401), UOp.range(TD, 402)
  pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  p_frag, v_frag = (_wmma_fragment(f, lane, rdna4) for f in (p_frag, v_frag))
  pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)
  ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
  n_tile_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).end(n_tile)
  acc, l_i, m_i = acc.after(n_tile_end), l_i.after(n_tile_end), m_i.after(n_tile_end)
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))
  o = o.reshape(WAVES_M, *row_shape, WAVES_N, TD, LANES_PER_WAVE_N) \
    .permute(0, 3, row_lane-1, 5, row_elem-1, 4).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

def flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  # cached flash attention on the half KV cache (already written through assigned_kv); valid_end stays bound at the graph level
  T_real, q_start = q.shape[2], None
  D, N, group = q.shape[3], assigned_kv.shape[3], q.shape[1] // assigned_kv.shape[2]
  decode = resolve(T_real == 1, False)
  # Non-power-of-two decode dimensions can lose tail-store masks. Q/P, K, and V use separate LDS allocations.
  supported = D % 32 == 0 and (D & (D-1) == 0 and N % 64 == 0 and group*((D+LDS_PAD)*2+8) <= 65536 if decode else
    D >= 64 and 2*(2*BLOCK_M*(D+LDS_PAD) + D*(BLOCK_N+LDS_PAD)) <= 65536 and N % BLOCK_N == 0 and q.max_shape[2] % BLOCK_M == 0)
  if not supported:
    k, v = (assigned_kv[i, :, :, :valid_end].float() for i in range(2))
    mask = None if decode else Tensor.full((T_real, valid_end), -math.inf, dtype=dtypes.float32, device=q.device).triu(valid_end-T_real+1)
    return q.float().scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
  if decode: return amd_flash_attention_decode(q.half(), assigned_kv, valid_end, cast(int, N))
  if isinstance(T_real, UOp):
    # symbolic chunk: pad the queries to the static tile; garbage rows are sliced off
    T_pad = q.max_shape[2]
    assert T_pad % BLOCK_M == 0, "chunk_size must be a multiple of 32"
    q, q_start = q.pad_to((*q.shape[:2], T_pad, q.shape[3])), valid_end - T_real
  B, H, T, D = q.shape
  out = Tensor.empty(B, H, T, D, dtype="float32", device=q.device, axis=q.uop.axis).reshape(B*H, T, D)
  fxn = functools.partial(_amd_flash_attention, valid_kv_len=valid_end, q_start=q_start, rdna4=_wmma_rdna4(q.device))
  if isinstance(valid_end, UOp): assigned_kv = Tensor(assigned_kv.uop.after(valid_end))
  out = Tensor.custom_kernel(out, q.half().reshape(B*H, T, D), assigned_kv, fxn=fxn)[0].reshape(B, H, T, D)
  return out if q_start is None else out[:, :, :T_real]

# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows, cols = tuple(row_base+i for i in range(row_tile)), tuple(lane + i*32 for i in range(key_dim//32))
  current = UOp.alloc((row_tile*key_dim//32,), dtypes.float32, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(state[bh, row, col].float() if initial is None else
    initial.where(0, state[bh, row, col].float()) for row in rows for col in cols))))
  token = UOp.range(tokens, 2, AxisType.LOOP)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_dim//32+i].load() for i in range(key_dim//32))
    decayed, bv = tuple(x * alpha[bh, token, col if alpha_dim > 1 else 0].load() for x,col in zip(previous, cols)), beta[bh, token].load()
    state_k = warp_reduce(sum((x*y for x,y in zip(decayed, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
    state_q = warp_reduce(sum((x*y for x,y in zip(decayed, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k) * bv
    updates += [x + delta*y for x,y in zip(decayed, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens) and state.shape == (batch, heads, value_dim, key_dim)
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, key_dim))
  assert key_dim % 32 == 0 and value_dim % 4 == 0
  assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
  assert state.uop.contiguous_view_offset() is not None, "recurrent state must be contiguous"
  if start_pos is not None:
    assert start_pos.uop.is_bound_var
    state = Tensor(state.uop.after(start_pos.uop))
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  call = _gated_delta_prefill_kernel(*params, None if start_pos is None else start_pos.uop.unbound()).call(*contig)
  return Tensor(contig[0].after(call))
