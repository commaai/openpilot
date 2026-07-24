import functools, io, pathlib, re, struct
from typing import Any, Callable

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, round_up
from tinygrad.nn.state import TensorIO

# ggml packs each iq grid entry as N bytes (N=4 for uint32 grids, N=8 for uint64 grids) in a single word. See ggml-common.h.
@functools.lru_cache(None)
def _ggml_iq_grid(device: str, grid: tuple[int, ...], grid_shape: tuple[int, int]) -> Tensor:
  values = [float((w >> (8*i)) & 0xFF) for w in grid for i in range(grid_shape[1])]
  return Tensor(values, dtype=dtypes.float32, device=device).reshape(grid_shape)

# native types {ggml_type: dtype}
_GGML_NATIVE = {0: dtypes.float32, 1: dtypes.float16, 24: dtypes.int8, 25: dtypes.int16,
                26: dtypes.int32, 27: dtypes.int64, 28: dtypes.float64, 30: dtypes.bfloat16}

# quant types {ggml_type: (number of elements, number of bytes)}
_GGML_QUANT = {2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24), 8:(32,34),
               12:(256,144), 13:(256,176), 14:(256,210), 18:(256,98), 21:(256,110), 22:(256,82), 23:(256,136), 39:(32,17), 41:(128,18)}

def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int) -> Tensor:
  """
  Converts ggml tensor data to a tinygrad tensor.

  Supported native types: float32 (id: 0), float16 (id: 1), int8 (id: 24),
  int16 (id: 25), int32 (id: 26), int64 (id: 27), float64 (id: 28), bfloat16 (id: 30)
  Supported quantized types: Q4_0 (id: 2), Q4_1 (id: 3), Q5_0 (id: 6),
  Q5_1 (id: 7), Q8_0 (id: 8), Q4_K (id: 12), Q5_K (id: 13),
  Q6_K (id: 14), IQ3_XXS (id: 18), IQ3_S (id: 21), IQ2_S (id: 22), IQ4_XS (id: 23), MXFP4 (id: 39), Q1_0 (id: 41)
  """
  # https://github.com/ggerganov/ggml/blob/323951f1bdcdfbd5b5ff3a9a7c3770e63b1a560e/include/ggml.h#L356

  if (dtype := _GGML_NATIVE.get(ggml_type)) is not None:
    return t[:dtype.itemsize * n].contiguous().bitcast(dtype)

  def q_to_uint8(t: Tensor, b: int) -> Tensor:
    # TODO: rewrite with arange?
    shift_tensor, bitmask = Tensor.stack(*[ Tensor(2**(i*b), device=t.device, dtype=t.dtype) for i in range(8//b) ]), 0xff >> (8 - b)
    return t.unsqueeze(-1).expand((*t.shape,8//b)).div(shift_tensor, rounding_mode="trunc").bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

  if (nelements_nbytes := _GGML_QUANT.get(ggml_type)) is not None:
    from tinygrad.runtime.autogen import ggml_common as _ggml
    blocks = t[:(n//nelements_nbytes[0])*nelements_nbytes[1]].reshape((-1, nelements_nbytes[1])).contiguous()
    if ggml_type == 2: return (q_to_uint8(blocks[:,2:], 4).bitcast(dtypes.int8) - 8) * blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
    if ggml_type == 3:
      d, m = (blocks[:,s:s+2].bitcast(dtypes.float16).cast(dtypes.float32) for s in [ 0, 2 ])
      return q_to_uint8(blocks[:,4:], 4).bitcast(dtypes.int8) * d + m
    if ggml_type in (6, 7):
      d = blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
      qh_off = 2 if ggml_type == 6 else 4
      qh = q_to_uint8(blocks[:,qh_off:qh_off+4], 1).reshape((-1, 8, 4)).transpose(-1, -2).flatten(-2).bitcast(dtypes.int8)
      q = q_to_uint8(blocks[:,qh_off+4:], 4).bitcast(dtypes.int8) + qh * 16
      return q * d + (blocks[:,2:4].bitcast(dtypes.float16).cast(dtypes.float32) if ggml_type == 7 else -16 * d)
    if ggml_type == 8: return blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32) * blocks[:,2:].bitcast(dtypes.int8)
     # Q4_K: 256 elements per 144-byte block (d:2, dmin:2, scales:12, qs:128)
     # Q5_K: 256 elements per 176-byte block (d:2, dmin:2, scales:12, qh:32, qs:128)
    if ggml_type in (12, 13):
      d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [0, 2])
      s = blocks[:,4:16]  # 12 bytes: 6-bit scales[0-3], 6-bit mins[0-3], high bits[4-7]
      sc = s[:,0:4].bitwise_and(63).cat(s[:,8:12].bitwise_and(0xF).bitwise_or(s[:,0:4].rshift(6).lshift(4)), dim=-1)
      mn = s[:,4:8].bitwise_and(63).cat(s[:,8:12].rshift(4).bitwise_or(s[:,4:8].rshift(6).lshift(4)), dim=-1)
      qs_off = 48 if ggml_type == 13 else 16
      q = Tensor.stack((qs:=blocks[:,qs_off:qs_off+128].reshape(-1,4,32)).bitwise_and(0xF), qs.rshift(4), dim=2).reshape(-1,8,32)
      if ggml_type == 13: q = q + q_to_uint8(blocks[:,16:48], 1).reshape(-1, 8, 32) * 16
      return (d * sc.unsqueeze(-1) * q - dmin * mn.unsqueeze(-1)).flatten(-2)
    if ggml_type == 14:
      xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
      scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
      d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256))
      return d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales
    if ggml_type == 18:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scale_words = blocks[:, 66:98].bitcast(dtypes.uint32)
      db = d * (scale_words.rshift(28).cast(dtypes.float32) + 0.5).reshape((-1, 8, 1, 1)) * 0.5
      sign_idx = scale_words.unsqueeze(-1).rshift(
        Tensor([0, 7, 14, 21], device=t.device, dtype=dtypes.uint32)).bitwise_and(0x7F).reshape((-1, 32)).cast(dtypes.int32)
      even_signs = Tensor([i | (0x80 if i.bit_count() % 2 else 0) for i in range(128)], dtype=dtypes.uint8, device=t.device)
      signs = (q_to_uint8(even_signs[sign_idx].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      grid = _ggml_iq_grid(t.device, _ggml.iq3xxs_grid, (256, 4))[blocks[:, 2:66]].reshape((-1, 8, 4, 8))
      return (db * grid * signs).flatten(-3)
    if ggml_type == 21:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scales = (1 + 2 * q_to_uint8(blocks[:, 106:110].reshape((-1, 4, 1)), 4).reshape((-1, 8))).cast(dtypes.float32).reshape((-1, 8, 1, 1))
      qh = q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 1).reshape((-1, 64)).cast(dtypes.uint16)
      signs = (q_to_uint8(blocks[:, 74:106].reshape((-1, 32, 1)), 1).reshape((-1, 256)) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      q = blocks[:, 2:66].cast(dtypes.uint16) + qh.lshift(8)
      return (d * scales * _ggml_iq_grid(t.device, _ggml.iq3s_grid, (512, 4))[q].reshape((-1, 8, 4, 8)) * signs).flatten(-3)
    if ggml_type == 22:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      db = d * (q_to_uint8(blocks[:, 74:82].reshape((-1, 8, 1)), 4).reshape((-1, 16)).cast(dtypes.float32) + 0.5).reshape((-1, 16, 1, 1)) * 0.25
      signs = (q_to_uint8(blocks[:, 34:66].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 16, 2, 8))
      qh = q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 2).reshape((-1, 32)).cast(dtypes.uint16)
      q = blocks[:, 2:34].cast(dtypes.uint16) + qh.lshift(8)
      return (db * _ggml_iq_grid(t.device, _ggml.iq2s_grid, (1024, 8))[q].reshape((-1, 16, 2, 8)) * signs).flatten(-3)
    if ggml_type == 23:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1))
      scale_shifts = Tensor([0, 2, 4, 6, 8, 10, 12, 14], device=t.device, dtype=dtypes.uint16)
      iq4_xs_lut = Tensor(list(_ggml.kvalues_iq4nl), dtype=dtypes.float32, device=t.device)
      scales_l = Tensor.stack((sl:=blocks[:, 4:8]).bitwise_and(0xF), sl.rshift(4), dim=2).reshape((-1, 8))
      scales_h = blocks[:, 2:4].bitcast(dtypes.uint16).unsqueeze(-1).rshift(scale_shifts).bitwise_and(0x03).reshape((-1, 8)).cast(dtypes.uint8)
      scales = (scales_l.bitwise_or(scales_h.lshift(4)).bitcast(dtypes.int8) - 32).cast(dtypes.float32).reshape((-1, 8, 1))
      q = (qs:=blocks[:, 8:].reshape((-1, 8, 16))).bitwise_and(0xF).cat(qs.rshift(4), dim=2)
      return (d * scales * iq4_xs_lut[q]).flatten(-2)
    if ggml_type == 39:
      e = blocks[:, 0].cast(dtypes.uint32)
      small_bits = Tensor([0x00200000, 0x00400000], dtype=dtypes.uint32, device=t.device)[e.clip(0, 1).cast(dtypes.int32)] # e = 0 or e = 1 case
      d = (e < 2).where(small_bits, ((e - 1) * 0x00800000).cast(dtypes.uint32)).bitcast(dtypes.float32).unsqueeze(-1)
      codes = q_to_uint8(blocks[:, 1:17], 4)
      fp4_lut = Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                       -0.0,-1.0,-2.0,-3.0,-4.0,-6.0,-8.0,-12.0],
                      dtype=dtypes.float32, device=t.device)
      fp4_val = fp4_lut[codes]
      return (fp4_val * d).flatten(-2)[:n]
    if ggml_type == 41:
      d = blocks[:,:2].bitcast(dtypes.float16)
      bits = q_to_uint8(blocks[:,2:], 1).reshape(-1, 8, 16).transpose(-1, -2).flatten(-2).bitcast(dtypes.int8)
      return d * (bits * 2 - 1)
  raise ValueError(f"GGML type '{ggml_type}' is not supported!")

def _read_unpack(fmt: str, n: int, r:io.BufferedIOBase): return struct.unpack(fmt, r.read(n))[0]
def read_str(r:io.BufferedIOBase): return str(r.read(read_uint64(r)), "utf-8")
def read_arr(r:io.BufferedIOBase):
  item_reader, n = readers[read_int32(r)], read_uint64(r)
  return [item_reader(r) for _ in range(n)]

readers: dict[int, Callable[[io.BufferedIOBase], Any]] = { 8: read_str, 9: read_arr,
  **{ t: functools.partial(_read_unpack, "<"+f, nb) for t,f,nb in \
    [ (0,"c",1), (1,"b",1), (2,"H",2), (3,"h",2), (4,"I",4), (5,"i",4), (6,"f",4), (7,"?",1), (10,"Q",8), (11,"q",8), (12,"d",8) ] } }
read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

def _gguf_parse(tensor: Tensor) -> tuple[dict, dict[str, Tensor]]:
  # TODO: remove the need for copy to default device
  tensor = tensor.to(None).realize()
  r = io.BufferedReader(TensorIO(tensor), 1_000_000)
  magic, version, n_tensors, n_kv = r.read(4), read_int32(r), read_int64(r), read_int64(r)
  if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")

  kv_data = {}
  for _ in range(n_kv):
    k, typ = read_str(r), read_int32(r)
    kv_data[k] = readers[typ](r)

  t_infos = [ (read_str(r), tuple(read_uint64(r) for _ in range(read_uint32(r))), read_int32(r), read_uint64(r)) for _ in range(n_tensors) ]
  alignment, pos = kv_data.get("general.alignment", 32), r.tell()
  data_start = round_up(pos, alignment)

  state_dict = {name: ggml_data_to_tensor(tensor[data_start + off:], prod(dims), typ).reshape(*reversed(dims)) for name, dims, typ, off in t_infos}
  return kv_data, state_dict

def _gguf_split_paths(path: pathlib.Path, kv: dict) -> list[pathlib.Path]:
  if (total := kv.get('split.count', 1)) <= 1: return [path]
  if kv.get('split.no', 0) != 0: raise ValueError(f"multi-part GGUF must be loaded from the first split, got split.no={kv['split.no']}")
  if not (m := re.match(r"^(.*)-00001-of-\d{5}\.gguf$", str(path))): raise ValueError(f"first split path must end with -00001-of-NNNNN.gguf: {path}")
  return [pathlib.Path(f"{m.group(1)}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total+1)]

def gguf_load(fn: Tensor|str|pathlib.Path) -> tuple[dict, dict[str, Tensor]]:
  """
  Loads a .gguf file, returning the `kv_data` and `state_dict`. Multi-part splits are auto-merged when loaded by path.

  ```python
  import pathlib
  from tinygrad import Device, Tensor
  from tinygrad.llm.gguf import gguf_load

  gguf_tensor = Tensor(pathlib.Path("Meta-Llama-3-8B-Instruct.Q4_0.gguf")).to(Device.DEFAULT)
  kv_data, state_dict = gguf_load(gguf_tensor)
  ```

  NOTE: The provided tensor must be on a device that supports execution.
  """
  kv, sd = _gguf_parse(fn if isinstance(fn, Tensor) else Tensor(pathlib.Path(fn)))
  if kv.get('split.count', 1) <= 1: return kv, sd
  if isinstance(fn, Tensor): raise ValueError("multi-part GGUF requires a path argument (got Tensor)")
  for pp in _gguf_split_paths(pathlib.Path(fn), kv)[1:]: sd.update(_gguf_parse(Tensor(pp))[1])
  return kv, sd
