import json, pathlib, zipfile, pickle, tarfile, struct, functools, io
from collections import OrderedDict
from typing import Any, Callable, BinaryIO, Iterable
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, argsort, DEBUG, Timing, CI, unwrap, GlobalCounters, tqdm, round_up, T
from tinygrad.shape.view import strides_for_shape

class TensorIO(io.RawIOBase, BinaryIO):
  def __init__(self, t: Tensor):
    if t.ndim != 1 or t.dtype != dtypes.uint8: raise ValueError("Tensor must be 1d and of dtype uint8!")
    self._position, self._tensor = 0, t

  def readable(self) -> bool: return True
  def read(self, size: int = -1) -> bytes:
    if (buf:=super().read(size)) is None: raise ValueError("io.RawIOBase.read returned None") # only happens if readinto returns None (never)
    return buf
  def readinto(self, buffer: Any) -> int:
    data = self._tensor[self._position:self._position+len(buffer)].data()
    buffer[:len(data)] = data
    self._position += len(data)
    return len(data)

  def seekable(self) -> bool: return True
  def seek(self, offset: int, whence: int = 0) -> int:
    self._position = min(len(self._tensor), max(0, [offset, self._position+offset, len(self._tensor)+offset][whence]))
    return self._position

  # required to correctly implement BinaryIO
  def __enter__(self): return self
  def write(self, s: Any): raise io.UnsupportedOperation("TensorIO.write not supported")
  def writelines(self, lines: Iterable[Any]): raise io.UnsupportedOperation("TensorIO.writelines not supported")

safe_dtypes = {"BOOL":dtypes.bool, "I8":dtypes.int8, "U8":dtypes.uint8, "I16":dtypes.int16, "U16":dtypes.uint16, "I32":dtypes.int, "U32":dtypes.uint,
               "I64":dtypes.int64, "U64":dtypes.uint64, "F16":dtypes.float16, "BF16":dtypes.bfloat16, "F32":dtypes.float32, "F64":dtypes.float64}
inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}

def accept_filename(func: Callable[[Tensor], T]) -> Callable[[Tensor|str|pathlib.Path], T]:
  @functools.wraps(func)
  def wrapper(fn: Tensor|str|pathlib.Path) -> T: return func(Tensor(pathlib.Path(fn)) if not isinstance(fn, Tensor) else fn)
  return wrapper

@accept_filename
def safe_load_metadata(t:Tensor) -> tuple[Tensor, int, dict[str, Any]]:
  """
  Loads a .safetensor file, returning the source tensor, data start position, and metadata.
  """
  data_start = int.from_bytes(t[0:8].data(), "little") + 8
  return t, data_start, json.loads(t[8:data_start].data().tobytes())

def safe_load(fn:Tensor|str|pathlib.Path) -> dict[str, Tensor]:
  """
  Loads a .safetensor file, returning the `state_dict`.

  ```python
  state_dict = nn.state.safe_load("test.safetensor")
  ```
  """
  t, data_start, metadata = safe_load_metadata(fn)
  data = t[data_start:]
  return { k: data[v['data_offsets'][0]:v['data_offsets'][1]].bitcast(safe_dtypes[v['dtype']]).reshape(v['shape'])
          for k, v in metadata.items() if k != "__metadata__" }

def safe_save(tensors:dict[str, Tensor], fn:str, metadata:dict[str, Any]|None=None):
  """
  Saves a `state_dict` to disk in a .safetensor file with optional metadata.

  ```python
  t = Tensor([1, 2, 3])
  nn.state.safe_save({'t':t}, "test.safetensor")
  ```
  """
  headers, offset = {}, 0
  if metadata: headers['__metadata__'] = metadata
  for k,v in tensors.items():
    headers[k] = {'dtype': inverse_safe_dtypes[v.dtype], 'shape': list(v.shape), 'data_offsets':[offset, offset+v.nbytes()]}
    offset += v.nbytes()
  j = json.dumps(headers, separators=(',', ':'))
  j += "\x20"*(round_up(len(j),8)-len(j))
  pathlib.Path(fn).unlink(missing_ok=True)
  t = Tensor.empty(8+len(j)+offset, dtype=dtypes.uint8, device=f"disk:{fn}")
  t[0:8].bitcast(dtypes.int64).assign([len(j)])
  t[8:8+len(j)].assign(list(j.encode('utf-8')))
  for k,v in safe_load(t).items(): v.assign(tensors[k])

# state dict

def get_state_dict(obj, prefix:str='', tensor_type=Tensor) -> dict[str, Tensor]:
  """
  Returns a `state_dict` of the object, with optional prefix.

  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(nn.state.get_state_dict(net).keys())
  ```
  """
  if isinstance(obj, tensor_type): return {prefix.strip('.'):obj}
  if hasattr(obj, '_asdict'): return get_state_dict(obj._asdict(), prefix, tensor_type)  # namedtuple
  if isinstance(obj, OrderedDict): return get_state_dict(dict(obj), prefix, tensor_type)
  if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__, prefix, tensor_type)
  state_dict = {}
  if isinstance(obj, (list, tuple)):
    for i,x in enumerate(obj): state_dict.update(get_state_dict(x, f"{prefix}{str(i)}.", tensor_type))
  elif isinstance(obj, dict):
    for k,v in obj.items(): state_dict.update(get_state_dict(v, f"{prefix}{str(k)}.", tensor_type))
  return state_dict

def get_parameters(obj) -> list[Tensor]:
  """
  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(len(nn.state.get_parameters(net)))
  ```
  """
  return list(get_state_dict(obj).values())

def load_state_dict(model, state_dict:dict[str, Tensor], strict=True, verbose=True, consume=False, realize=True) -> list[Tensor]:
  """
  Loads a `state_dict` into a model. Return the loaded Tensors.

  ```python
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  state_dict = nn.state.get_state_dict(net)
  nn.state.load_state_dict(net, state_dict)
  ```
  """
  start_mem_used = GlobalCounters.mem_used
  ret = []
  with Timing("loaded weights in ",
              lambda et_ns: f", {(B:=(GlobalCounters.mem_used-start_mem_used))/1e9:.2f} GB loaded at {B/et_ns:.2f} GB/s", enabled=verbose):
    model_state_dict = get_state_dict(model)
    if DEBUG >= 1 and len(state_dict) > len(model_state_dict):
      print("WARNING: unused weights in state_dict", sorted(list(state_dict.keys() - model_state_dict.keys())))
    for k,v in (t := tqdm(model_state_dict.items(), disable=CI or not verbose)):
      t.desc = f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, {k:50s}: "
      if k not in state_dict and not strict:
        if DEBUG >= 1: print(f"WARNING: not loading {k}")
        continue
      if v.shape != state_dict[k].shape:
        raise ValueError(f'Shape mismatch in layer `{k}`: Expected shape {v.shape}, but found {state_dict[k].shape} in state dict.')
      if isinstance(v.device, tuple):
        if isinstance(state_dict[k].device, tuple): v.replace(state_dict[k])
        else: v.replace(state_dict[k].shard(v.device, v.uop.axis))
      else: v.replace(state_dict[k].to(v.device))
      if realize: v.realize()
      if consume: del state_dict[k]
      ret.append(v)
  return ret

@accept_filename
def tar_extract(t: Tensor) -> dict[str, Tensor]:
  """
  ```python
  tar_extract(fn: Tensor | str | Path) -> dict[str, Tensor]
  ```

  Extracts files from a tar archive and returns them as a dictionary of names (keys) and tensors (values).

  ```python
  tensors = nn.state.tar_extract(Tensor(pathlib.Path("archive.tar")))
  ```
  """
  with tarfile.open(fileobj=TensorIO(t), mode="r") as tar:
    return {member.name:t[member.offset_data:member.offset_data+member.size] for member in tar if member.type == tarfile.REGTYPE}

# torch support!

@accept_filename
def torch_load(t:Tensor) -> dict[str, Tensor]:
  """
  ```python
  torch_load(fn: Tensor | str | Path) -> dict[str, Tensor]
  ```

  Loads a torch .pth file, returning the `state_dict`.

  ```python
  state_dict = nn.state.torch_load("test.pth")
  ```
  """
  offsets: dict[str|int, int] = {}
  lens: dict[str|int, int] = {}
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=None, backward_hooks=None, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    lens[storage[2]] = storage[4] * storage[1].itemsize
    if storage[2] not in offsets: return None
    byte_offset = offsets[storage[2]]+storage_offset*storage[1].itemsize
    ret = t[byte_offset:byte_offset+prod(size)*storage[1].itemsize].bitcast(storage[1])

    # 7 lines to deal with permuted tensors. NOTE: this currently requires reading off the disk
    shape_strides = [(s, st) for s,st in zip(size, stride) if s != 1]
    permute_indexes = [len(shape_strides)-1-y for y in argsort([x[1] for x in shape_strides])]
    if tuple(permute_indexes) != tuple(range(len(permute_indexes))):
      intermediate_shape = tuple([shape_strides[x][0] for x in argsort(permute_indexes)])
      assert tuple([shape_strides[i][1] for i in argsort(permute_indexes)]) == strides_for_shape(intermediate_shape), "nonpermutable strides"
      if DEBUG >= 3: print(f"WARNING: this torch load is slow. to permute {intermediate_shape} with {permute_indexes}")
      assert storage[1] != dtypes.bfloat16, "can't permute BF16"
      # TODO: find a nice way to support all shapetracker on disktensors
      ret = ret.to(None).reshape(intermediate_shape).permute(permute_indexes)

    return ret.reshape(size)

  class Parameter:
    def __setstate__(self, state): self.tensor = state[0]

  deserialized_objects: dict[str, Any] = {}
  intercept = {"HalfStorage": dtypes.float16, "FloatStorage": dtypes.float32, "BFloat16Storage": dtypes.bfloat16,
               "IntStorage": dtypes.int32, "BoolStorage": dtypes.bool,
               "LongStorage": dtypes.int64, "_rebuild_tensor_v2": _rebuild_tensor_v2, "FloatTensor": None, "Parameter": Parameter}
  whitelist = {"torch", "collections", "numpy", "_codecs"}  # NOTE: this is not for security, only speed
  class Dummy: pass
  class TorchPickle(pickle.Unpickler):
    def find_class(self, module, name):
      module_root = module.split(".")[0]
      if module_root not in whitelist:
        if DEBUG >= 2: print(f"WARNING: returning Dummy for {module} {name}")
        return Dummy
      return intercept[name] if module_root == "torch" else super().find_class(module, name)
    def persistent_load(self, pid): return deserialized_objects.get(pid, pid)

  fobj = io.BufferedReader(TensorIO(t))
  def passthrough_reset(v: bool): return fobj.seek(0, 0) or v

  if passthrough_reset(zipfile.is_zipfile(fobj)): # NOTE: passthrough_reset required to support python < 3.14
    myzip = zipfile.ZipFile(fobj, 'r')
    base_name = myzip.namelist()[0].split('/', 1)[0]
    for n in myzip.namelist():
      if n.startswith(f'{base_name}/data/'):
        with myzip.open(n) as myfile:
          offsets[n.split("/")[-1]] = myfile._orig_compress_start # type: ignore
    with myzip.open(f'{base_name}/data.pkl') as myfile:
      return TorchPickle(myfile).load()
  elif passthrough_reset(tarfile.is_tarfile(fobj)): # NOTE: passthrough_reset required to support python < 3.11
    with tarfile.open(fileobj=fobj, mode="r") as tar:
      storages_offset = tar.getmember('storages').offset_data
      f = unwrap(tar.extractfile('storages'))
      for i in range(TorchPickle(f).load()):  # num_storages
        (key, _, storage_type), sz = TorchPickle(f).load(), struct.unpack('<q', f.read(8))[0]
        offsets[key] = storages_offset + f.tell()
        f.seek(sz*storage_type.itemsize, 1)
      f = unwrap(tar.extractfile('tensors'))
      for _ in range(TorchPickle(f).load()):  # num_tensors
        (key, storage_id, _), ndim, _ = TorchPickle(f).load(), struct.unpack('<i', f.read(4))[0], f.read(4)
        size, stride = struct.unpack(f'<{ndim}q', f.read(8 * ndim)), struct.unpack(f'<{ndim}q', f.read(8 * ndim))
        storage_offset = struct.unpack('<q', f.read(8))[0]
        deserialized_objects[str(key)] = _rebuild_tensor_v2((None, storage_type, storage_id, None, -1), storage_offset, size, stride)
      return {k:v.tensor if isinstance(v, Parameter) else v for k,v in TorchPickle(unwrap(tar.extractfile('pickle'))).load().items()}
  else:
    pkl = TorchPickle(fobj)
    _, _, _, rwd, _, ids, base_offset = pkl.load(), pkl.load(), pkl.load(), fobj.tell(), pkl.load(), pkl.load(), fobj.tell()
    for i in ids:
      offsets[i] = base_offset + 8
      base_offset += 8 + lens[i]
    fobj.seek(rwd)
    return TorchPickle(fobj).load()

def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int) -> Tensor:
  """
  Converts ggml tensor data to a tinygrad tensor.

  Supported native types: float32 (id: 0), float16 (id: 1), int8 (id: 16), int16 (id: 17), int32 (id: 18)
  Supported quantized types: Q4_0 (id: 2), Q4_1 (id: 3), Q8_0 (id: 8), Q6_K (id: 14)
  """
  # https://github.com/ggerganov/ggml/blob/6dccc647264f5429df2624f36138f601e7ce23e5/include/ggml.h#L356

  # native types
  if (dtype := { 0: dtypes.float32, 1: dtypes.float16, 16: dtypes.int8, 17: dtypes.int16, 18: dtypes.int32 }.get(ggml_type)) is not None:
    return t[:dtype.itemsize * n].bitcast(dtype)

  def q_to_uint8(t: Tensor, b: int) -> Tensor:
    # TODO: rewrite with arange?
    shift_tensor, bitmask = Tensor.stack(*[ Tensor(2**(i*b), device=t.device, dtype=t.dtype) for i in range(8//b) ]), 0xff >> (8 - b)
    return t.unsqueeze(-1).expand((*t.shape,8//b)).idiv(shift_tensor).bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

  # map to (number of elements, number of bytes)
  if (nelements_nbytes := { 2: (32, 18), 3: (32, 20), 14: (256, 210), 8: (32, 34) }.get(ggml_type)) is not None:
    blocks = t[:(n//nelements_nbytes[0])*nelements_nbytes[1]].reshape((-1, nelements_nbytes[1]))
    if ggml_type == 2: return (q_to_uint8(blocks[:,2:], 4).bitcast(dtypes.int8) - 8) * blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
    if ggml_type == 3:
      d, m = (blocks[:,s:s+2].bitcast(dtypes.float16).cast(dtypes.float32) for s in [ 0, 2 ])
      return q_to_uint8(blocks[:,4:], 4).bitcast(dtypes.int8) * d + m
    if ggml_type == 8: return blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32) * blocks[:,2:].bitcast(dtypes.int8)
    if ggml_type == 14:
      xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
      scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
      d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256))
      return d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales
  raise ValueError(f"GGML type '{ggml_type}' is not supported!")

@accept_filename
def gguf_load(tensor: Tensor) -> tuple[dict, dict[str, Tensor]]:
  """
  Loads a .gguf file, returning the `kv_data` and `state_dict`.

  ```python
  gguf_tensor = Tensor(pathlib.Path("Meta-Llama-3-8B-Instruct.Q4_0.gguf")).to(Device.DEFAULT)
  kv_data, state_dict = nn.state.gguf_load(gguf_tensor)
  ```

  NOTE: The provided tensor must be on a device that supports execution.
  """
  reader, kv_data, state_dict = io.BufferedReader(TensorIO(tensor), 1_000_000), {}, {}
  def read_unpack(fmt: str, n: int): return struct.unpack(fmt, reader.read(n))[0]
  def read_str(): return str(reader.read(read_uint64()), "utf-8")
  def read_arr():
    reader, n = readers[read_int32()], read_uint64()
    return [ reader() for _ in range(n) ]

  readers: dict[int, Callable[[], Any]] = { 8: read_str, 9: read_arr, **{ t: functools.partial(read_unpack, "<"+f, nb) for t,f,nb in \
    [ (0,"c",1), (1,"b",1), (2,"H",2), (3,"h",2), (4,"I",4), (5,"i",4), (6,"f",4), (7,"?",1), (10,"Q",8), (11,"q",8), (12,"d",8) ] } }
  read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

  magic, version, n_tensors, n_kv = reader.read(4), read_int32(), read_int64(), read_int64()
  if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")
  for _ in range(n_kv):
    k, typ = read_str(), read_int32()
    kv_data[k] = readers[typ]()

  t_infos = [ (read_str(), tuple(read_uint64() for _ in range(read_uint32())), read_int32(), read_uint64()) for _ in range(n_tensors) ]
  alignment, pos = kv_data.get("general.alignment", 32), reader.tell()
  data_start = round_up(pos, alignment)

  for name, dims, typ, off in t_infos: state_dict[name] = ggml_data_to_tensor(tensor[data_start + off:], prod(dims), typ).reshape(*reversed(dims))

  return kv_data, state_dict
