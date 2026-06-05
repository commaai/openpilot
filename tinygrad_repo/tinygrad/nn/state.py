import json, pathlib, zipfile, pickle, tarfile, struct, functools, io, zlib
from collections import OrderedDict
from typing import Any, Callable, BinaryIO, Iterable, cast
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, argsort, DEBUG, Timing, GlobalCounters, tqdm, round_up, T, strides_for_shape

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
    for k,v in (t := tqdm(model_state_dict.items(), disable=None if verbose else True)):
      t.desc = f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, {k:50s}: "
      if k not in state_dict and not strict:
        if DEBUG >= 1: print(f"WARNING: not loading {k}")
        continue
      if v.shape != state_dict[k].shape:
        if {(), (1,)} == {state_dict[k].shape, v.shape}: state_dict[k] = state_dict[k].reshape(v.shape)
        else: raise ValueError(f'Shape mismatch in layer `{k}`: Expected shape {v.shape}, but found {state_dict[k].shape} in state dict.')
      if isinstance(v.device, tuple):
        if isinstance(state_dict[k].device, tuple): v.replace(state_dict[k])
        else: v.replace(state_dict[k].shard(v.device, v.uop.axis))
      else: v.replace(state_dict[k].to(v.device))
      if realize: v.realize()
      if consume: del state_dict[k]
      ret.append(v)
  return ret

@accept_filename
def zip_extract(t: Tensor) -> dict[str, Tensor]:
  files: dict[str, Tensor] = {}
  with zipfile.ZipFile(TensorIO(t), "r") as myzip:
    # sadly, the extra length needs to be read from the local header of each file.
    # this is a limitation of the zip file format
    header_contents = [t[zi.header_offset+26:zi.header_offset+30].bitcast(dtypes.uint16).to('CPU') for zi in myzip.filelist]
    Tensor.realize(*header_contents)
    for zi, header_content in zip(myzip.filelist, header_contents):
      # header_offset + sizeFileHeader + File name length + Extra field length
      file_offset = zi.header_offset + 30 + sum(cast(list[int], header_content.tolist()))
      files[zi.filename] = t[file_offset:file_offset+zi.compress_size]
      match zi.compress_type:
        case zipfile.ZIP_STORED: pass
        # TODO: we need a zlib UOp so this can be lazy
        case zipfile.ZIP_DEFLATED: files[zi.filename] = Tensor(zlib.decompress(files[zi.filename].data(), -15))
        case _: raise NotImplementedError(f"compression {zi.compress_type} not supported")
  return files

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
  storage_source: dict[str|int, Tensor] = {}
  lens: dict[str|int, int] = {}

  def _rebuild_tensor(storage, storage_offset, size, stride):
    return _rebuild_tensor_v2(storage, storage_offset, size, stride)

  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=None, backward_hooks=None, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    lens[storage[2]] = storage[4] * storage[1].itemsize
    if storage[2] not in storage_source: return None
    byte_start, byte_end = storage_offset*storage[1].itemsize, (storage_offset + prod(size))*storage[1].itemsize
    ret = storage_source[storage[2]][byte_start:byte_end].bitcast(storage[1])

    # 7 lines to deal with permuted tensors. NOTE: this currently requires reading off the disk
    shape_strides = [(s, st) for s,st in zip(size, stride) if s != 1]
    permute_indexes = [len(shape_strides)-1-y for y in argsort([x[1] for x in shape_strides])]
    if tuple(permute_indexes) != tuple(range(len(permute_indexes))):
      intermediate_shape = tuple([shape_strides[x][0] for x in argsort(permute_indexes)])
      assert tuple([shape_strides[i][1] for i in argsort(permute_indexes)]) == strides_for_shape(intermediate_shape), "nonpermutable strides"
      if DEBUG >= 3: print(f"WARNING: this torch load is slow. to permute {intermediate_shape} with {permute_indexes}")
      assert storage[1] != dtypes.bfloat16, "can't permute BF16"
      # TODO: find a nice way to support all movement ops on disktensors
      ret = ret.to(None).reshape(intermediate_shape).permute(permute_indexes)

    return ret.reshape(size)

  class Parameter:
    def __setstate__(self, state): self.tensor = state[0]

  deserialized_objects: dict[str, Any] = {}
  intercept = {"HalfStorage": dtypes.float16, "FloatStorage": dtypes.float32, "BFloat16Storage": dtypes.bfloat16,
               "IntStorage": dtypes.int32, "BoolStorage": dtypes.bool,
               "LongStorage": dtypes.int64, "_rebuild_tensor": _rebuild_tensor, "_rebuild_tensor_v2": _rebuild_tensor_v2,
               "FloatTensor": None, "Parameter": Parameter}
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
    files = zip_extract(t)
    base_name = next(iter(files)).split('/', 1)[0]
    # keyed by persistent_id in pickle file
    storage_source = {fn.split("/")[-1]: data for fn, data in files.items() if fn.startswith(f"{base_name}/data/") and not fn.endswith(".pkl")}
    return TorchPickle(io.BufferedReader(TensorIO(files[f"{base_name}/data.pkl"]), 1_000_000)).load()
  elif passthrough_reset(tarfile.is_tarfile(fobj)): # NOTE: passthrough_reset required to support python < 3.11
    files = tar_extract(t)
    f = io.BufferedReader(TensorIO(files["storages"]), 1_000_000)
    # slice source tensor t
    for _ in range(TorchPickle(f).load()):
      (key, _, storage_type), sz = TorchPickle(f).load(), struct.unpack('<q', f.read(8))[0]
      byte_offset = f.tell()
      storage_source[key] = files["storages"][byte_offset:byte_offset + sz * storage_type.itemsize]
      f.seek(sz * storage_type.itemsize, 1)
    f = io.BufferedReader(TensorIO(files["tensors"]), 1_000_000)
    # get tensor metadata
    for _ in range(TorchPickle(f).load()):
      (key, storage_id, _), ndim, _ = TorchPickle(f).load(), struct.unpack('<i', f.read(4))[0], f.read(4)
      size, stride = struct.unpack(f'<{ndim}q', f.read(8 * ndim)), struct.unpack(f'<{ndim}q', f.read(8 * ndim))
      storage_offset = struct.unpack('<q', f.read(8))[0]
      deserialized_objects[str(key)] = _rebuild_tensor_v2((None, storage_type, storage_id, None, -1), storage_offset, size, stride)
    pkl_data = TorchPickle(io.BufferedReader(TensorIO(files["pickle"]), 1_000_000)).load()
    return {k: v.tensor if isinstance(v, Parameter) else v for k, v in pkl_data.items()}
  else:
    pkl = TorchPickle(fobj)
    _, _, _, rwd, _, ids, base_offset = pkl.load(), pkl.load(), pkl.load(), fobj.tell(), pkl.load(), pkl.load(), fobj.tell()
    # slice source tensor t
    for i in ids:
      storage_source[i] = t[base_offset + 8:base_offset + 8 + lens[i]]
      base_offset += 8 + lens[i]
    fobj.seek(rwd)
    return TorchPickle(fobj).load()
