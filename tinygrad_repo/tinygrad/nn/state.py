import os, json, pathlib, zipfile, pickle
from tqdm import tqdm
from typing import Dict, Union, List, Optional, Any, Tuple
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, prod, argsort, DEBUG, Timing, GlobalCounters, CI
from tinygrad.shape.view import strides_for_shape
from tinygrad.ops import Device

safe_dtypes = {"F16": dtypes.float16, "F32": dtypes.float32, "U8": dtypes.uint8, "I8": dtypes.int8, "I32": dtypes.int32, "I64": dtypes.int64}
inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}

def safe_load_metadata(fn:Union[Tensor,str]) -> Tuple[Tensor, int, Any]:
  t = fn if isinstance(fn, Tensor) else Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")
  json_len = t[0:1].cast(dtypes.int64).numpy()[0]
  return (t, json_len, json.loads(t[8:8+json_len].numpy().tobytes()))

def safe_load(fn:Union[Tensor,str]) -> Dict[str, Tensor]:
  t, json_len, metadata = safe_load_metadata(fn)
  return {k:t[8+json_len+v['data_offsets'][0]:].cast(safe_dtypes[v['dtype']])[:prod(v['shape'])].reshape(v['shape']) for k,v in metadata.items() if k != "__metadata__"}

def safe_save(tensors:Dict[str, Tensor], fn:str, metadata:Optional[Dict[str, Any]]=None):
  headers, offset = {}, 0
  if metadata: headers['__metadata__'] = metadata
  for k,v in tensors.items():
    headers[k] = {'dtype': inverse_safe_dtypes[v.dtype], 'shape': list(v.shape), 'data_offsets':[offset, offset+v.nbytes()]}
    offset += v.nbytes()
  j = json.dumps(headers, separators=(',', ':'))
  j += "\x20"*((8-len(j)%8)%8)
  pathlib.Path(fn).unlink(missing_ok=True)
  t = Tensor.empty(8+len(j)+offset, dtype=dtypes.uint8, device=f"disk:{fn}")
  t[0:1].cast(dtypes.int64).assign([len(j)])
  t[8:8+len(j)].assign(Tensor(list(j.encode('utf-8')), dtype=dtypes.uint8, device="cpu"))
  for k,v in safe_load(t).items(): v.assign(tensors[k])

# state dict

from collections import OrderedDict
def get_state_dict(obj, prefix:str='', tensor_type=Tensor) -> Dict[str, Tensor]:
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
def get_parameters(obj) -> List[Tensor]: return list(get_state_dict(obj).values())

def load_state_dict(model, state_dict, strict=True, verbose=True):
  with Timing("loaded weights in ", lambda et_ns: f", {GlobalCounters.mem_used/1e9:.2f} GB loaded at {GlobalCounters.mem_used/et_ns:.2f} GB/s"):
    model_state_dict = get_state_dict(model)
    if DEBUG >= 1 and len(state_dict) > len(model_state_dict): print("WARNING: unused weights in state_dict", sorted(list(state_dict.keys() - model_state_dict.keys())))
    for k,v in (t := tqdm(model_state_dict.items(), disable=CI or not verbose)):
      t.set_description(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, {k:50s}")
      if k not in state_dict and not strict:
        if DEBUG >= 1: print(f"WARNING: not loading {k}")
        continue
      v.assign(state_dict[k].to(v.device)).realize()

# torch support!

def torch_load(fn:str):
  t = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")

  offsets: Dict[str, int] = {}
  lens: Dict[str, int] = {}
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    lens[storage[2]] = storage[4] * storage[1].itemsize
    if storage[2] not in offsets: return None
    byte_offset = offsets[storage[2]]+storage_offset*storage[1].itemsize
    ret = t[byte_offset:byte_offset+prod(size)].cast(storage[1])
    # convert bfloat16 -> float16 using LLVM for Llama 2
    # upstream LLaMA also does this conversion:
    # https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L95
    # TODO: should this be done in the example instead? or maybe we don't need this anymore with better bfloat16 support
    if storage[1] == dtypes.bfloat16:
      ret = ret.bitcast(dtypes.uint16).to("CPU").cast(dtypes.uint32).mul(1<<16).bitcast(dtypes.float32).to(Device.DEFAULT).half()
      #ret = ret.to("LLVM").half().to(Device.DEFAULT)

    # 7 lines to deal with permuted tensors. NOTE: this currently requires reading off the disk
    shape_strides = [(s, st) for s,st in zip(size, stride) if s != 1]
    permute_indexes = [len(shape_strides)-1-y for y in argsort([x[1] for x in shape_strides])]
    if tuple(permute_indexes) != tuple(range(len(permute_indexes))):
      intermediate_shape = tuple([shape_strides[x][0] for x in argsort(permute_indexes)])
      assert tuple([shape_strides[i][1] for i in argsort(permute_indexes)]) == strides_for_shape(intermediate_shape), "nonpermutable strides"
      if DEBUG >= 2: print(f"WARNING: this torch load is slow. CPU to permute {intermediate_shape} with {permute_indexes}")
      # TODO: find a nice way to support all shapetracker on disktensors
      ret = ret.cpu().reshape(intermediate_shape).permute(permute_indexes)

    return ret.reshape(size)

  intercept = {"HalfStorage": dtypes.float16, "FloatStorage": dtypes.float32, "BFloat16Storage": dtypes.bfloat16, "IntStorage": dtypes.int32, "LongStorage": dtypes.int64, "_rebuild_tensor_v2": _rebuild_tensor_v2}
  whitelist = {"torch", "collections", "numpy", "_codecs"}  # NOTE: this is not for security, only speed
  class Dummy: pass
  class TorchPickle(pickle.Unpickler):
    def find_class(self, module, name):
      module_root = module.split(".")[0]
      if module_root not in whitelist:
        if DEBUG >= 2: print(f"WARNING: returning Dummy for {module} {name}")
        return Dummy
      return intercept[name] if module_root == "torch" else super().find_class(module, name)
    def persistent_load(self, pid): return pid

  if tuple(t[0:2].numpy()) == (0x50, 0x4b):
    myzip = zipfile.ZipFile(fn, 'r')
    base_name = myzip.namelist()[0].split('/', 1)[0]
    for n in myzip.namelist():
      if n.startswith(f'{base_name}/data/'):
        with myzip.open(n) as myfile:
          offsets[n.split("/")[-1]] = myfile._orig_compress_start # type: ignore
    with myzip.open(f'{base_name}/data.pkl') as myfile:
      return TorchPickle(myfile).load()
  else:
    with open(fn, "rb") as f:
      pkl = TorchPickle(f)
      _, _, _, rwd, _, ids, base_offset = pkl.load(), pkl.load(), pkl.load(), f.tell(), pkl.load(), pkl.load(), f.tell()
      for i in ids:
        offsets[i] = base_offset + 8
        base_offset += 8 + lens[i]
      f.seek(rwd)
      return TorchPickle(f).load()
