from __future__ import annotations
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
import importlib
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod, getenv, DEBUG, dtypes
from typing import List,Dict
from onnx.onnx_pb import AttributeProto, ModelProto, TensorProto, TypeProto
try:
  from onnx.helper import tensor_dtype_to_np_dtype
except ImportError:
  # for onnx < 1.13
  from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
  tensor_dtype_to_np_dtype = lambda x: TENSOR_TYPE_TO_NP_TYPE[x]

# global numpy cache for parameters
numpy_cache = {}
def safe_numpy(t) -> np.ndarray:
  if not isinstance(t, Tensor): return t
  global numpy_cache
  if t not in numpy_cache:
    if DEBUG >= 3: print("numpy cache miss", t)
    tmp = t.numpy()
    numpy_cache[t] = tmp if len(tmp.shape) else tmp.reshape(1)
  assert len(numpy_cache[t].shape) > 0
  return numpy_cache[t]

onnx_ops = importlib.import_module('extra.onnx_ops')

ONNXLIMIT = getenv("ONNXLIMIT", -1)

def get_run_onnx(onnx_model: ModelProto):
  def type_parse(type_proto: TypeProto):
    ret = []
    while True:
      attr = type_proto.WhichOneof('value')
      if attr == 'tensor_type':
        if "dim_value" not in getattr(type_proto, attr).shape.dim.__dir__(): return () # variable type, unable to determine shape
        elif not ret:
          return tuple([x.dim_value for x in getattr(type_proto, attr).shape.dim])
        else:
          ret.extend([(x.dim_value,) for x in getattr(type_proto, attr).shape.dim])
          return tuple(ret)
      elif attr == 'sequence_type':
        type_proto = getattr(type_proto, attr).elem_type
        ret.append(1)
      elif attr == 'map_type': raise NotImplementedError(f"map_type is not implemented: {type_proto}")
      elif attr == 'opaque_type': raise NotImplementedError(f"opaque_type is not implemented: {type_proto}")
      elif attr == 'sparse_tensor_type': raise NotImplementedError(f"sparse_tensor_type is not implemented: {type_proto}")
      elif attr == 'optional_type': type_proto = getattr(type_proto, attr).elem_type
      else: raise Exception(f"unknown attr: {attr}, {type_proto}")

  def buffer_parse(inp: TensorProto) -> Tensor:
    if inp.data_type in (1,10,6,7):
      # TODO: this is shared with below
      if len(inp.float_data) > 0:
        ret = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
      elif len(inp.int64_data) > 0:
        ret = Tensor(np.array(inp.int64_data, dtype=np.int64).reshape(inp.dims), requires_grad=False)
      elif len(inp.int32_data) > 0:
        ret = Tensor(np.array(inp.int32_data, dtype=np.int32).reshape(inp.dims), requires_grad=False)
      else:
        ret = Tensor(np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).reshape(inp.dims).astype(np.float32).copy(), requires_grad=False)
    else:
      raise Exception(f"bad data type {inp.name} {inp.dims} {inp.data_type}")
    return ret

  def attribute_parse(a: AttributeProto) -> float | int | str | Tensor | tuple[float] | tuple[int]:
    # TODO: this is not complete, see onnx/onnx_ml_pb2.pyi for a complete list
    if a.type == AttributeProto.FLOAT: return float(a.f)
    elif a.type == AttributeProto.INT: return int(a.i)
    elif a.type == AttributeProto.STRING: return a.s.decode("utf-8")
    elif a.type == AttributeProto.TENSOR: return buffer_parse(a.t) # TENSOR
    elif a.type == AttributeProto.FLOATS: return tuple(float(x) for x in a.floats)
    elif a.type == AttributeProto.INTS: return tuple(int(x) for x in a.ints)
    elif a.type == AttributeProto.STRINGS: return tuple(x.decode("utf-8") for x in a.strings)
    elif a.type == AttributeProto.GRAPH: raise Exception(f"graph not implemented: {a.g}")
    else: raise Exception(f"can't parse {a.type} {a}")
  def attribute_to_dict(a: RepeatedCompositeFieldContainer[AttributeProto]): return {x.name:attribute_parse(x) for x in a}

  tensors: Dict[str, Tensor] = {}

  # get weights and biases
  for inp in onnx_model.graph.initializer:
    if len(inp.raw_data) > 0:
      tensors[inp.name] = buffer_parse(inp)
    elif len(inp.float_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
    elif len(inp.int64_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.int64_data, dtype=np.int64).reshape(inp.dims), requires_grad=False)
    elif len(inp.raw_data) == 0:
      tensors[inp.name] = Tensor(np.array([], dtype=np.float32), requires_grad=False)
    else:
      print(inp.name, inp.dims, inp.data_type, len(inp.raw_data))
      print(inp)
      raise Exception("no data")

  # preparse the attributes
  attribute_dict = {}
  domain = ""
  for num,n in enumerate(onnx_model.graph.node):
    attribute_dict[num] = attribute_to_dict(n.attribute)
    if n.domain: domain = n.domain

  onnx_model_version = onnx_model.opset_import[0].version

  def run_onnx(inputs={}, debug=0):
    debug = getenv("DEBUGONNX") or debug
    input_tensors: Dict[str,Tensor] = {}
    intermediate_tensors: Dict[str,Tensor] = {}
    output_tensor_names = [x.name for x in onnx_model.graph.output]

    # get inputs
    for inp in onnx_model.graph.input:
      if inp.name in tensors: continue
      shape = type_parse(inp.type)
      if inp.name in inputs:
        if isinstance(inputs[inp.name], Tensor):
          input_tensors[inp.name] = inputs[inp.name]
        elif isinstance(inputs[inp.name], list):
          input_tensors[inp.name] = [Tensor(i, requires_grad=False) for i in inputs[inp.name]]
        elif domain == "ai.onnx.preview.training": # not sure if in real use the domain is "ai.onnx.preview.training"
          input_tensors[inp.name] = Tensor(inputs[inp.name], requires_grad=True) # TODO there isn't a good way to parse which inp requires_grad, some are manually turned off in optimizer ops
        else:
          input_tensors[inp.name] = Tensor(inputs[inp.name], requires_grad=False)
        if shape: # if only input_tensor is not variable type
          input_shape = input_tensors[inp.name].shape if isinstance(input_tensors[inp.name], Tensor) else (1, *[i.shape for i in input_tensors[inp.name]])
          assert input_shape == shape, f"wrong shape for input {inp.name}, {input_shape} isn't {shape}"
      else:
        raise Exception(f"no data for {inp.name} with shape {shape}")

    def fetch_tensor(x: str):
      if x in tensors: return tensors[x]
      if x in intermediate_tensors: return intermediate_tensors[x]
      if x != str(): return input_tensors[x]
      return None

    for num,n in enumerate(onnx_model.graph.node):
      inp: List[Tensor] = []
      if debug >= 3: print("inputs:")
      for x in n.input:
        t = fetch_tensor(x)
        if debug >= 3: print(f"\t{x} - {t}")
        inp.append(t)
      opt: Dict = attribute_dict[num]
      if debug >= 1: print(f"{num}: op {n.op_type} shape {[x.shape if isinstance(x, Tensor) else x for x in inp]} opt {opt}")
      # some ops live here because they require some local variables
      if n.op_type == "Split": # have to use n.output for cases when num_outputs is absent
        axis = opt.get("axis", 0)
        split = None if len(inp) == 1 else [int(x) for x in safe_numpy(inp[1])]
        if split is None:
          split = [inp[0].shape[axis] // len(n.output)] * len(n.output)
          for i in range(inp[0].shape[axis] % len(n.output)):
            split[i] += 1
        i, ret = 0, []
        arg = [(0,x) for x in inp[0].shape]
        for s in split:
          arg[axis] = (i,i+s)
          ret.append(inp[0].shrink(arg=tuple(arg)))
          i = i+s
        ret = tuple(ret)
      elif n.op_type == "Slice": # need to check onnx_model_version
        if onnx_model_version < 10:
          axes, ends, starts, steps = list(opt.get("axes", range(inp[0].ndim))), list(opt["ends"]), list(opt["starts"]), [1]*inp[0].ndim
        else:
          starts, ends = inp[1:3]
          axes = safe_numpy(Tensor.arange(inp[0].ndim, dtype=dtypes.int32) if len(inp) <= 3 else inp[3]).tolist()
          steps = safe_numpy(inp[4]) if len(inp) > 4 else [1]*inp[0].ndim
          starts, ends = safe_numpy(starts.ceil().cast(dtypes.int32)).tolist(), safe_numpy(ends.ceil().cast(dtypes.int32)).tolist()
        arg = [(0,x,1) for x in inp[0].shape]
        for i, axis in enumerate(axes):
          axis = int(axis) + inp[0].ndim if axis < 0 else int(axis)
          starts[i], ends[i] = starts[i] + inp[0].shape[axis] if starts[i] < 0 else starts[i], ends[i] + inp[0].shape[axis] if ends[i] < 0 else ends[i]
          starts[i], ends[i] = max(0, min(starts[i], inp[0].shape[axis])), max(0, min(ends[i], inp[0].shape[axis]))
          if starts[i] > ends[i] and steps[i] >= 0: steps[i] = -steps[i]
          arg[axis] = (starts[i], ends[i], steps[i])
        new_shape = tuple((s, e) if st > 0 else (e+1, s+1) for s, e, st in arg)
        if any(s==e for s,e in new_shape): ret = inp[0].shrink(new_shape)
        else: ret = inp[0].__getitem__(tuple([slice(s,e,int(st)) for s,e,st in arg]))
      elif n.op_type == "Gradient": # need to call backward on intermediate_tensors
        assert len(opt["xs"]) == len(inp), f"len(opt['xs']):{len(opt['xs'])}, len(inp):{len(inp)} output and input has to match"
        y = opt["y"]
        intermediate_tensors[y].backward()
        ret = tuple([t.grad for t in inp])
      elif hasattr(onnx_ops, n.op_type):
        fxn = getattr(onnx_ops, n.op_type)
        if isinstance(fxn, dict):
          for k in sorted(fxn.keys()):
            if k <= onnx_model_version:
              real_fxn = fxn[k]
        else:
          real_fxn = fxn
        ret = real_fxn(*inp, **opt)
      else:
        print("UNSUPPORTED", n.op_type, n.input, n.output)
        raise Exception(f"op_type {n.op_type} not supported")
      if not isinstance(ret, tuple): ret = (ret, )
      assert len(n.output) <= len(ret), f"expected output size must be less than {len(ret)}, it's {n.output}"
      if debug >= 2: print([x.shape if isinstance(x, Tensor) else None for x in ret])
      if debug >= 2: print("outputs:")
      for i in range(len(n.output)):
        if debug >= 2: print(f"\t{n.output[i]} - {ret[i]}")
        intermediate_tensors[n.output[i]] = ret[i]
      if num == ONNXLIMIT:
        output_tensor_names = n.output
        break

    return {outp:intermediate_tensors[outp] for outp in output_tensor_names}
  return run_onnx
