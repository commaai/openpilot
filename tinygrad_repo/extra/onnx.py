import os
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod
from tinygrad.ops import DEBUG
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

ONNXLIMIT = int(os.getenv("ONNXLIMIT", "-1"))

def get_run_onnx(onnx_model):
  def shape_to_tuple(s): return tuple(x.dim_value for x in s.dim)
  def buffer_parse(inp):
    if inp.data_type in (1,10,7):
      ret = Tensor(np.frombuffer(inp.raw_data, dtype=TENSOR_TYPE_TO_NP_TYPE[inp.data_type]).reshape(inp.dims).astype(np.float32).copy(), requires_grad=False)
    else:
      raise Exception(f"bad data type {inp.name} {inp.dims} {inp.data_type}")
    return ret

  def attribute_parse(a):
    if a.type == 7: return tuple([int(x) for x in a.ints])
    elif a.type == 4: return buffer_parse(a.t)  # TENSOR
    elif a.type == 2: return int(a.i)
    elif a.type == 1: return float(a.f)
    else: raise Exception(f"can't parse {a.type} {a}")
  def attribute_to_dict(a): return {x.name:attribute_parse(x) for x in a}

  tensors = {}

  # get weights and biases
  for inp in onnx_model.graph.initializer:
    if len(inp.raw_data) > 0:
      tensors[inp.name] = buffer_parse(inp)
    elif len(inp.float_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
    elif len(inp.int64_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.int64_data, dtype=np.float32).reshape(inp.dims), requires_grad=False)
    else:
      print(inp.name, inp.dims, inp.data_type, len(inp.raw_data))
      print(inp)
      raise Exception("no data")
    if DEBUG >= 1:
      print("realize", inp.name)
    tensors[inp.name].realize()

  # preparse the attributes
  attribute_dict = {}
  for num,n in enumerate(onnx_model.graph.node):
    attribute_dict[num] = attribute_to_dict(n.attribute)

  # and cache them
  numpy_cache = {}
  def safe_numpy(t):
    nonlocal numpy_cache
    if t not in numpy_cache:
      if DEBUG >= 1:
        print("numpy cache miss", t)
      numpy_cache[t] = t.numpy()
    return numpy_cache[t]

  def run_onnx(inputs={}, debug=False):
    input_tensors = {}
    intermediate_tensors = {}
    output_tensor_names = [x.name for x in onnx_model.graph.output]

    # get inputs
    for inp in onnx_model.graph.input:
      if inp.name in tensors: continue
      shape = shape_to_tuple(inp.type.tensor_type.shape)
      if shape[0] == 0: shape = tuple([1]+list(shape[1:]))   # 1 batch size
      if inp.name in inputs:
        input_shape = inputs[inp.name].shape
        assert input_shape == shape, f"wrong shape for input {inp.name}, {input_shape} isn't {shape}"
        if isinstance(inputs[inp.name], Tensor):
          input_tensors[inp.name] = inputs[inp.name]
        else:
          input_tensors[inp.name] = Tensor(inputs[inp.name], requires_grad=False)
        for _,v in input_tensors.items(): v.realize()
      else:
        raise Exception(f"no data for {inp.name} with shape {shape}")

    for num,n in enumerate(onnx_model.graph.node):
      if debug: print(f"{num}: op {n.op_type}")
      inp = [tensors[x] if x in tensors else (intermediate_tensors[x] if x in intermediate_tensors else input_tensors[x]) for x in n.input]
      opt = attribute_dict[num]

      # free ones
      if n.op_type == "Relu": ret = inp[0].relu()
      elif n.op_type == "Sigmoid": ret = inp[0].sigmoid()
      elif n.op_type == "Tanh": ret = inp[0].tanh()
      elif n.op_type == "Softmax": ret = inp[0].softmax()
      elif n.op_type == "MatMul": ret = inp[0].matmul(inp[1])
      # one liners
      elif n.op_type == "Elu": ret = inp[0].elu(alpha=opt['alpha'])
      elif n.op_type == "Clip": ret = inp[0].clip(*(inp[1:] if len(inp) > 1 else (opt.get('min', -3.4e38), opt.get('max', 3.4e38))))
      elif n.op_type == "Concat": ret = inp[0].cat(*inp[1:], dim=opt['axis'])
      elif n.op_type == "Flatten": ret = inp[0].flatten(opt['axis'] if 'axis' in opt else 0)
      elif n.op_type == "Transpose": ret = inp[0].permute(order=opt['perm'])
      elif n.op_type == "Squeeze": ret = inp[0].reshape([s for i,s in enumerate(inp[0].shape) if i not in opt['axes']])
      elif n.op_type == "Unsqueeze": ret = inp[0].reshape(np.insert(inp[0].shape, opt['axes'][0], 1).tolist())
      elif n.op_type == "ReduceL2": ret = inp[0].pow(2).sum(axis=opt['axes'], keepdim=opt['keepdims']).sqrt()
      elif n.op_type == "ReduceSum": ret = inp[0].sum(axis=opt['axes'], keepdim=opt['keepdims'])
      elif n.op_type == "GlobalAveragePool": ret = inp[0].mean(axis=tuple(range(2, len(inp[0].shape))), keepdim=True)
      elif n.op_type == "Shape": ret = inp[0].shape
      elif n.op_type == "Expand": ret = inp[0].reshape([1]*(max(len(inp[0].shape), len(inp[1]))-len(inp[0].shape)) + list(inp[0].shape)) # just broadcast
      elif n.op_type == "Div": ret = inp[0].div(inp[1])
      elif n.op_type == "Constant": ret = opt['value']
      elif n.op_type == "Reshape": ret = inp[0].reshape([int(x) for x in safe_numpy(inp[1])])
      elif n.op_type == "Gather":
        # TODO: is this correct? seems to work for simple gather ops
        axis = opt['axis']
        shape = list(inp[0].shape)
        indices = [shape[axis]+int(x) if x<0 else int(x) for x in safe_numpy(inp[1])]
        args = [[(0,x) if j != axis else (i,i+1) for j, x in enumerate(shape)] for i in indices]
        ret = inp[0].slice(arg=args[0]).cat(*[inp[0].slice(arg=arg) for arg in args[1:]], dim=axis)
        ret = ret.reshape([s for i,s in enumerate(shape) if i != axis]) if len(indices) == 1 else ret # squeeze if needed
      elif n.op_type == "BatchNormalization":
        invstd = inp[4].add(opt.get('epsilon', 1e-5))**-0.5
        ret = inp[0].batchnorm(inp[1], inp[2], inp[3], invstd)
      elif n.op_type == "Gemm": ret = inp[0].linear(inp[1].transpose() if opt.get('transB', 0) == 1 else inp[1], inp[2])
      elif n.op_type == "Conv":
        x,w,b = inp if len(inp) == 3 else (inp[0], inp[1], None)
        assert 'dilations' not in opt or opt['dilations'] == (1,1)
        if opt['pads'][0] == opt['pads'][2] and opt['pads'][1] == opt['pads'][3]:
          # symmetric padding
          # TODO: is this backward?
          ret = x.conv2d(w, b, stride=opt['strides'], groups=opt.get('group', 1), padding=opt['pads'][0:2])
        else:
          x = x.pad2d((opt['pads'][0], opt['pads'][2], opt['pads'][1], opt['pads'][3]))
          ret = x.conv2d(w, b, stride=opt['strides'], groups=opt.get('group', 1))
      elif n.op_type in ["Add", "Sub", "Mul"]:
        # TODO: add this to tinygrad? i don't think it's in torch
        if len(inp[0].shape) != len(inp[1].shape) and prod(inp[0].shape) == prod(inp[1].shape):
          inp[1] = inp[1].reshape(inp[0].shape)
        # TODO: is this right?
        if 'broadcast' in opt: inp[1] = inp[1].reshape([-1 if i == opt['broadcast'] else 1 for i in range(len(inp[0].shape))])
        if n.op_type == "Add": ret = inp[0] + inp[1]
        if n.op_type == "Sub": ret = inp[0] - inp[1]
        if n.op_type == "Mul": ret = inp[0] * inp[1]
      elif n.op_type == "Split":
        i = 0
        arg = [(0,x) for x in inp[0].shape]
        for o,s in zip(n.output, opt['split']):
          arg[opt['axis']] = (i,i+s)
          intermediate_tensors[o] = inp[0].slice(arg=arg)
          i = i+s
        continue
      elif n.op_type == "AveragePool":
        assert opt['kernel_shape'] == opt['strides'] or opt['strides'] == (1,1)
        ret = inp[0].avg_pool2d(opt['kernel_shape'])
      elif n.op_type == "MaxPool":
        assert opt['kernel_shape'] == opt['strides']
        #opt['kernel_shape'] = opt['strides']
        # TODO: this is untested and probably wrong
        ret = inp[0].pad2d(opt['pads'])
        ret = ret.max_pool2d(opt['kernel_shape'])
        # strides aren't supported in max_pool
        #chan = ret.shape[1]
        #w = Tensor.eye(chan).reshape((chan, chan, 1, 1))
        #ret = ret.conv2d(w, stride=opt['strides'])
      elif n.op_type == "Slice":
        assert onnx_model.opset_import[0].version == 10
        arg = [(0,x) for x in inp[0].shape]
        starts, ends, axes = inp[1:4]
        assert axes.shape == (1,)
        axis, starts, ends  = int(safe_numpy(axes)[0]), int(safe_numpy(starts)[0]), int(safe_numpy(ends)[0])
        ends = min(ends, inp[0].shape[axis])
        starts = starts + inp[0].shape[axis] if starts < 0 else starts
        arg[axis] = (starts, ends)
        ret = inp[0].slice(arg=arg)
      else:
        print("UNSUPPORTED", n.op_type, n.input, n.output)
        raise Exception(f"op_type {n.op_type} not supported")
      assert len(n.output) == 1
      if debug: print(ret.shape)
      intermediate_tensors[n.output[0]] = ret
      #print(ret.numpy().mean())
      if num == ONNXLIMIT:
        output_tensor_names = n.output
        break

    return {outp:intermediate_tensors[outp] for outp in output_tensor_names}
  return run_onnx
