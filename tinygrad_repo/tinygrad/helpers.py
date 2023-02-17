from collections import namedtuple
import os, math

def dedup(x): return list(dict.fromkeys(x))   # retains list order
def prod(x): return math.prod(x)
def argfix(*x): return tuple() if len(x) == 0 else tuple(x[0]) if isinstance(x[0], tuple) or isinstance(x[0], list) else tuple(x)
def argsort(x): return sorted(range(len(x)), key=x.__getitem__) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items): return all(x == items[0] for x in items) if len(items) > 0 else True
def colored(st, color): return f"\u001b[{30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color)}m{st}\u001b[0m"  # replace the termcolor library with one line
def partition(lst, fxn): return [x for x in lst if fxn(x)], [x for x in lst if not fxn(x)]
def modn(x, a): return -((-x)%a) if x < 0 else x%a

def reduce_shape(shape, axis): return tuple(1 if i in axis else shape[i] for i in range(len(shape)))
def shape_to_axis(old_shape, new_shape):
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple([i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b])

ConvArgs = namedtuple('ConvArgs', ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'sy', 'sx', 'bs', 'cout', 'py', 'py_', 'px', 'px_', 'dy', 'dx', 'out_shape'])
def get_conv_args(x_shape, w_shape, stride=1, groups=1, padding=0, dilation=1, out_shape=None):
  # TODO: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
  cout,cin,H,W = w_shape
  sy,sx = (stride, stride) if isinstance(stride, int) else stride
  if not isinstance(padding, int) and len(padding) == 4:
    px,px_,py,py_ = padding
  else:
    py,px = (padding, padding) if isinstance(padding, int) else padding
    py_, px_ = py, px
  dy,dx = (dilation, dilation) if isinstance(dilation, int) else dilation
  bs,cin_,iy,ix = x_shape

  # this can change px_ and py_ to make the out_shape right
  # TODO: copy padding names from http://nvdla.org/hw/v1/ias/unit_description.html
  if out_shape is not None:
    py_ = (out_shape[2] - 1) * sy + 1 + dy * (H-1) - iy - py
    px_ = (out_shape[3] - 1) * sx + 1 + dx * (W-1) - ix - px

  # TODO: should be easy to support asymmetric padding by changing output size
  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html describes these sizes well
  oy = (iy + py + py_ - dy * (H-1) - 1)//sy + 1
  ox = (ix + px + px_ - dx * (W-1) - 1)//sx + 1
  if cin*groups != cin_:
    raise Exception(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
  assert cout % groups == 0 and (out_shape is None or out_shape == (bs, cout, oy, ox))
  return ConvArgs(H, W, groups, cout//groups, cin, oy, ox, iy, ix, sy, sx, bs, cout, py, py_, px, px_, dy, dx, (bs, cout, oy, ox))

def get_available_llops():
  import importlib, inspect
  _buffers, DEFAULT = {}, "CPU"
  for op in [os.path.splitext(x)[0] for x in sorted(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "llops"))) if x.startswith("ops_")]:
    name = op[len("ops_"):].upper()
    DEFAULT = name if os.environ.get(name, 0) == "1" else DEFAULT
    try:
      _buffers[name] = [cls for cname, cls in inspect.getmembers(importlib.import_module('tinygrad.llops.'+op), inspect.isclass) if (cname.upper() == name + "BUFFER")][0]
    except ImportError as e:  # NOTE: this can't be put on one line due to mypy issue
      print(op, "not available", e)
  return _buffers, DEFAULT