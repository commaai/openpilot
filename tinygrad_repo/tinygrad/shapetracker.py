# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools
from typing import Tuple, Union, List
from tinygrad.helpers import prod

def divmodidx(acc, d, mod=True):
  lr = f"(idx//{acc})" if acc != 1 else "idx"
  return f"({lr}%{d})" if mod else lr  # don't mod the top shape dimension

@functools.lru_cache(maxsize=None)
def to_shape_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> List[Tuple[int, int]]:
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0])]
  for i in range(1, len(shape)):
    if (strides[i] != 0 and ret[-1][1] == shape[i]*strides[i]) or (strides[i] == 0 and ret[-1][1] == 0):
      ret[-1] = (ret[-1][0] * shape[i], strides[i])
    else:
      ret.append((shape[i], strides[i]))
  return ret

class View:
  def __init__(self, shape, strides, offset:int=0):
    self.shape, self.strides, self.offset = tuple(shape), tuple(strides), offset
    self.shape_strides = to_shape_strides(self.shape, self.strides)

  def __repr__(self): return f"View<{self.shape}, {self.strides}, {self.offset}>"

  @functools.cached_property
  def contiguous(self):
    return self.offset == 0 and all(s1 == s2 or s == 1 for s,s1,s2 in zip(self.shape, self.strides, strides_for_shape(self.shape)))

  @functools.cached_property
  def expr(self):
    ret = [f"{self.offset}"] if self.offset != 0 else []
    acc = 1
    for i,(d,s) in enumerate(self.shape_strides[::-1]):
      if d != 1 and s != 0:
        lr = divmodidx(acc, d, i != len(self.shape_strides)-1 and d != prod(self.shape))
        lr = f"({lr}*{s})" if s != 1 else lr
        ret.append(lr)
      acc *= d
    return 'idx=' + ('+'.join(ret) if len(ret) > 0 else "0")

class ZeroView:
  def __init__(self, old_shape, arg):
    self.shape = []
    expr, acc = ['valid'], 1
    for s,(x,y) in list(zip(old_shape, arg))[::-1]:
      self.shape = [y-x] + self.shape
      base = divmodidx(acc, self.shape[0], len(self.shape) != len(old_shape)) + f"+{x}"
      expr += ([f"(({base}) >= 0)"] if x < 0 else []) + ([f"(({base}) < {s})"] if y > s else [])
      acc *= self.shape[0]
    self.expr = 'valid=' + ' && '.join(expr)

ViewTypes = Union[View, ZeroView]

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[int, ...]) -> Tuple[int, ...]:
  strides = [1]
  for d in shape[::-1][:-1]:
    strides = [d*strides[0]] + strides
  return tuple(strides)

@functools.lru_cache(maxsize=None)
def view_from_shape(shape:Tuple[int, ...]) -> View:
  assert all(isinstance(x, int) for x in shape) and len(shape) != 0
  return View(tuple(shape), strides_for_shape(shape))

class ShapeTracker:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]]):
    self.views : List[ViewTypes] = shape.views[:] if isinstance(shape, ShapeTracker) else [view_from_shape(shape)]
  def __repr__(self): return f"{'Complex' if len(self.views) > 1 else ''}ShapeTracker<{self.shape}, {self.views}>"

  @property
  def contiguous(self): return len(self.views) == 1 and self.views[-1].contiguous

  @property
  def shape(self): return self.views[-1].shape

  @property
  def strides(self): return self.views[-1].strides

  @property
  def offset(self): return self.views[-1].offset

  def expr(self): return ';'.join([v.expr for v in self.views[::-1] if v.expr != 'idx=idx' and v.expr != 'valid=valid'])
  def movement_op(self, op, arg):
    getattr(self, str(op).split(".")[1].lower())(*arg)
    return self
  def needs_valid(self): return any(isinstance(v, ZeroView) for v in self.views)

  # TODO: do we really need this for conv?
  # if we replace, confirm the ops taken fold into one view
  def strided(self, *arg):
    view = View([x[0] for x in arg], [x[1] for x in arg])
    if self.contiguous:
      self.views[-1] = view
    else:
      self.views.append(view)

  def reshape(self, *new_shape):
    assert all(isinstance(x, int) for x in new_shape)
    assert prod(self.shape) == prod(new_shape), f"can't reshape {self.shape} -> {new_shape}"

    # check if this is adding or removing 1s (only)
    if tuple([x for x in self.shape if x != 1]) == tuple([x for x in new_shape if x != 1]):
      old_strides = [y for x,y in zip(self.shape, self.strides) if x != 1]
      new_strides = [0 if x == 1 else old_strides.pop(0) for x in new_shape]
      self.views[-1] = View(new_shape, new_strides, self.offset)
      return

    view = View(new_shape, strides_for_shape(new_shape))
    if self.contiguous:
      self.views[-1] = view   # NOTE: if it's contiguous it can't have an offset
    else:
      self.views.append(view)

  def permute(self, *axis):
    assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis)
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    self.views[-1] = View([self.shape[a] for a in axis], [self.strides[a] for a in axis], self.offset)

  # TODO: this is a special case of slice with strides, remove it
  # though it's nice that it can't change size
  def flip(self, *axis): self.stride(*[-1 if i in axis else 1 for i in range(len((self.shape)))])

  # *** under this line are not invertible ***

  # TODO: take this functionality out of slice
  def pad(self, *arg):
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.shape)
    return self.shrink(*[(-b,s+e) for s,(b,e) in zip(self.shape, arg)])

  # TODO: take the pad functionality out of shrink
  def shrink(self, *arg):
    assert len(arg) == len(self.shape)
    offset = sum([self.strides[i]*x for i,(x,_) in enumerate(arg)])
    zeroview = ZeroView(self.shape, arg)
    self.views[-1] = View([y-x for x,y in arg], self.strides, self.offset+offset)
    if zeroview.expr != "valid=valid":
      # if we add a ZeroView, we add another (stock) view also for modding
      self.views += [zeroview, View(self.shape, strides_for_shape(self.shape))]

  def expand(self, *new_shape):
    assert all(isinstance(x, int) for x in new_shape)
    assert all(x == y or x == 1 for x,y in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
    strides = [s if x == y else 0 for s,(x,y) in zip(self.strides, zip(self.shape, new_shape))]
    self.views[-1] = View(new_shape, strides, self.offset)

  # TODO: combine with slice? this doesn't require a ZeroView, though slice shouldn't always either
  def stride(self, *mul):
    assert all(isinstance(x, int) for x in mul)
    strides = [z*m for z,m in zip(self.strides, mul)]
    new_shape = [(s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul)]
    offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
    self.views[-1] = View(new_shape, strides, self.offset + offset)

