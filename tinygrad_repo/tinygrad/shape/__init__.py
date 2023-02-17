# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import os
import functools
from typing import Tuple, Union, List, Optional
from tinygrad.helpers import prod
from tinygrad.shape.symbolic import Variable

# TODO: fix DEBUG import
DEBUG = int(os.getenv("DEBUG", "0"))

@functools.lru_cache(maxsize=None)
def to_shape_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> List[Tuple[int, int]]:
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0])] if len(shape) > 0 else []
  for i in range(1, len(shape)):
    if (strides[i] != 0 and ret[-1][1] == shape[i]*strides[i]) or ret[-1][0] == 1 or (strides[i] == 0 and ret[-1][1] == 0):
      ret[-1] = (ret[-1][0] * shape[i], strides[i])
    else:
      ret.append((shape[i], strides[i]))
  return ret

class View:
  def __init__(self, shape, strides, offset:int=0):
    self.shape, self.strides, self.offset = tuple(shape), tuple(strides), offset
    self.shape_strides = to_shape_strides(self.shape, self.strides)

  def __repr__(self): return f"View({self.shape}, {self.strides}, {self.offset})"

  @functools.cached_property
  def contiguous(self):
    return self.offset == 0 and all(s1 == s2 or s == 1 for s,s1,s2 in zip(self.shape, self.strides, strides_for_shape(self.shape)))

  def expr_node(self, idx):
    ret = [Variable.num(self.offset)]
    acc = 1
    for d,s in self.shape_strides[::-1]:
      if d != 1 and s != 0:
        ret.append(((idx//acc)%d)*s)
      acc *= d
    return Variable.sum(ret)

  @functools.cached_property
  def expr(self):
    return 'idx=' + str(self.expr_node(Variable('idx', 0, prod([x[0] for x in self.shape_strides])-1)))

  # generate an expression if you have a variable or expression for each index
  def expr_idxs(self, idxs):
    return Variable.sum([Variable.num(self.offset)] + [Variable(idxs[i], 0, sh-1)*st for i,(sh,st) in enumerate(zip(self.shape, self.strides)) if sh != 1 and st != 0])

class ZeroView:
  def __init__(self, old_shape, arg):
    self.old_shape, self.arg, self.shape = old_shape, arg, []

  def expr_node(self, valid, idx):
    expr, acc = [valid] if valid is not None else [], 1
    for s,(x,y) in list(zip(self.old_shape, self.arg))[::-1]:
      self.shape = [y-x] + self.shape
      base = idx//acc
      base = (base % self.shape[0]) + x
      expr += ([base >= 0] if x < 0 else []) + ([base < s] if y > s else [])
      acc *= self.shape[0]
    return Variable.ands(expr)

  @functools.cached_property
  def expr(self):
    max_idx = prod([y-x for x,y in self.arg])
    return 'valid=' + str(self.expr_node(Variable('valid', 0, 1), Variable('idx', 0, max_idx-1)))

  def __repr__(self): return f"ZeroView({self.old_shape}, {self.arg})"

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
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], views:Optional[List[ViewTypes]]=None):
    self.views : List[ViewTypes] = views if views is not None else (shape.views[:] if isinstance(shape, ShapeTracker) else [view_from_shape(shape)])
  def __repr__(self): return f"ShapeTracker(shape={self.shape}, views={self.views})"

  @property
  def contiguous(self): return len(self.views) == 1 and self.views[-1].contiguous

  @property
  def shape(self): return self.views[-1].shape

  @property
  def strides(self): return self.views[-1].strides

  @property
  def offset(self): return self.views[-1].offset

  def expr_node(self):
    idx = Variable('idx', 0, prod(self.shape)-1)
    valid = None #Variable.num(1)
    for v in self.views[::-1]:
      if isinstance(v, ZeroView): valid = v.expr_node(valid, idx)
      else: idx = v.expr_node(idx)
    return idx, valid
  
  def expr(self):
    idx, valid = self.expr_node()
    if valid is not None and str(valid) != "valid": return f"valid={valid};idx={idx}"
    else: return f"idx={idx}"

  #def expr(self): return ';'.join([v.expr for v in self.views[::-1] if v.expr != 'idx=idx' and v.expr != 'valid=valid'])
  def movement_op(self, op, arg): return getattr(self, str(op).split(".")[1].lower())(*arg)
  def needs_valid(self): return any(isinstance(v, ZeroView) for v in self.views)

  # TODO: do we really need this for conv?
  # if we replace, confirm the ops taken fold into one view
  def strided(self, *arg):
    view = View([x[0] for x in arg], [x[1] for x in arg])
    # TODO: this does not always require a new view if non contiguous
    if self.views[-1].contiguous:
      self.views[-1] = view
    else:
      self.views.append(view)
    return self

  def reshape(self, *new_shape):
    assert all(isinstance(x, int) and x != 0 for x in new_shape), f"shape must be ints and can't contain 0 {new_shape}"
    assert prod(self.shape) == prod(new_shape), f"can't reshape {self.shape} -> {new_shape}"

    # check if this is adding or removing 1s (only)
    if tuple([x for x in self.shape if x != 1]) == tuple([x for x in new_shape if x != 1]):
      old_strides = [y for x,y in zip(self.shape, self.strides) if x != 1]
      new_strides = [0 if x == 1 else old_strides.pop(0) for x in new_shape]
      self.views[-1] = View(new_shape, new_strides, self.offset)
      return self
    
    # check if the new dimensions factorize from the old ones
    # NOTE: if you don't make a copy here, the list is popped in the lrucache
    min_shape_strides = to_shape_strides(self.shape, self.strides)[:]
    curr_dim, curr_stride = min_shape_strides.pop(0)
    new_strides = []
    for s in new_shape:
      if curr_dim%s == 0:
        curr_dim //= s
        new_strides.append(curr_stride * curr_dim)
        if curr_dim == 1:
          if len(min_shape_strides) == 0:
            # there might still be 1s in the shape
            while len(new_strides) != len(new_shape):
              assert new_shape[len(new_strides)] == 1
              new_strides.append(1)
            break
          curr_dim, curr_stride = min_shape_strides.pop(0)
      else:
        break   # didn't factorize

    if len(new_shape) == len(new_strides):
      self.views[-1] = View(new_shape, new_strides, self.offset)
      return self

    view = View(new_shape, strides_for_shape(new_shape))
    if self.contiguous:
      self.views[-1] = view   # NOTE: if it's contiguous it can't have an offset
    else:
      if DEBUG >= 2:
        print(f"WARNING: reshape from {self.shape} w strides {self.strides} -> {new_shape} is creating another view")
      self.views.append(view)
    return self

  def permute(self, *axis):
    assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis), f"invalid permute {axis} for {self.shape}"
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    self.views[-1] = View([self.shape[a] for a in axis], [self.strides[a] for a in axis], self.offset)
    return self

  # TODO: this is a special case of slice with strides, remove it
  # though it's nice that it can't change size
  def flip(self, *axis): return self.stride(*[-1 if i in axis else 1 for i in range(len((self.shape)))])

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
    return self

  def expand(self, *new_shape):
    assert all(isinstance(x, int) for x in new_shape)
    assert all(x == y or x == 1 for x,y in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
    strides = [s if x == y else 0 for s,(x,y) in zip(self.strides, zip(self.shape, new_shape))]
    self.views[-1] = View(new_shape, strides, self.offset)
    return self

  # TODO: combine with slice? this doesn't require a ZeroView, though slice shouldn't always either
  def stride(self, *mul):
    assert all(isinstance(x, int) for x in mul)
    strides = [z*m for z,m in zip(self.strides, mul)]
    new_shape = [(s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul)]
    offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
    self.views[-1] = View(new_shape, strides, self.offset + offset)
    return self
