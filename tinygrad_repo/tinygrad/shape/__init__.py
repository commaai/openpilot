# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools
from typing import Tuple, Union, List, Optional, cast
from tinygrad.helpers import prod, DEBUG
from tinygrad.shape.symbolic import Variable, MulNode, NumNode, Node

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
  def __init__(self, shape:Tuple[int, ...], strides:Tuple[int, ...], offset:int=0):
    self.shape, self.strides, self.offset = shape, tuple(stride if shp != 1 else 0 for stride,shp in zip(strides, shape)), offset
    self.shape_strides = to_shape_strides(self.shape, self.strides)
    self.contiguous : bool = self.offset == 0 and all(s1 == s2 or s == 1 for s,s1,s2 in zip(self.shape, self.strides, strides_for_shape(self.shape)))

  def __repr__(self): return f"View({self.shape}, {self.strides}, {self.offset})"

  def expr_node(self, idx=None, offset:Union[Node, int]=0):
    if idx is None: idx = Variable('idx', 0, prod(self.shape))
    ret = [Variable.num(self.offset)+offset]
    acc = 1
    for d,s in self.shape_strides[::-1]:
      ret.append(((idx//acc)%d)*s)
      acc *= d
    return Variable.sum(ret)

  # generate an expression if you have a variable or expression for each index
  def expr_idxs(self, idxs, offset:Union[Node, int]=0):
    return Variable.sum([Variable.num(self.offset)+offset] + [Variable(idx, 0, sh-1)*st for idx,sh,st in zip(idxs, self.shape, self.strides) if sh != 1 and st != 0])

class ZeroView:
  def __init__(self, old_shape:Tuple[int, ...], arg):
    self.old_shape, self.arg = old_shape, arg
    self.shape : Tuple[int, ...] = tuple([y-x for x,y in self.arg])
    # fake properties
    self.strides, self.contiguous, self.offset = strides_for_shape(self.shape), False, 0

  def __repr__(self): return f"ZeroView({self.old_shape}, {self.arg})"

  def expr_node(self, idx=None, valid=None):
    if idx is None: idx = Variable('idx', 0, prod([y-x for x,y in self.arg]))
    expr, acc = [valid] if valid is not None else [], 1
    for s,ns,(x,y) in list(zip(self.old_shape, self.shape, self.arg))[::-1]:
      base = ((idx//acc) % ns) + x
      expr += ([base >= 0] if x < 0 else []) + ([base < s] if y > s else [])
      acc *= ns
    return Variable.ands(expr)

  def expr_idxs(self, idxs, offset=0): raise NotImplementedError("ZeroView doesn't support expr_idxs")

ViewTypes = Union[View, ZeroView]

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[int, ...]) -> Tuple[int, ...]:
  strides = [1]
  for d in shape[::-1][:-1]: strides = [d*strides[0]] + strides
  return tuple(st if s != 1 else 0 for st, s in zip(strides, shape))

@functools.lru_cache(maxsize=None)
def view_from_shape(shape:Tuple[int, ...]) -> View:
  assert all(isinstance(x, int) for x in shape) and len(shape) != 0
  return View(tuple(shape), strides_for_shape(shape))

def merge_views(vm2:View, vm1:View) -> Optional[View]:
  new_strides, new_offset = [], vm2.expr_node(Variable.num(vm1.offset))
  assert isinstance(new_offset, NumNode), "new_offset wasn't a number?!?"
  for s,st in zip(vm1.shape, vm1.strides):
    this_dim = View(vm2.shape, vm2.strides).expr_node(Variable('idx', 0, s-1)*st)
    if s == 1:
      new_strides.append(0)   # all shape 1 can have stride 0
    elif isinstance(this_dim, NumNode) and this_dim.b == 0:
      new_strides.append(0)
    elif isinstance(this_dim, Variable):
      new_strides.append(1)
    elif isinstance(this_dim, MulNode) and isinstance(this_dim.a, Variable):
      new_strides.append(this_dim.b)
    else:
      if DEBUG >= 4: print("can't simplify", s, this_dim.render())
      break
  return View(vm1.shape, tuple(new_strides), new_offset.b) if len(new_strides) == len(vm1.strides) else None

class ShapeTracker:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], views:Optional[List[ViewTypes]]=None):
    self.views : List[ViewTypes] = views if views is not None else (shape.views[:] if isinstance(shape, ShapeTracker) else [view_from_shape(shape)])
  def __repr__(self): return f"ShapeTracker(shape={self.shape}, views={self.views})"
  def copy(self) -> ShapeTracker: return ShapeTracker(self.shape, self.views[:])

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[-1].contiguous

  @property
  def shape(self) -> Tuple[int, ...]: return self.views[-1].shape

  @property
  def strides(self) -> Tuple[int, ...]: return self.views[-1].strides

  @property
  def offset(self) -> int: return self.views[-1].offset

  # this is the real size
  def size(self): return prod([s for s,st in zip(self.shape, self.strides) if st != 0])

  def _expr_idx(self, idx):
    valid = Variable.num(1)
    for v in self.views[0:-1][::-1]:
      if isinstance(v, ZeroView): valid = v.expr_node(idx, valid)
      else: idx = v.expr_node(idx)
    return idx, valid

  def simplify(self):
    if len(self.views) >= 2 and isinstance(self.views[-2], View) and isinstance(self.views[-1], View):
      new_view = merge_views(self.views[-2], self.views[-1])
      if new_view:
        if DEBUG >= 4: print(f"st simplify : {self.views[-2]} + {self.views[-1]} = {new_view}")
        self.views = self.views[:-2] + [new_view]
        self.simplify()

  # TODO: arg order is reversed here
  def expr_idxs(self, offset=0, idxs=None):
    if idxs is None: idxs = [f"idx{i}" for i in range(len(self.shape))]
    return self._expr_idx(self.views[-1].expr_idxs(idxs, offset))

  def expr_node(self, idx='idx', offset=0):
    return self._expr_idx(self.views[-1].expr_node(Variable(idx, 0, prod(self.shape)-1), offset))

  def movement_op(self, op, arg:Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]]) -> ShapeTracker:
    return getattr(self, str(op).split(".")[1].lower())(arg)
  def needs_valid(self) -> bool:
    return any(isinstance(v, ZeroView) for v in self.views)

  def reshape(self, new_shape : Tuple[int, ...]) -> ShapeTracker:
    assert isinstance(new_shape, tuple)
    if self.shape == new_shape: return self
    assert all(isinstance(x, int) and x != 0 for x in new_shape), f"shape must be ints and can't contain 0 {new_shape}"
    assert prod(self.shape) == prod(new_shape), f"can't reshape {self.shape} -> {new_shape}"

    view = View(new_shape, strides_for_shape(new_shape))
    if self.contiguous: self.views[-1] = view   # NOTE: if it's contiguous it can't have an offset
    else:
      # NOTE: the last view in self.views is never a ZeroView
      if (merged_view := merge_views(cast(View, self.views[-1]), view)) is not None: self.views[-1] = merged_view
      else: self.views.append(view)
    return self

  def permute(self, axis : Tuple[int, ...]) -> ShapeTracker:
    assert isinstance(axis, tuple)
    assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis), f"invalid permute {axis} for {self.shape}"
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    self.views[-1] = View(tuple(self.shape[a] for a in axis), tuple(self.strides[a] for a in axis), self.offset)
    return self

  # TODO: this is a special case of slice with strides, remove it
  # though it's nice that it can't change size
  def flip(self, axis : Tuple[int, ...]) -> ShapeTracker:
    return self.stride(tuple(-1 if i in axis else 1 for i in range(len((self.shape)))))

  # *** under this line are not invertible ***

  # TODO: take this functionality out of slice
  def pad(self, arg : Tuple[Tuple[int, int], ...]) -> ShapeTracker:
    assert isinstance(arg, tuple)
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.shape)
    return self.shrink(tuple((-b,s+e) for s,(b,e) in zip(self.shape, arg)))

  # TODO: take the pad functionality out of shrink
  def shrink(self, arg : Tuple[Tuple[int, int], ...]) -> ShapeTracker:
    assert isinstance(arg, tuple)
    assert len(arg) == len(self.shape)
    offset = sum([self.strides[i]*x for i,(x,_) in enumerate(arg)])
    zeroview = ZeroView(self.shape, arg)
    self.views[-1] = View(tuple(y-x for x,y in arg), self.strides, self.offset+offset)
    if zeroview.expr_node().min == 0:  # may be invalid
      # if we add a ZeroView, we add another (stock) view also for modding
      self.views += [zeroview, View(self.shape, strides_for_shape(self.shape))]
    return self

  def expand(self, new_shape : Tuple[int, ...]) -> ShapeTracker:
    assert isinstance(new_shape, tuple)
    assert all(isinstance(x, int) for x in new_shape), f"non ints for expand in {new_shape}"
    assert all(x == y or x == 1 for x,y in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
    strides : Tuple[int, ...] = tuple(s if x == y else 0 for s,(x,y) in zip(self.strides, zip(self.shape, new_shape)))
    self.views[-1] = View(new_shape, strides, self.offset)
    return self

  # TODO: combine with slice? this doesn't require a ZeroView, though slice shouldn't always either
  def stride(self, mul : Tuple[int, ...]) -> ShapeTracker:
    assert isinstance(mul, tuple)
    assert all(isinstance(x, int) for x in mul)
    strides = tuple(z*m for z,m in zip(self.strides, mul))
    new_shape = tuple((s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul))
    offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
    self.views[-1] = View(new_shape, strides, self.offset + offset)
    return self

# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
def get_contraction(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]):
  if len(new_shape) > len(old_shape): return None
  new_shape_i : int = 0
  shape_idx_groups : List[List[int]] = [[] for _ in range(len(new_shape))]
  for old_shape_i, t in enumerate(old_shape):
    if new_shape[new_shape_i] % t != 0 or prod([old_shape[x] for x in shape_idx_groups[new_shape_i]]) * t > new_shape[new_shape_i]:
      return None
    shape_idx_groups[new_shape_i].append(old_shape_i)
    if prod([old_shape[x] for x in shape_idx_groups[new_shape_i]]) == new_shape[new_shape_i] and new_shape_i < len(new_shape) - 1:
      new_shape_i += 1
  return shape_idx_groups
