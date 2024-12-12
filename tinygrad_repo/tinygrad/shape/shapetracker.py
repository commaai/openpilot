# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from dataclasses import dataclass
import functools
from typing import Tuple, List, Optional, Dict, Set
from tinygrad.helpers import merge_dicts, getenv
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, Ops, graph_rewrite, split_uop, symbolic_flat, Variable, sint, uop_given_valid, simplify_valid

@functools.lru_cache(None)
def views_to_indexed_uops(views: Tuple[View, ...], _idxs:Optional[Tuple[UOp, ...]]=None) -> Tuple[UOp, UOp]:
  idx, valid = views[-1].to_indexed_uops(_idxs)
  for view in reversed(views[0:-1]):
    view = view.minify()
    acc, idxs = 1, []
    for d in reversed(view.shape):
      idxs.append((idx//acc)%d)
      acc *= d
    idx, valid = view.to_indexed_uops(idxs[::-1], valid)
  return idx, valid

@functools.lru_cache(None)
def views_to_real_strides(views: Tuple[View, ...], ignore_valid=False) -> Tuple[Optional[sint], ...]:
  # NOTE: if a stride is not always valid, it will be None
  if len(views) == 1 and views[-1].mask is None: return views[-1].strides
  ret: List[Optional[sint]] = [None] * len(views[-1].shape)
  idx, valid = (graph_rewrite(u, symbolic_flat) for u in views_to_indexed_uops(views))
  # TODO: always apply these in to_indexed_uops?
  if (newvalid:=simplify_valid(valid)) is not None: valid = newvalid
  if (newidx:=uop_given_valid(valid, idx)) is not None: idx = graph_rewrite(newidx, symbolic_flat)
  for c in split_uop(idx, Ops.ADD):
    if c.op is Ops.RANGE: ret[c.arg] = 1
    if c.op is Ops.MUL and c.src[0].op is Ops.RANGE and c.src[1].op is Ops.CONST: ret[c.src[0].arg] = c.src[1].arg
    if c.op is Ops.MUL and c.src[1].op is Ops.RANGE and c.src[0].op is Ops.CONST: ret[c.src[1].arg] = c.src[0].arg
  used_ranges = [x.arg for x in idx.toposort if x.op is Ops.RANGE]
  ret = [x if i in used_ranges else 0 for i,x in enumerate(ret)]
  if not ignore_valid:
    for masked_axis in [x.arg for x in valid.toposort if x.op is Ops.RANGE]: ret[masked_axis] = None
  return tuple(ret)

@dataclass(frozen=True, order=True)
class ShapeTracker:
  views: Tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[ShapeTracker]:
    inverted_views:List[View] = []
    for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]):
      if (inverted:= v.invert(s)) is None: return None
      inverted_views.append(inverted)
    return ShapeTracker(tuple(inverted_views)).reshape(out_shape)

  @staticmethod
  def from_shape(shape:Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker((View.create(shape),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def consecutive(self) -> bool: return len(self.views) == 1 and (v:=self.views[0]).mask is None and v.strides == strides_for_shape(v.shape)

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def reduce(self, axis:Tuple[int, ...]) -> Tuple[sint, ...]: return tuple(1 if i in axis else s for i,s in enumerate(self.shape))

  def to_uop(self) -> UOp: return UOp(Ops.VIEW, dtypes.void, (), self)
  def to_indexed_uops(self, _idxs:Optional[List[UOp]|Tuple[UOp, ...]]=None) -> Tuple[UOp, UOp]:
    return views_to_indexed_uops(self.views, tuple(_idxs) if _idxs is not None else None)

  def real_size(self) -> int:
    if 0 in self.shape: return 0
    idx, valid = self.to_indexed_uops()
    if not valid.vmax: return 0
    assert idx.vmax < 1e12, f"real_size broken for {self}"
    return int(idx.vmax+1)

  def vars(self) -> Set[Variable]: return set().union(*[v.vars() for v in self.views])

  @property
  def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> Tuple[ShapeTracker, Dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]: return views_to_real_strides(self.views, ignore_valid)
  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.to_indexed_uops()
    return axis in [x.arg for x in graph_rewrite(valid, symbolic_flat).toposort if x.op is Ops.RANGE]

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def stride(self, mul: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))

  def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))
