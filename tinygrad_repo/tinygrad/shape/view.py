from __future__ import annotations
import functools, operator, itertools, math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, cast, Sequence
from tinygrad.dtype import dtypes
from tinygrad.ops import resolve, UOp, Variable, sint, sym_infer, smax, smin, sint_to_uop
from tinygrad.helpers import prod, all_int, argsort, flatten, ceildiv

@functools.lru_cache(maxsize=None)
def canonicalize_strides(shape:Tuple[sint, ...], strides:Tuple[sint, ...]) -> Tuple[sint, ...]:
  return tuple(0 if s == 1 else st for s, st in zip(shape, strides))

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[sint, ...]) -> Tuple[sint, ...]:
  if not shape: return ()
  strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))[::-1]
  return canonicalize_strides(shape, strides)

@functools.lru_cache(maxsize=None)
def merge_dims(shape:Tuple[int, ...], strides:Tuple[int, ...], mask:Optional[Tuple[Tuple[int, int], ...]]=None) -> Tuple[Tuple[int, int, int], ...]:
  # merge contiguous sub-parts or zero strided dims
  # any stride 0, masked from dim=1, or contiguous part is merged into next dim.
  # stride != 0 to stride == 0 starts a new merging block
  # ret = Tuple[(merged_size, stride, merged size w/o zero stride), ...]
  if not shape: return ()
  assert len(shape) == len(strides) and (mask is None or len(shape) == len(mask))
  ret = [(shape[0], strides[0], shape[0] if strides[0] != 0 else 0)]
  # merge this dim to next dim if size is 1
  merging = (mask[0][1] - mask[0][0] == 1) if mask is not None else shape[0] == 1
  for i, (s, st) in enumerate(zip(shape[1:], strides[1:]), start=1):
    # always merge 1
    if s == 1: continue
    last_s, last_st, last_pre_expand_s = ret[-1]
    # merge last dim with this dim if merging or strides matched
    if merging or last_st == s * st: ret[-1] = (last_s * s, st, (s if merging else last_pre_expand_s * s))
    else: ret.append((s, st, s))
    # merge this dim to next dim if size is 1
    merging = (mask[i][1] - mask[i][0] == 1) if mask is not None else s == 1
  return tuple(ret)

@functools.lru_cache(maxsize=None)
def _reshape_mask(_mask:Optional[Tuple[Tuple[sint, sint], ...]], old_shape:Tuple[sint, ...], new_shape:Tuple[sint, ...]) \
  -> Optional[Tuple[Tuple[sint, sint], ...]]:
  """Returns the new mask if reshape is possible, and None if not possible."""
  if _mask is None: return tuple((0, s) for s in new_shape)
  if any(not all_int(m) for m in _mask): return None
  if any(m[1] - m[0] < 1 for m in _mask): return ((0, 0),) * len(new_shape)  # zero mask

  new_mask: List[Tuple[int, int]] = []
  # _mask is all int here
  r_masks, r_shape, r_new_shape = reversed(cast(Tuple[Tuple[int, int], ...], _mask)), reversed(old_shape), reversed(new_shape)
  curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))

  while len(new_mask) < len(new_shape):
    (l, r), next_stride = mask, new_dim * curr_stride

    if old_dim >= next_stride: # need to split mask.
      if old_dim == next_stride: # simply copy the mask and get next batch for merging
        new_mask.append((l // curr_stride, (r - 1) // curr_stride + 1))
        curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))

      else: # mask can only be splitted if reshape doesn't cut across the mask.
        if (((l % next_stride != 0 or r % next_stride != 0) and l // next_stride != (r - 1) // next_stride)
            or old_dim % next_stride != 0): return None
        new_mask.append((l % next_stride // curr_stride, (r - 1) % next_stride // curr_stride + 1))
        curr_stride, new_dim = next_stride,  next(r_new_shape, 1) # need to get mask for next dimension

    else:
      next_mask = next(r_masks, (0, 1))
      # combine if the mask can unfold continuously
      if mask != (0, old_dim) and next_mask[1] - next_mask[0] != 1: return None
      mask, old_dim = (next_mask[0] * old_dim + l, (next_mask[1] - 1) * old_dim + r), old_dim * next(r_shape, 1)

  for mask in r_masks: # if the old shape has leading 1s, need to make sure their mask is (0,1)
    if mask != (0, 1): return ((0, 0),) * len(new_shape) # invalid mask

  return tuple(reversed(new_mask))

def un1d(shape:Tuple[sint, ...], offs:sint) -> List[sint]:
  result = []
  for stride in strides_for_shape(shape):
    here = offs // stride if stride != 0 else 0
    result.append(here)
    offs -= here * stride
  return result

@dataclass(frozen=True)
class View:
  shape:Tuple[sint, ...]
  strides:Tuple[sint, ...]
  offset:sint
  mask:Optional[Tuple[Tuple[sint, sint], ...]]
  contiguous:bool

  @functools.cached_property
  def t(self):
    return tuple(x.tuplize if isinstance(x, UOp) else (x,) \
                 for x in self.shape+self.strides+(self.offset,)+(tuple(flatten(self.mask)) if self.mask is not None else tuple()))
  def __lt__(self, o:View): return self.t < o.t

  def to_indexed_uops(self:View, idxs:Optional[Sequence[UOp]]=None, vexpr:UOp=UOp.const(dtypes.bool, True)) -> Tuple[UOp, UOp]:
    """(idx, valid)"""
    if idxs is None: idxs = [UOp.range(dtypes.int, 0, s, i) for i,s in enumerate(self.shape)]
    iexpr = sint_to_uop(self.offset)
    for idx,sh,st,m in zip(idxs, self.shape, self.strides, self.mask if self.mask is not None else itertools.repeat(None)):
      if resolve(sh != 1) and resolve(st != 0): iexpr = iexpr + idx*st
      if m is not None:
        if resolve(m[0] != 0): vexpr = vexpr * (idx >= m[0])
        if resolve(m[1] != sh): vexpr = vexpr * (idx < m[1])
    return iexpr, vexpr

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def size(self) -> int:
    ret = prod([x.vmax if isinstance(x, UOp) else x for x in self.shape])
    assert isinstance(ret, int), f"{ret=} is not int"
    return ret

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def create(shape:Tuple[sint, ...], strides:Optional[Tuple[sint, ...]]=None, offset:sint=0, mask:Optional[Tuple[Tuple[sint, sint], ...]]=None):
    # TODO: this resolve shouldn't be needed
    if not all(resolve(s >= 0) for s in shape): raise ValueError(f"Trying to create View with negative dimension: {shape=}")
    strides = canonicalize_strides(shape, strides) if strides else strides_for_shape(shape)
    # canonicalize 0 in shape
    if 0 in shape: return View(shape, (0,) * len(shape), offset=0, mask=None, contiguous=True)
    # canonicalize empty mask
    if mask is not None and all(m == (0,s) for m,s in zip(mask, shape)): mask = None
    # if any dimension has size >1, but is masked such that only one index in the dimension is unmasked
    # then its stride can also be set to 0, albeit with a corresponding adjustment required to the offset
    if mask and any(elim := [not resolve(b+1 < e) for b,e in mask]):
      if any(not resolve(b < e) for b,e in mask):
        strides, offset, mask = (0,) * len(shape), 0, ((0,0),) * len(shape)
      offset += sum((strides[i] * mask[i][0]) if e else 0 for i, e in enumerate(elim))
      strides = tuple(0 if e else st for st,e in zip(strides, elim))
    # simplify as we go
    if isinstance(offset, UOp): offset = cast(sint, offset.ssimplify())
    shape = tuple(cast(sint, x.ssimplify()) if isinstance(x, UOp) else x for x in shape)
    # TODO: enabling stride simplification breaks it
    """
    strides = tuple(x.ssimplify() if isinstance(x, UOp) else x for x in strides)
    if mask: mask = tuple((s.ssimplify() if isinstance(s, UOp) else s, e.ssimplify() if isinstance(e, UOp) else e) for s,e in mask)
    """
    contiguous = offset == 0 and mask is None and strides == strides_for_shape(shape)
    return View(shape, strides, offset, mask, contiguous)

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def vars(self) -> Set[Variable]:
    flatten_mask = tuple(x for m in self.mask for x in m) if self.mask is not None else tuple()
    return functools.reduce(operator.or_, [x.vars() for x in self.shape+self.strides+(self.offset,)+flatten_mask if isinstance(x, UOp)], set())

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def unbind(self) -> Tuple[View, Dict[Variable, int]]:
    var_unboundvar_val = [(v, v.unbind()) for v in self.vars()]
    unbound_vars = {v:uv for v,(uv,_) in var_unboundvar_val}
    def substitute(x): return x if isinstance(x, int) else x.substitute(unbound_vars)
    new_shape = tuple(map(substitute, self.shape))
    new_strides = tuple(map(substitute, self.strides))
    new_offset = substitute(self.offset)
    new_mask = tuple((substitute(x[0]), substitute(x[1])) for x in self.mask) if self.mask is not None else None
    return View.create(new_shape, new_strides, new_offset, new_mask), dict(x[1] for x in var_unboundvar_val)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, vm1:View) -> Optional[View]:
    vm2 = self
    if vm2.contiguous: return vm1
    if vm1.contiguous and vm1.shape == vm2.shape: return vm2
    if vm1.contiguous and vm1.size() == vm2.size() and (ret := vm2.reshape(vm1.shape)) is not None: return ret
    if vm1.mask:
      for b,e in vm1.mask:
        if resolve(b >= e, False): return View.create(vm1.shape, (0,) * len(vm1.shape), 0, ((0,0),) * len(vm1.shape))
      return (merged := vm2 + vm1.shrink(vm1.mask)) and merged.pad(tuple((b,s-e) for (b,e),s in zip(vm1.mask, vm1.shape)))

    # Project vm1's offset and strides on to vm2.
    origin = un1d(vm2.shape, vm1.offset)
    terms: List[List[Tuple[int, sint]]] = [[] for _ in origin]
    strides: List[sint] = [0] * len(vm1.shape)
    for d1, st in enumerate(vm1.strides):
      if st == 0: continue
      for d2, (o, s1) in enumerate(zip(origin, un1d(vm2.shape, vm1.offset + st))):
        if (s1 := s1 - o) == 0: continue
        terms[d2].append((d1, s1))
        strides[d1] += s1 * vm2.strides[d2]

    # Merge dimensions in vm2 if required.
    # NB: Merging too many dimensions can make it difficult to project vm2's mask, hence only combining when required.
    if not all_int(vm1.shape): return None
    idxs: List[UOp] = [UOp.variable(f"idx{i}", 0, s-1) for i,s in enumerate(vm1.shape)]
    merged_size, merged_term = 1, UOp.const(dtypes.int, 0)
    extents: List[Tuple[sint, UOp]] = []
    for term, s, o in zip(reversed(terms), reversed(vm2.shape), reversed(origin)):
      merged_term += (sum([idxs[d1] * s1 for d1, s1 in term]) + o) * merged_size
      merged_size *= s
      if resolve(merged_term < merged_size, False) and resolve(0 <= merged_term, False):
        extents.append((merged_size, merged_term))
        merged_size, merged_term = 1, UOp.const(dtypes.int, 0)
    if resolve(merged_term != 0): return None
    if (vm2_shape := tuple(s for s,_ in reversed(extents))) != vm2.shape:
      reshaped_vm2 = vm2.reshape(vm2_shape)
      if reshaped_vm2 is None: return None
      if reshaped_vm2.shape != vm2.shape: return reshaped_vm2 + vm1

    if vm2.mask:
      # Try to project vm2's mask on to vm1.
      newb, newe, bad = [0] * len(vm1.shape), list(vm1.shape), False
      for (b, e), o, term, (_, t) in zip(vm2.mask, origin, terms, reversed(extents)):
        if resolve(b <= t.vmin and t.vmax < e, False): continue
        if not all_int([o, b, e]):
          bad = True
          continue
        if len(term) != 1:
          if not term and newe: newe[0] = 0
          else: bad = True
          continue
        d1, s1 = term[0]
        if not isinstance(s1, int) or not isinstance(newe[d1], int):
          bad = True
          continue
        newb[d1] = max(newb[d1], math.ceil((b - o if s1 > 0 else e - o - 1) / s1))
        newe[d1] = min(newe[d1], (b - o if s1 < 0 else e - o - 1) // s1 + 1)

      # If any of vm1 was masked off, try again with that mask in place.
      for b, e, s in zip(newb, newe, vm1.shape):
        if (b, e) != (0, s):
          return vm2 + View.create(vm1.shape, vm1.strides, vm1.offset, tuple(zip(newb, newe)))
      # Otherwise if vm2's mask was violated, then cannot merge.
      if bad: return None

    return View.create(vm1.shape, tuple(strides), sum(o * s for o, s in zip(origin, vm2.strides)) + vm2.offset)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[View]:
    ret = View.create(self.shape)
    if self.mask: ret = ret.shrink(self.mask)
    ret = ret.stride(tuple(-1 if x < 0 else 1 for x in self.strides)).permute(argsort(tuple(-x if x > 0 else x for x in self.strides)))
    return ret if prod(ret.shape) == prod(out_shape) else None   # don't support shrink, expand, or stride != (-1, 1)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def minify(self):
    min_shape = tuple(x[0] for x in merge_dims(self.shape, self.strides, self.mask))
    return nv if (nv := self.reshape(min_shape)) else self

  def __unsafe_resize(self, arg: Tuple[Tuple[sint, sint], ...], mask=None) -> View:
    offset = sum([s * x[0] for s, x in zip(self.strides,arg)])
    if self.mask:
      # move the old mask
      nmask = tuple([(smax(0, smin(mx-ax,ay-ax)), smax(0, smin(my-ax,ay-ax))) for (mx,my),(ax,ay) in zip(self.mask, arg)])
      # merge the masks if we have two
      mask = tuple([(smax(mx1, mx2), smin(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
    shape = [y-x for x,y in arg]
    if mask is not None and all(m[0] == 0 and m[1] == s for m,s in zip(mask, shape)): mask = None
    return View.create(tuple(s.ssimplify() if isinstance(s, UOp) else s for s in shape), self.strides, self.offset+offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def pad(self, arg: Tuple[Tuple[sint, sint], ...]) -> View:
    assert len(arg) == len(self.shape), f"invalid pad {arg} for {self.shape}"
    # NOTE: not checking for symbolic arg
    for b,e in arg: assert not all_int([b,e]) or b>=0 and e>=0, f"invalid pad {arg} for {self.shape}"
    if any(resolve(b!=0) or resolve(e!=0) for b, e in arg):
      zvarg = tuple([(-b,s+e) for s,(b,e) in zip(self.shape, arg)])
      mask = tuple([(b,s+b) for s,(b,_) in zip(self.shape, arg)])
      return self.__unsafe_resize(zvarg, mask=mask)
    return self

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> View:
    assert len(arg) == len(self.shape), f"invalid shrink {arg} for {self.shape}"
    # NOTE: not checking for symbolic arg
    for s,(b,e) in zip(self.shape,arg): assert not all_int([s,b,e]) or (0<=b<=e<=s), f"invalid shrink {arg} for {self.shape}"
    return self.__unsafe_resize(arg)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def expand(self, new_shape: Tuple[sint, ...]) -> View:
    if len(new_shape) != len(self.shape): raise ValueError(f"expand arg {new_shape=} must have same number of dimensions as shape {self.shape=}")
    # NOTE: does not check multiple of symbolic shape
    assert all(resolve(s == ns) or s == 1 for s,ns in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
    if 0 in self.shape: return View.create(new_shape)
    # NOTE: can the mask ever be (0,0)?
    # TODO: this resolve may not be needed, but it's hard because vars need to be sorted
    mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if resolve(s != ns, False) else m) \
                  for m,s,ns in zip(self.mask, self.shape, new_shape)]) if self.mask else None
    return View.create(new_shape, self.strides, self.offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def permute(self, axis: Tuple[int, ...]) -> View:
    assert sorted(axis) == list(range(len(self.shape))), f"invalid permutation {axis} of len {len(self.shape)}"
    return View.create(tuple(self.shape[a] for a in axis), tuple(self.strides[a] for a in axis), self.offset,
                       tuple(self.mask[a] for a in axis) if self.mask is not None else None)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def stride(self, mul: Tuple[int, ...]) -> View:
    # except for the negative case, you can build this from the others. invertible in the negative case
    assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {self.shape}"
    strides = tuple([z*m for z,m in zip(self.strides, mul)])
    new_shape = tuple([ceildiv(s, abs(m)) for s,m in zip(self.shape, mul)])
    offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
    mask = tuple([(ceildiv(mx if m > 0 else s-my, abs(m)), ceildiv(my if m > 0 else s-mx, abs(m))) \
                  for (mx,my),s,m in zip(self.mask, self.shape, mul)]) if self.mask is not None else None
    return View.create(new_shape, strides, self.offset + offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def reshape(self, new_shape: Tuple[sint, ...]) -> Optional[View]:
    if self.shape == new_shape: return self

    assert all(x >= 0 for x in new_shape), f"shape can't contain negative numbers {new_shape}"
    # check for the same size
    if (self_all_int := all_int(self.shape)):
      assert all(isinstance(s, (int, UOp)) for s in new_shape), f"{self.shape=} -> {new_shape=} contains non (int, Variable) dim"
      if resolve(prod(self.shape) != prod(new_shape), False): raise ValueError(f"size mismatched, can't reshape {self.shape=} -> {new_shape=}")

    if 0 in self.shape: return View.create(new_shape)
    if new_shape == () and self.mask and any(mx==my for (mx,my) in self.mask): return None

    # after the asserts, it's okay to check contiguous
    if self.contiguous: return View.create(new_shape)

    # if it's not contiguous and new shape is symbolic, check if it's directly replaceable
    if self_all_int and not all_int(new_shape):
      if len(self.shape) != len(new_shape): raise ValueError(f"cannot symbolic reshape non-contiguous {self} -> {new_shape}")
      for si, so in zip(self.shape, new_shape):
        if not isinstance(so, int): so = sym_infer(so, dict([v.unbind() for v in so.vars()]))
        if si != so: raise ValueError(f"cannot symbolic reshape non-contiguous {self} -> {new_shape}")
      # all dimensions matched, return the new view directly
      return View(new_shape, self.strides, self.offset, self.mask, self.contiguous)

    r_strides, r_new_shape = [], reversed(new_shape)
    for merged_size, new_stride, real_size in reversed(merge_dims(self.shape, self.strides, self.mask)):
      # TODO: write with get_contraction
      acc = 1
      # TODO: third resolve shouldn't be needed
      while resolve(acc <= merged_size) and resolve(acc != merged_size) and resolve((new_dim := next(r_new_shape, 0)) > 0):
        r_strides.append(new_stride * acc)
        acc = acc * new_dim
        if not resolve(acc < real_size): new_stride = 0
      if resolve(acc != merged_size): return None
    new_strides = (0,) * (len(new_shape) - len(r_strides)) + tuple(r_strides[::-1])

    if (new_mask:=_reshape_mask(self.mask, self.shape, new_shape)) is not None:
      extra_offset = (sum(m[0] * s for m,s in zip(self.mask, self.strides)) if self.mask else 0) - \
                     (sum(m[0] * s for m,s in zip(new_mask, new_strides)))
      return View.create(new_shape, new_strides, self.offset + extra_offset, new_mask)

    return None
