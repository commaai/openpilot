from __future__ import annotations
import itertools, functools
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, List, Tuple, cast, Dict, Final, DefaultDict, Callable, Sequence
from enum import Enum, auto

from tinygrad.ops import GroupOp, KernelInfo, UOp, Ops, can_pad, print_uops, type_verify, resolve, Variable, sint, \
  graph_rewrite, track_rewrites, view_left
from tinygrad.device import Device
from tinygrad.renderer import Renderer, TensorCore, ProgramSpec
from tinygrad.dtype import ImageDType
from tinygrad.helpers import all_same, colored, ansilen, dedup, getenv, prod, round_up, all_int, to_function_name, diskcache_put
from tinygrad.helpers import DEBUG, TC_OPT, USE_TC, AMX
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import strides_for_shape
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.codegen.uopgraph import full_graph_rewrite
from tinygrad.codegen.lowerer import rewrite_shapetracker_with_index, get_contraction

class OptOps(Enum):
  TC = auto(); UPCAST = auto(); UPCASTMID = auto(); UNROLL = auto(); LOCAL = auto() # noqa: E702
  GROUP = auto(); GROUPTOP = auto(); NOLOCALS = auto(); PADTO = auto(); SWAP = auto() # noqa: E702
  def __lt__(self, x:OptOps): return self.value < x.value

class KernelOptError(Exception): pass

def check(cond:bool, msg:str=""):
  if not cond: raise KernelOptError(msg)

@dataclass(frozen=True, order=True)
class Opt:
  op: OptOps
  axis: Optional[int] = None
  amt: Optional[int] = None
  def __repr__(self): return f"Opt(op={self.op}, axis={self.axis}, amt={self.amt})"
  def real_axis(self, k:Kernel):
    if self.axis is None: return -1
    if self.op is OptOps.UNROLL: return k.first_reduce+self.axis
    if self.op in {OptOps.GROUP, OptOps.GROUPTOP}: return k.first_reduce+k.group_for_reduces+self.axis
    return self.axis

@dataclass
class TensorCoreOptions:
  axes: Tuple[int, ...] # the location of the original N and M axes if still in the shape
  axes_exist: Tuple[bool, ...] # true if the original N and M axes are still in the shape
  axis_pads: Tuple[Tuple[int, int], ...]
  def fix_axes(self, removed_axis:int): # adjust the TC axes if necesssary when a dimension is removed
    axes, axes_exist = list(self.axes), list(self.axes_exist)
    for tc_dim in [i for i in range(2) if axes_exist[i]]:
      if removed_axis < axes[tc_dim]: axes[tc_dim] -= 1
      elif removed_axis == axes[tc_dim]: axes_exist[tc_dim] = False
    self.axes, self.axes_exist = tuple(axes), tuple(axes_exist)

class Kernel:
  def __init__(self, ast:UOp, opts:Optional[Renderer]=None):
    if ast.op is Ops.SINK: self.ast = ast

    self.opts = opts if opts is not None else Device[Device.DEFAULT].renderer
    try: uop_sts_map = verify_ast(self.ast)
    except AssertionError as e:
      print("INVALID AST")
      print(self.ast)
      raise e

    self.reduceops = [x for x in self.ast.toposort if x.op is Ops.REDUCE_AXIS]

    self.vars: List[Variable] = self.ast.variables()
    # NOTE: this requires a specific order with the [::-1], this is likely a bug
    self.bufs: List[UOp] = [x for x in self.ast.toposort if x.op in GroupOp.Buffer][::-1]

    # get earlybufs, before any reduceops
    earlybufs: List[UOp] = [x for reduceop in self.reduceops for x in reduceop.src[0].toposort if x.op in GroupOp.Buffer]
    self.full_buf_index: int = self.bufs.index(earlybufs[0]) if earlybufs else 0
    # NOTE: full_shape can be wrong if there's a tree of reduces

    # create new shapetrackers inside this kernel, we will permute them
    self.sts: List[ShapeTracker] = [x.st_arg for x in self.bufs]

    # add the shapetrackers for each reduce
    # we use this to track which axes are reduced in each reduce
    for x in self.reduceops:
      self.sts.append(uop_sts_map[x])
      self.sts.append(uop_sts_map[x.src[0]])

    # move all reduce axes to the end
    reduce = list(enumerate(zip(self.full_shape, self.output_shape)))
    permute = tuple([i for i,(s,n) in reduce if not resolve(s != n)] + [i for i,(s,n) in reduce if resolve(s != n)])
    self.reshape_and_permute(None, permute)

    # parameters for optimization
    self.applied_opts: List[Opt] = []
    self.group_for_reduces: int = 0
    self.upcasted: int = 0
    self.local_dims: int = 0
    self.tensor_core: Optional[TensorCore] = None
    self.tensor_core_opts: Optional[TensorCoreOptions] = None
    self.use_tensor_cores: int = 0
    # the local aliased buffers for A and B
    self.bufs_for_tensor_core: Dict[UOp, Tuple[int, int]] = {}
    self.dont_use_locals: bool = False

    # group simplifies
    self.simplify_ones()
    self.simplify_merge_adjacent()

  def copy(self):
    ret = type(self).__new__(type(self))

    # base linearizer params
    ret.opts, ret.ast = self.opts, self.ast

    # things downstream of the AST
    ret.reduceops, ret.vars, ret.bufs, ret.full_buf_index = self.reduceops, self.vars, self.bufs, self.full_buf_index
    ret.sts = self.sts[:len(ret.bufs)+len(ret.reduceops)*2] # NOTE: must redo the local buffers with TC in beam

    # parameters for optimizations
    ret.applied_opts, ret.group_for_reduces, ret.upcasted, ret.local_dims, ret.dont_use_locals = \
      self.applied_opts[:], self.group_for_reduces, self.upcasted, self.local_dims, self.dont_use_locals
    ret.tensor_core, ret.tensor_core_opts, ret.bufs_for_tensor_core, ret.use_tensor_cores = \
      self.tensor_core, self.tensor_core_opts, self.bufs_for_tensor_core, self.use_tensor_cores

    return ret

  @property
  def membufs(self) -> List[UOp]: return dedup([x.src[0] for x in self.bufs if x.op in {Ops.LOAD, Ops.STORE}])

  # TODO: these need more tests or it might silently be no-op
  def float4_axis(self, i:int): return [x-self.first_upcast for x in self.sts[i].unit_stride_axes() if x >= self.first_upcast and self.sts[i].shape[x]%4 == 0]  # noqa: E501

  def upcasted_axis(self, i:int) -> List[Tuple[int, Optional[sint], bool]]:
    upcasted_shape, upcasted_stride = self.sts[i].shape[self.first_upcast:], self.sts[i].real_strides()[self.first_upcast:]
    assert all_int(upcasted_shape), f"cannot upcast a symbolic amount {upcasted_shape=}"
    return list(zip(upcasted_shape, upcasted_stride,
                    [x!=y for x,y in zip(self.sts[0].shape[self.first_upcast:], self.full_shape[self.first_upcast:])]))

  @property
  def first_reduce(self) -> int:
    return [resolve(x!=y) for x,y in zip(self.sts[0].shape[:self.first_upcast]+(0,), self.full_shape[:self.first_upcast]+(1,))].index(True)

  @property
  def first_upcast(self) -> int: return self.shape_len-self.upcasted

  @property
  def reduceop(self) -> Optional[UOp]: return self.reduceops[0] if len(self.reduceops) > 0 else None

  @property
  def output_shape(self) -> Tuple[sint, ...]: return self.sts[0].shape

  @property
  def full_shape(self) -> Tuple[sint, ...]: return self.sts[self.full_buf_index].shape

  @property
  def full_unupcasted_shape(self) -> Tuple[sint, ...]: return self.full_shape[:self.first_upcast]

  @property
  def shape_len(self) -> int: return len(self.sts[0].shape)

  @property
  def upcast_in_mid_reduce_axes(self) -> List[int]:
    return [j for j in range(self.first_reduce, self.first_reduce+self.group_for_reduces) if self.full_shape[j] == self.sts[0].shape[j]]

  @property
  def global_dims(self) -> int: return self.first_reduce-self.local_dims

  # there's eight chunks of the shape
  # blue   -- global dims
  # cyan   -- local dims (warp ones first)
  #  *** self.first_reduce
  # green  -- reduce-local dims
  # white  -- reduce-late upcasted dim (self.upcast_in_mid_reduce_axes)
  # red    -- reduce loops
  #  *** self.upcasted
  # purple -- reduce upcasted
  # yellow -- normal upcasted dimensions
  def colors(self) -> List[str]:
    # first non local non reduce dims are global (blue)
    colors = ["blue"] * self.global_dims if not self.dont_use_locals else ["BLUE"] * self.global_dims
    # after global are local_dims; warp ones used in tensor cores must be closest to first_reduce (cyan)
    colors += ["cyan"] * self.local_dims
    # between first_reduce and first_reduce + group_for_reduces, they are either upcast mid reduce (white), or late upcasted (green)
    colors += ["white" if i in self.upcast_in_mid_reduce_axes else "green" for i in range(self.first_reduce, self.first_reduce + self.group_for_reduces)]  # noqa: E501
    # between first_reduce + group_for_reduces and upcasted, they are reduce (red)
    colors += ["red"] * (self.first_upcast - (self.first_reduce + self.group_for_reduces))
    # upcasted dimensions are reduce (magenta) or normal (yellow)
    colors += ["magenta" if self.full_shape[i] != self.sts[0].shape[i] else "yellow" for i in range(self.first_upcast, self.shape_len)]
    assert len(colors) == self.shape_len, "colors size mismatch"
    return colors

  def colored_shape(self, pad:Optional[int]=None, dense=False) -> str:
    shape_strs = [(s if dense else f"{s:4d}") if isinstance(s, int) else s.render() for s in self.full_shape]
    ret = ' '.join(colored(s, color) for s,color in zip(shape_strs, self.colors()))
    if pad: ret += ' '*(pad-ansilen(ret))
    return ret

  # ******************** base simplifiers ********************

  # apply reshape and permute to all shapetrackers
  def reshape_and_permute(self, new_shape_fxn:Optional[Callable[[Tuple[sint, ...]], Sequence[sint]]], axis:Optional[Sequence[int]]):
    def reshape(st:ShapeTracker): return st.reshape(tuple(new_shape_fxn(st.shape))) if new_shape_fxn is not None else st
    def permute(st:ShapeTracker): return st.permute(tuple(axis)) if axis is not None else st
    self.sts = [permute(reshape(st)) for st in self.sts]

  # drops the final dimension
  def upcast(self):
    check(self.full_shape[-1] != 1, "can't upcast a dimension with size 1")
    self.upcasted += 1

  # axis : the axis to pull from
  # amount : the amount to take
  # top : if you want to pull that amount from the top
  # insert_before : place to insert the new stuff
  def shift_to(self, axis, amount, top=False, insert_before=None):
    if insert_before is None: insert_before = self.shape_len
    move_axis = axis if top else axis+1
    if move_axis < insert_before: insert_before += 1
    self.reshape_and_permute(
      lambda x: x[0:axis] + (((amount, x[axis]//amount) if top else (x[axis]//amount, amount)) if x[axis] > 1 else (1,1)) + x[axis+1:],
      [i for i in range(insert_before) if i != move_axis] + [move_axis] + [i for i in range(insert_before, self.shape_len+1) if i != move_axis])

  # ******************** complex simplifiers ********************

  def simplify_ones(self) -> bool:
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    if self.shape_len == 0: return False
    all_ones = [s==1 for s in self.full_shape]
    self.local_dims -= sum(all_ones[self.first_reduce-self.local_dims:self.first_reduce])
    self.upcasted -= sum(all_ones[self.first_upcast:]) # TODO: no necessary since upcasted axis can't be un-upcasted
    self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)
    return any(all_ones)

  def simplify_merge_adjacent(self):
    if self.shape_len == 0: return
    shapes, strides = [x.shape for x in self.sts], [x.real_strides() for x in self.sts]

    # if it's an image, insert fake strides such that this fusion doesn't happen across image axes
    if isinstance(self.membufs[0].dtype, ImageDType):
      base_shape = self.membufs[0].dtype.shape
      if shape_idx_groups := get_contraction(self.output_shape, base_shape):
        special_strides: Tuple[sint, ...] = tuple()
        for i,g in enumerate(shape_idx_groups):
          shape_piece = tuple(self.output_shape[x] for x in g)
          assert prod(shape_piece) == base_shape[i], f"get_contraction was wrong? {shape_piece} != {base_shape[i]}"
          special_strides += strides_for_shape(shape_piece)
        # adding the fake image shape
        shapes.append(self.output_shape)
        strides.append(special_strides)

    # merge dimensions if we can, multi _merge_dims
    # NOTE: this does not always preserve the reduce dimension
    # TODO: move this into shapetracker, with tests!
    # TODO: how does this work with multi-reduce?
    rets = [[(s[0], st[0])] for s,st in zip(shapes, strides)]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for s,st,ret in zip(shapes, strides, rets):
        # TODO: added the always mergeability of 1s, is this right? if so, add to shapetracker in the 1 case
        si, sti, last_st = s[i], st[i], ret[-1][1]
        can_merge.append((sti is not None) and ((sti != 0 and last_st == si*sti) or (sti == 0 and last_st == 0)))
      # more can merge than this
      mergeable = all(can_merge) and i != self.first_reduce
      for j,(s,st) in enumerate(zip(shapes, strides)):
        if mergeable: rets[j][-1] = (rets[j][-1][0] * s[i], st[i])
        else: rets[j].append((s[i], st[i]))

    # do the reshapes
    for i,x in enumerate(rets[:len(self.sts)]): self.sts[i] = self.sts[i].reshape(tuple([y[0] for y in x]))

  # ******************** high level optimizers ********************

  def _create_tc_opts(self, reduceop:UOp, tc:TensorCore, axis:int, opt_level:int) -> Optional[TensorCoreOptions]:
    has_cast = tc.dtype_in != tc.dtype_out
    if has_cast and not (reduceop.src[0].op is Ops.CAST and reduceop.src[0].dtype == tc.dtype_out): return None

    mul_op = reduceop.src[0].src[0] if has_cast else reduceop.src[0]
    if mul_op.op is not Ops.MUL: return None

    def buf_index(src:UOp) -> Optional[int]:
      # TODO: apply tc even if the sources are not from LOAD
      if src.op is Ops.LOAD and src.dtype == tc.dtype_in: return self.bufs.index(src)
      try:
        if opt_level >= 1 and src.op is Ops.CAST and src.dtype == tc.dtype_in: return self.bufs.index(src.src[0])
      except ValueError: return None
      return None
    if (buf0:=buf_index(mul_op.src[0])) is None or (buf1:=buf_index(mul_op.src[1])) is None: return None

    buf0_strides, buf1_strides = self.sts[buf0].real_strides(), self.sts[buf1].real_strides()
    axis_buf0 = [(i,self.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides[:self.first_reduce]) if s == 0]
    axis_buf1 = [(i,self.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides[:self.first_reduce]) if s == 0]
    if not (axis_buf0 and axis_buf1 and ((self.shape_len-self.first_reduce) == 1 or (opt_level >= 1))): return None

    axis_choices = list(itertools.product(axis_buf0, axis_buf1, range(self.first_reduce, self.shape_len)))
    if not (axis < len(axis_choices)): return None

    s0, s1, s2 = axis_choices[-(axis+1)][0][0], axis_choices[-(axis+1)][1][0], axis_choices[-(axis+1)][2]  # s0 is n, s1 is m, s2 is k
    axis_pads = tuple((x, tc.dims[i]) for i, x in enumerate([s0, s1, s2]) if resolve(self.full_shape[x]%tc.dims[i] != 0))
    if axis_pads and (opt_level < 2): return None
    self.bufs_for_tensor_core[reduceop] = (buf0, buf1)
    if DEBUG >= 3: print("TENSOR CORES", axis_buf0, axis_buf1, tc)
    return TensorCoreOptions(axes=(s0, s1, s2), axes_exist=(True, True), axis_pads=axis_pads)

  def _apply_tc_opt(self, use_tensor_cores:int, axis:int, opt_level:int) -> bool:
    if use_tensor_cores and self.reduceop is not None and self.reduceop.arg[0] is Ops.ADD:
      for tc in self.opts.tensor_cores:
        tensor_core_opts = [self._create_tc_opts(reduceop, tc, axis, opt_level) for reduceop in self.reduceops]
        # can only fuse reduces with the same tc options
        assert all_same(tensor_core_opts)
        if tensor_core_opts[0] is None: continue
        # tensor core -- unroll the reduce dim, upcast input and local the correct thread pattern
        self.tensor_core_opts = tc_opts = tensor_core_opts[0]

        # attempt to pad the tensor axes that require it
        try:
          for axis, dim in tc_opts.axis_pads: self.apply_opt(Opt(OptOps.PADTO, axis, dim), append_opt=False) # PADTO might fail
        except KernelOptError: continue
        for tc_dim, amt in tc.reduce_axes: self.apply_opt(Opt(OptOps.UNROLL,tc_opts.axes[2]-self.first_reduce,amt), append_opt=False)
        for opt in tc.opts_seq:
          if opt == "UP":
            for tc_dim, amt in tc.early_upcast_axes: self.apply_opt(Opt(OptOps.UPCAST,tc_opts.axes[tc_dim],amt), append_opt=False)
          elif opt == "LC":
            for tc_dim, amt in tc.threads: self.apply_opt(Opt(OptOps.LOCAL,tc_opts.axes[tc_dim],amt), append_opt=False)
        self.tensor_core = tc
        self.use_tensor_cores = use_tensor_cores  # TC=2 will do the shape ops without the WMMA
        return True
    return False

  def apply_tensor_cores(self, use_tensor_cores=1, extra_opts:Optional[List[Opt]]=None, axis:int=0, tc_opt:Optional[int]=None) -> bool:
    """ Attempts to apply a tensor core optimization to the kernel.  If one exists and applies properly, return true, otherwise return false.
    Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).

    Keyword arguments:
    use_tensor_cores -- controls how tensor cores are applied (default 1)
      0: will disable any tensor core matching
      1: enable tensor cores
      2: apply tensor core shape but don't use UOp.WMMA
    extra_opts -- additional Opt's to apply after the tensor core instead of the hand-coded additional Opt's (default None)
    tc_opt -- controls which kinds of kernels may be eligible for tensor cores application (default 2 during BEAM, 0 otherwise)
      0: applies to only kernels with a single reduce axis and direct UOps.LOAD into Ops.MUL
      1: allows kernels with multiple reduce axes and also multiplication of UOps.CAST'd buffers
      2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
    """
    if tc_opt is None: tc_opt = TC_OPT.value
    if not self.opts.tensor_cores and use_tensor_cores != 2: return False
    try: # check TC first and apply hand-coded opts if successful
      self.apply_opt(Opt(OptOps.TC, axis, tc_opt))

      if (tc_opts:=self.tensor_core_opts) is not None:
        if extra_opts is not None:
          for opt in extra_opts: self.apply_opt(opt)
        else:
          if (self.opts.device == "CLANG" and AMX): return True # skip hand-coded TC opts if AMX, upcasting will make kernel slower
          # hand-coded TC opts
          for tc_dim in [tc_dim for tc_dim in [1,0] if tc_opts.axes_exist[tc_dim]]: # attempt to upcast M and N
            szs = [sz for sz in [5,4,3,2] if self.full_shape[tc_opts.axes[tc_dim]] % sz == 0]
            if szs: self.apply_opt(Opt(OptOps.UPCAST, tc_opts.axes[tc_dim], szs[0]))

          if tc_opts.axes_exist[0] and (szs := [sz for sz in [4,2] if self.full_shape[tc_opts.axes[0]] % sz == 0]): # attempt to local N
            self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[0], szs[0]))
      return True
    except KernelOptError:
      return False

  def apply_opt(self, opt:Opt, append_opt:bool=True):
    if self.dont_use_locals: check(opt.op not in {OptOps.LOCAL, OptOps.GROUP, OptOps.GROUPTOP, OptOps.UPCASTMID}, "not using locals")

    if opt.op is OptOps.TC:
      check(len(self.applied_opts) == 0, "tensor core opts must be first") # TODO: things like PADTO might be fine
      check(opt.axis is not None and opt.amt is not None, "tensor core opts must have an axis and amt")
      check((use_tensor_cores:=USE_TC.value) == 2 or len(self.opts.tensor_cores) > 0, "must have tensor cores or TC=2")
      check(self._apply_tc_opt(use_tensor_cores, cast(int, opt.axis), cast(int, opt.amt)), "no tensor core available")
      self.applied_opts.append(opt)
      return

    axis = opt.real_axis(self)
    check(axis < len(self.full_shape), "invalid axis")

    if opt.op is OptOps.SWAP: amt = cast(int, opt.amt)  # amt is an axis in the SWAPs
    elif opt.amt is not None:
      amt = opt.amt if opt.amt != 0 else self.full_shape[axis]
      check(isinstance(amt, int) and amt != 1, "shift/padto of amt 1 or Node is meaningless")
      if opt.op is not OptOps.PADTO: check(self.full_shape[axis] % amt == 0, "no longer valid shift")
    else: amt = -1

    if self.reduceop is not None and (opt.op in {OptOps.GROUP, OptOps.GROUPTOP} or \
                                      (self.group_for_reduces and opt.op not in {OptOps.NOLOCALS, OptOps.PADTO})):
      acc_sz = self.reduceop.dtype.itemsize
      upcast_sz = prod([a for a,b in zip(self.full_shape[self.first_upcast:], self.sts[0].shape[self.first_upcast:]) if a == b])
      local_sz = prod(self.full_shape[self.first_reduce-self.local_dims:self.first_reduce+self.group_for_reduces])
      smem_sz = amt*acc_sz*upcast_sz*local_sz
      check(smem_sz <= self.opts.shared_max, f"exceeds maximum shared memory size: needs {smem_sz}, max {self.opts.shared_max}")

    if opt.op is OptOps.LOCAL:    # cyan
      check(self.opts.has_local, "target does not support local")
      check(axis < self.global_dims, "local is for globals")
      self.shift_to(axis, amt, insert_before=self.first_reduce)
      self.local_dims += 1
    elif opt.op in {OptOps.GROUP, OptOps.GROUPTOP}:   # green
      check(self.opts.has_local and self.opts.has_shared, "target does not support local or shared mem")
      check(self.first_reduce + self.group_for_reduces <= axis < self.first_upcast, "must be reduce axis to group")
      check(not self.tensor_core, "can't group with tensor cores")
      check(len(reduce_axes:=[i for r in self.reduceops for i in r.axis_arg]) == len(set(reduce_axes)), "can't group with parallel reduces")
      self.shift_to(axis, amt, top=(opt.op is OptOps.GROUPTOP), insert_before=self.first_reduce + self.group_for_reduces)
      self.group_for_reduces += 1
    elif opt.op is OptOps.UNROLL:                     # purple
      check(axis < self.first_upcast, "can't upcasted already upcasted")
      check(amt <= 32, "don't unroll more than 32")
      # TODO: fix upcast_count to put purples before yellows. broken because of METAL tensor cores
      #upcast_count = sum(x == y for x,y in zip(self.full_shape[-self.upcasted:], self.output_shape[-self.upcasted:])) if self.upcasted else 0
      #self.shift_to(axis, amt, insert_before=None if upcast_count == 0 else self.shape_len-upcast_count)
      if self.full_shape[axis] == amt and axis == self.first_reduce: self.local_dims += 1 # first_reduce will ++, so offset loss in simplify_ones
      if self.full_shape[axis] == amt and axis < self.first_reduce+self.group_for_reduces: self.group_for_reduces -= 1 # fully unrolling a GROUP
      self.shift_to(axis, amt, insert_before=None)
      self.upcast()
    elif opt.op is OptOps.UPCAST:                     # yellow
      check(axis < self.first_reduce, "upcast is for non-reduce")
      check(not (self.tensor_core and self.global_dims <= axis < self.global_dims+len(self.tensor_core.threads)), "can't upcast TC locals")
      check(amt <= 16, "don't upcast more than 16")
      self.shift_to(axis, amt, insert_before=None)
      self.upcast()
    elif opt.op is OptOps.UPCASTMID:                  # white
      check(self.bufs[0].src[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduces != 0 and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1, "invalid upcast mid reduce")  # noqa: E501
      axes = self.sts[0].unit_stride_axes()
      check(len(axes) == 1, f"wrong number of stride 1 axis : {axes}")
      check(axes[0] == axis, "wrong axis")
      check(amt == 4, "don't upcast mid anything but 4")
      self.shift_to(axis, amt, insert_before=self.first_reduce + self.group_for_reduces)
      self.group_for_reduces += 1
    elif opt.op is OptOps.NOLOCALS:
      check(self.opts.has_local and not self.dont_use_locals, "NOLOCALS is meaningless if target does not support local or already not using locals")
      check(self.local_dims == 0 and self.group_for_reduces == 0, "can't have no locals with locals")
      self.dont_use_locals = True
    elif opt.op is OptOps.SWAP:
      check(axis < amt < self.global_dims, f"swap is only for globals with axis < amt, getting {amt=}, {axis=}, {self.global_dims=}")
      permute = list(range(self.shape_len))
      permute[axis], permute[amt] = permute[amt], permute[axis]
      self.reshape_and_permute(None, tuple(permute))
    elif opt.op is OptOps.PADTO:
      check(not self.vars, "does not work with symbolic shape")
      check(axis < self.first_upcast, "cannot pad upcasted")
      # ok to pad SUM if all parent ALU ops have f(0) = 0
      if (r:=self.reduceop) is not None and self.first_reduce <= axis: check(r.arg[0] is Ops.ADD and can_pad(r, {}, set()), f"cannot pad {r}")
      padded = False
      for i,st in enumerate(self.sts):
        if (s:=st.shape[axis]) == 1: continue  # reduced
        check(s > amt//4, f"pad adds more than quadruple the work {st.shape[axis]=} > {amt//4=}")
        if (ru := round_up(cast(int, s), amt) - s):
          # pad right seems to be faster
          self.sts[i] = st.pad(((0,0),) * axis + ((0,ru),) + ((0,0),) * (len(st.shape)-axis-1))
          padded = True
      check(padded, "nothing was padded")

    if append_opt: self.applied_opts.append(opt)
    if self.simplify_ones() and self.tensor_core_opts:
      self.tensor_core_opts.fix_axes(axis) # fix up axes in TC opts if required after simplify_ones()

  def required_optimizations(self) -> Kernel:
    if isinstance(self.membufs[0].dtype, ImageDType):
      unit_stride_axes_mul_4 = [i for i in self.sts[0].unit_stride_axes(ignore_valid=True) if self.sts[0].shape[i]%4 == 0]
      assert unit_stride_axes_mul_4, f"needs a unit stride axis in {self.bufs[0]}"
      if all(x < self.first_upcast for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in self.upcast_in_mid_reduce_axes:
        self.apply_opt(Opt(OptOps.UPCAST, unit_stride_axes_mul_4[0], 4))
    return self

  def hand_coded_optimizations(self) -> Kernel:
    self.required_optimizations()

    # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
    MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
    if self.opts.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
        self.reduceop is not None and self.reduceop.arg[0] is Ops.ADD and len(self.full_shape) >= 2 and self.opts.has_shared and \
        (mulop:=self.reduceop.src[0]).op is Ops.MUL and mulop.src[0].op is Ops.LOAD and mulop.src[1].op is Ops.LOAD:
      st0, st1 = self.sts[self.bufs.index(mulop.src[0])], self.sts[self.bufs.index(mulop.src[1])]
      strides0, strides1 = st0.real_strides(), st1.real_strides()
      def has_expanded_axis(shape, strides): return any(resolve(s > 1) and not resolve(st != 0) for s,st in zip(shape,strides))
      if strides0[self.first_reduce] == 1 and not (has_expanded_axis(st0.shape, strides0) and has_expanded_axis(st1.shape, strides1)):
        for global_idx in range(self.global_dims):
          if self.full_shape[self.first_reduce]%MV_THREADS_PER_ROW == 0 and self.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
            if DEBUG >= 3:
              print(f"MATVEC: {self.full_shape=} {self.first_reduce=} {strides0=} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
            if MV_THREADS_PER_ROW > 1: self.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
            if MV_BLOCKSIZE > 1: self.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
            if MV_ROWS_PER_THREAD > 1: self.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
            return self

    if self.opts.has_local and self.opts.has_shared and all_int(self.sts[0].shape[:self.first_reduce]):
      # are we grouping? (requires local shape support)
      if not self.float4_axis(0) and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:  # noqa: E501
        # TODO: use 1024 if it's allowed in a smarter way
        for sz in ([256, 16] if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
          if all(st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts):
            try: # may fail due to excessive smem usage
              self.apply_opt(Opt(OptOps.GROUPTOP, 0, sz))
              break
            except KernelOptError: pass

      # are we upcasting in mid reduce? (only for images)
      if self.bufs[0].src[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduces and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1:  # noqa: E501
        axes = self.sts[0].unit_stride_axes()
        assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
        if self.sts[0].shape[axes[0]]%4 == 0:
          self.apply_opt(Opt(OptOps.UPCASTMID, axes[0], 4))

    # upcast float4 images
    for buf_index,buf in enumerate(self.bufs):
      unit_stride_axes_mul_4 = [i for i in self.sts[buf_index].unit_stride_axes(ignore_valid=True) if self.sts[buf_index].shape[i]%4 == 0]
      if buf.src[0].dtype.__class__ is ImageDType:
        #assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {self.bufs[buf_index]}"
        if len(unit_stride_axes_mul_4) and all(x < self.first_upcast for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in self.upcast_in_mid_reduce_axes:  # noqa: E501
          if unit_stride_axes_mul_4[0] < self.first_reduce:
            self.apply_opt(Opt(OptOps.UPCAST, unit_stride_axes_mul_4[0], 4))
          else:
            self.apply_opt(Opt(OptOps.UNROLL, unit_stride_axes_mul_4[0]-self.first_reduce, 4))

    # no more opt if we are grouping
    if self.group_for_reduces: return self

    # **** below this line need to be optional and benchmarked ****

    # TODO: doing extra upcasts with images doesn't work for some reason (maybe has to do with to_image_idx)
    # to trigger the above bug, remove prod(self.full_shape[self.first_upcast:]) from the below
    # expression and run test/test_ops.py with IMAGE=2
    # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
    # this can be made much smarter
    to_upcast: List[int] = []
    # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
    for axis in range(self.first_reduce):
      # we might want to be able to split axes that are masked, or refuse to merge them in simplify_merge_adjacent
      # for now skip upcasting here if there is a symbolic axis
      if isinstance(self.full_shape[axis], int) and self.full_shape[axis] <= 7 and any(st.axis_is_masked(axis) for st in self.sts) and \
        prod(self.full_shape[self.first_upcast:]) * prod(self.full_shape[j] for j in to_upcast) * self.full_shape[axis] <= 7 * 7:
        if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
        to_upcast.append(axis)
    for axis in to_upcast[::-1]: self.apply_opt(Opt(OptOps.UPCAST, axis, 0))

    # potentially do more upcasts of non reduce axes based on a heuristic
    upcasted_axis = set()
    while resolve(prod(self.sts[0].shape[:self.first_reduce]) >= 1024):
      xb_choices = []
      for axis, upcast_amount in itertools.product(range(self.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
        # if we haven't upcasted it, it's not symbolic, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
        if axis not in upcasted_axis and isinstance(self.full_shape[axis], int) and self.full_shape[axis]%upcast_amount == 0 and any(st.views[-1].strides[axis] == 0 and not any(x[1] == 0 for x in self.upcasted_axis(buf_index)) for buf_index, st in enumerate(self.sts)):  # noqa: E501
          xb_choices.append((sum(st.views[-1].strides[axis]>0 for st in self.sts), sum(st.views[-1].strides[axis] for st in self.sts), axis, upcast_amount))  # noqa: E501
      if xb_choices:
        xb_choices = sorted(xb_choices)
        if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
        self.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
        upcasted_axis.add(xb_choices[0][2])
      else: break

    # if last dim is small(ish) and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast.
    if self.first_reduce < self.first_upcast and (prod(self.full_shape[self.first_upcast:]) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))) and (self.upcasted == 0 or prod(self.full_shape[-self.upcasted:]) < 64):  # noqa: E501
      if isinstance(s:=self.full_unupcasted_shape[-1], int) and s <= 32:  # NOTE: cannot loop unroll symbolic axis
        self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, 0))
        # if it's small, upcast a second reduce dimension too
        if self.first_reduce < self.first_upcast and s <= 3 and isinstance(s2:=self.full_unupcasted_shape[-1], int) and s2 <= 3:
          self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, 0))
      else:
        for splits in [4]:
          if self.full_unupcasted_shape[-1]%splits == 0:
            self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, splits))
            break

    # if nothing at all is upcasted and it's easy to, do an upcast
    # TODO: this is breaking the tests
    for splits in [4]:
      if self.upcasted == 0 and self.full_unupcasted_shape and self.full_unupcasted_shape[-1] % splits == 0:
        self.apply_opt(Opt(OptOps.UPCAST, len(self.full_unupcasted_shape)-1, splits))

    # **** local groups ****

    if self.opts.has_local:
      if getenv("NOLOCALS") and self.local_dims == 0 and not self.group_for_reduces:
        self.apply_opt(Opt(OptOps.NOLOCALS))
      else:
        # prioritize making expand axes local
        local_axis_ranking = [(any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))), axis) for axis in range(len(self.full_shape[:self.first_reduce]))]  # noqa: E501
        to_local: List[Tuple[int, int]] = []
        for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
          local_size = prod(sz for _, sz in to_local)
          local_sz: Optional[int] = next((x for x in ([32] * (axis == 0) + [16, 8, 4, 3, 2]) if self.full_shape[axis] % x == 0 and local_size * x <= 128), None)  # noqa: E501
          if local_sz is not None: to_local.append((axis, local_sz))
        deleted_shape = 0
        for axis, local_sz in sorted(to_local[:3]):
          axis = axis - deleted_shape
          will_delete_shape = local_sz == self.full_shape[axis]
          self.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
          if will_delete_shape: deleted_shape += 1

    return self

  # **** kernel outputs ****

  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)
  @functools.cached_property
  def name(self) -> str:
    # kernel name (before late upcast)
    kernel_type = "r" if self.reduceop is not None else ("C" if all(x.op is Ops.SINK or x.op in GroupOp.Buffer for x in self.ast.toposort) else "E")
    suffix = colored('_', 'BLACK').join([colored(x.render() if isinstance(x, UOp) else str(x), c) for x,c in zip(self.full_shape, self.colors())])
    name = kernel_type + (f"{len(self.ast.src)}" if len(self.ast.src) > 1 else "") + "_" + suffix

    # name the function something unique
    Kernel.kernel_cnt[(function_name := to_function_name(name))] += 1
    num = f"n{Kernel.kernel_cnt[function_name]-1}" if Kernel.kernel_cnt[function_name] > 1 else ""
    return name + colored(num, 'BLACK')

  def get_optimized_ast(self) -> UOp:
    @functools.lru_cache(None)
    def fixup_ast(op:UOp) -> UOp:
      ret = op.replace(src=tuple(fixup_ast(x) for x in op.src))
      if op.op in GroupOp.Buffer and op in self.bufs:
        st_uop = self.sts[self.bufs.index(op)].to_uop()
        return ret.replace(src=(st_uop,)) if op.op is Ops.VALID else ret.replace(src=(ret.src[0], st_uop, *ret.src[2:]))
      if op.op is Ops.SINK: return ret.replace(arg = KernelInfo(self.local_dims, self.upcasted, self.dont_use_locals))
      if op.op is Ops.REDUCE_AXIS:
        reduce_idx = len(self.bufs) + self.reduceops.index(op) * 2

        def reduced_axes(start, stop):
          return tuple(i for i in range(start, stop) if resolve(self.sts[reduce_idx].shape[i] != self.sts[reduce_idx + 1].shape[i]))
        axes = reduced_axes(self.first_reduce + self.group_for_reduces, self.shape_len)
        grouped_axes = reduced_axes(self.first_reduce, self.first_reduce + self.group_for_reduces)

        if (tc := self.tensor_core) and (self.use_tensor_cores == 1 or self.use_tensor_cores == 3):
          def fix_st(st: ShapeTracker, wd_pattern, tcd_pattern):
            st = ShapeTracker.from_shape(st.shape) # st needs to be contiguous
            wd, warp_dims = self.global_dims,  tuple(sz for _, sz in tc.threads)
            tcd, tcd_dims = self.first_upcast, tuple(sz for _, sz in tc.reduce_axes + tc.early_upcast_axes)

            assert st.shape[wd:wd+len(warp_dims)] == warp_dims, f"warp dims wrong: {st.shape[wd:wd+len(warp_dims)]=} != {warp_dims=}"
            assert st.shape[tcd:tcd+len(tcd_dims)] == tcd_dims, f"tcd dims wrong: {st.shape[tcd:tcd+len(tcd_dims)]=} != {tcd_dims=}"
            assert tc.expanded_shape is not None

            new_shape = st.shape[:tcd] + tc.expanded_shape + st.shape[tcd+len(tcd_dims):]  # expand the tcd
            permaxis = list(range(wd)) + [y + (wd if x == 0 else tcd) for x,y in wd_pattern]  + list(range(wd+len(warp_dims),tcd)) + \
                                         [y + (wd if x == 0 else tcd) for x,y in tcd_pattern] + list(range(tcd+len(tc.expanded_shape),len(new_shape)))
            return st.reshape(new_shape).permute(tuple(permaxis)).reshape(st.shape).simplify()

          srcs = list((ret.src[0] if ret.src[0].op is not Ops.CAST else ret.src[0].src[0]).src)
          for i, tc_pattern in enumerate([tc.st1_pattern, tc.st2_pattern]):
            if tc_pattern: srcs[i] = srcs[i].view(fix_st(srcs[i].st_arg if srcs[i].op is Ops.LOAD else srcs[i].src[0].st_arg, *tc_pattern))

            if self.use_tensor_cores == 3:  # for TC=3, emulate the warp addressing with locals
              local_shape = tuple(1 if i >= self.first_reduce and i < self.first_upcast else s for i, s in enumerate(self.full_shape))
              st = store_st = ShapeTracker.from_shape(local_shape)
              local_buffer = UOp(Ops.DEFINE_LOCAL, tc.dtype_in.ptr(local=True), (), (f"temp{i + 1}", st.real_size()))
              if tc_pattern: store_st = fix_st(store_st, *tc_pattern)
              local_store = UOp.store(local_buffer, store_st.to_uop(), srcs[i])
              srcs[i] = UOp(Ops.LOAD, tc.dtype_in, (local_buffer, st.to_uop(), local_store))

          tc_reduce_axes = tuple(self.first_upcast + ax for ax, _ in tc.reduce_axes)
          if self.use_tensor_cores == 1: # real WMMA, use CONTRACT/EXPAND to get the vectorization right
            upcast_axes = tuple(tuple((self.first_upcast + ax, sz) for ax, sz in up) for up in tc.upcast_axes)
            wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, self.opts.device, prod(sz for _, sz in tc.threads), upcast_axes, tc_reduce_axes)
            wmma_sz = [prod(x[1] for x in l) for l in upcast_axes]
            wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(wmma_sz[2]), src=(
              UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(wmma_sz[0]), src=(srcs[0],), arg=upcast_axes[0]),
              UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(wmma_sz[1]), src=(srcs[1],), arg=upcast_axes[1]),
              UOp.const(tc.dtype_out.vec(wmma_sz[2]), 0.0)), arg=wmma_arg)
            tc_uop = UOp(Ops.EXPAND, tc.dtype_out, (wmma,), arg=upcast_axes[2])

          else: # for TC=3 MUL/SUM instead of WMMA
            tc_uop = UOp(Ops.REDUCE_AXIS, tc.dtype_out, ((srcs[0] * srcs[1]).cast(tc.dtype_out),), (Ops.ADD, tc_reduce_axes))

          new_reduce_axes = tuple(i for i in axes if i not in tc_reduce_axes)
          return ret.replace(src=(tc_uop,), arg=(Ops.ADD, new_reduce_axes)) if new_reduce_axes else tc_uop

        ret = ret.replace(arg = (op.arg[0], axes))
        if self.group_for_reduces and grouped_axes:
          local_shape = (1,) * self.global_dims + self.full_shape[self.global_dims:self.global_dims+self.local_dims] + \
            tuple([self.full_shape[i] if self.sts[reduce_idx].shape[i] != self.sts[reduce_idx+1].shape[i] else 1 \
              for i in range(self.first_reduce, self.first_reduce+self.group_for_reduces)]) + \
            (1,) * (self.shape_len - self.upcasted - self.group_for_reduces - self.first_reduce) + tuple([x[0] for x in self.upcasted_axis(0)])
          st_uop = ShapeTracker.from_shape(local_shape).to_uop()
          local_buffer = UOp(Ops.DEFINE_LOCAL, op.dtype.ptr(local=True), (), (f"temp{self.reduceops.index(op)+1}", st_uop.arg.real_size()))
          local_load = UOp(Ops.LOAD, op.dtype, (local_buffer, st_uop, UOp.store(local_buffer, st_uop, ret)))
          grouped_reduce = UOp(Ops.REDUCE_AXIS, op.dtype, (local_load,), arg=(op.arg[0], grouped_axes))
          if op is self.reduceops[-1]: return grouped_reduce
          st_uop = ShapeTracker.from_shape(tuple([1 if i in grouped_axes else a for i,a in enumerate(local_shape)])).to_uop()
          return UOp(Ops.LOAD, op.dtype, (local_buffer, st_uop, UOp.store(local_buffer, st_uop, grouped_reduce)))

      return ret

    return graph_rewrite(fixup_ast(self.ast), view_left)

  # **** this is the lowerer ****

  @track_rewrites()
  def linearize(self) -> Kernel:
    modified_ast = self.get_optimized_ast()

    if DEBUG >= 3:
      print(self.name)
      if getenv("RAWAST"): print(self.ast)
      print(modified_ast)
      print(self.applied_opts)
    verify_ast(modified_ast)

    self.uops:List[UOp] = linearize_uop(full_graph_rewrite(rewrite_shapetracker_with_index(modified_ast, self.opts), self.opts))
    if DEBUG >= 5: print_uops(self.uops)
    return self

  def to_program(self, name_override:Optional[str]=None) -> ProgramSpec:
    self.linearize()
    src = self.opts.render(name:=to_function_name(ansiname:=(name_override if name_override is not None else self.name)), self.uops)

    if getenv("RUN_PROCESS_REPLAY"):
      from test.external.process_replay.helpers import get_process_replay_ctx
      diskcache_put("kernel_process_replay", str(id(self)), (self.ast, self.opts, self.applied_opts, name, *get_process_replay_ctx(), src))

    # group non-local bufs by the op type (LOAD or STORE) and the buffer arg. take the max access of that buffer in bytes
    # TODO: these max and min don't work on symbolic, and results are very wrong.
    mem_bytes = sum(max(x.src[0].dtype.itemsize * x.st_arg.real_size() for x in group)
      for _, group in itertools.groupby([x for x in self.ast.toposort if x.op in GroupOp.Buffer and x.src[0].op is Ops.DEFINE_GLOBAL],
                        key=lambda x: (x.op, x.src[0].arg)))
    return ProgramSpec(ansiname, src, self.opts.device, self.uops, mem_estimate=mem_bytes,
                   global_size=[1,1,1] if self.opts.has_local else None, local_size=[1,1,1] if self.opts.has_local else None)

# the living definition of intermediate UOps

def _assert_valid_uop(uop:UOp, st:ShapeTracker, sts:Dict[UOp, ShapeTracker]) -> None:
  if not uop.has_st or uop in sts: return
  # restore globals from the two stage reduce
  if uop.op is Ops.LOAD and uop.src[0].op is Ops.DEFINE_LOCAL:
    _assert_valid_uop(local_reduce:=uop.src[2].src[2], uop.st_arg, sts)
    sts[uop] = sts[local_reduce]
    return
  for x in uop.src: _assert_valid_uop(x, st, sts)
  # only reduceuop is allowed to change shape, limited to turning n to 1
  if uop.op in {Ops.REDUCE_AXIS, Ops.WMMA}: st = ShapeTracker.from_shape(sts[uop.src[0]].reduce(uop.axis_arg))
  # movementops are pushed to VIEW
  elif uop.op is Ops.VIEW:
    assert len(uop.src) == 0, f"can't swizzle in kernel yet {uop}"
    st = uop.arg
  # everything else inherits shape
  else:
    if len(src_sts:=[sts[x] for x in uop.src if x in sts]) == 0: return None
    st = src_sts[0]
    if not all_same(shapes:=[x.shape for x in src_sts]):
      if all_same(sizes:=[prod(x) for x in shapes]): raise AssertionError(f"found implicit reshape {shapes}")
      raise AssertionError(f"found implicit expand {sizes} {shapes}")
  sts[uop] = st

def verify_ast(ast:UOp) -> Dict[UOp, ShapeTracker]:
  assert ast.op is Ops.SINK and all(x.op is Ops.STORE for x in ast.src), "must be SINK"
  assert all_same([x.st_arg.size for x in ast.src]), "outputs must be exactly the same size"
  sts: Dict[UOp, ShapeTracker] = {}
  for out in ast.src: _assert_valid_uop(out, out.st_arg, sts)
  shape_dims = [sorted(dedup(dims)) for dims in zip(*[x.shape for x in sts.values()])]
  assert all(len(x) == 1 or (len(x) == 2 and x[0] == 1) for x in shape_dims), f"shapes must have either 1 or n in each dimension, {shape_dims}"
  type_verify(list(sts))
  return sts
