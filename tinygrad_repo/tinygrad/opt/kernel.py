from __future__ import annotations
import itertools, functools, math
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, cast, Final, Callable, Sequence
from enum import Enum, auto

from tinygrad.uop.ops import GroupOp, KernelInfo, UOp, Ops, can_pad, resolve, Variable, sint, graph_rewrite, smax, AxisType
from tinygrad.uop.spec import type_verify, ast_spec
from tinygrad.device import Device
from tinygrad.opt.tc import TensorCore
from tinygrad.renderer import Renderer
from tinygrad.dtype import ImageDType
from tinygrad.helpers import all_same, colored, ansilen, dedup, prod, round_up, to_function_name, unwrap, DEBUG, TC_SELECT, TC_OPT, AMX
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import strides_for_shape, get_contraction
from tinygrad.kernelize.kernelize import view_left

class OptOps(Enum):
  TC = auto(); UPCAST = auto(); UNROLL = auto(); LOCAL = auto() # noqa: E702
  GROUP = auto(); GROUPTOP = auto(); NOLOCALS = auto(); PADTO = auto(); SWAP = auto() # noqa: E702
  def __lt__(self, x:OptOps): return self.value < x.value

@dataclass(frozen=True, order=True)
class Opt:
  op: OptOps
  axis: Optional[int] = None
  arg: Optional[int | tuple] = None
  def __repr__(self): return f"Opt(op={self.op}, axis={self.axis}, arg={self.arg})"

axis_letters = {AxisType.GLOBAL: "g", AxisType.LOCAL: "l", AxisType.LOOP: "L", AxisType.UPCAST: "u",
                AxisType.GROUP_REDUCE: "G", AxisType.REDUCE: "R", AxisType.UNROLL: "r"}
axis_colors = {AxisType.GLOBAL: "blue", AxisType.LOCAL: "cyan", AxisType.LOOP: "WHITE", AxisType.UPCAST: "yellow",
               AxisType.GROUP_REDUCE: "green", AxisType.REDUCE: "red", AxisType.UNROLL: "magenta"}

class KernelOptError(Exception): pass
def check(cond:bool, msg:str=""):
  if not cond: raise KernelOptError(msg)

@dataclass
class TensorCoreOptions:
  axes: tuple[int, ...] # the location of the original N and M axes if still in the shape
  axes_exist: tuple[bool, ...] # true if the original N and M axes are still in the shape
  axis_pads: tuple[tuple[int, int], ...]
  def fix_axes(self, removed_axis:int): # adjust the TC axes if necessary when a dimension is removed
    axes, axes_exist = list(self.axes), list(self.axes_exist)
    for tc_dim in [i for i in range(2) if axes_exist[i]]:
      if removed_axis < axes[tc_dim]: axes[tc_dim] -= 1
      elif removed_axis == axes[tc_dim]: axes_exist[tc_dim] = False
    self.axes, self.axes_exist = tuple(axes), tuple(axes_exist)

class Kernel:
  def __init__(self, ast:UOp, opts:Optional[Renderer]=None):
    assert ast.op is Ops.SINK, ast.op
    self.ast = ast

    self.opts = opts if opts is not None else Device[Device.DEFAULT].renderer
    # verify AST matches the spec
    if __debug__: type_verify(list(self.ast.toposort()), ast_spec)

    self.vars: list[Variable] = self.ast.variables()
    # NOTE: this requires a specific order with the [::-1], this is likely a bug
    self.bufs: list[UOp] = [x for x in self.ast.toposort() if x.op in GroupOp.Buffer][::-1]

    # create new shapetrackers inside this kernel, we will permute them
    self.sts: list[ShapeTracker] = [x.st_arg for x in self.bufs]

    # add the shapetrackers for each reduce
    # we use this to track which axes are reduced in each reduce
    self.reduceops = [x for x in self.ast.toposort() if x.op is Ops.REDUCE_AXIS]
    for x in self.reduceops:
      self.sts.append(unwrap(x.st))
      self.sts.append(unwrap(x.src[0].st))

    # add a shapetracker to the end to track the full shape, with 0 strides so it can merge
    self.sts.append(ShapeTracker.from_shape(tuple([smax(*s) for s in zip(*[x.shape for x in self.sts])]), (0,)*len(self.sts[0].shape)))

    # parameters for optimization
    self.tensor_core: Optional[TensorCore] = None
    self.tensor_core_opts: Optional[TensorCoreOptions] = None
    self.use_tensor_cores: int = 0
    self.applied_opts: list[Opt] = []
    self.dont_use_locals = False
    self.finalized: bool = False

    # group simplifies
    self.simplify_ones()
    self.simplify_merge_adjacent()

    # axis types
    global_loops = AxisType.GLOBAL if self.opts.has_local else AxisType.LOOP
    self.axis_types: list[AxisType] = [AxisType.REDUCE if resolve(x!=y) else global_loops for x,y in zip(self.sts[0].shape, self.sts[-1].shape)]

    # confirm all reduce axes are at the end
    final_reduces = [i for i,(s,n) in enumerate(zip(self.full_shape, self.output_shape)) if resolve(s != n)]
    if final_reduces != list(range(len(self.full_shape)-len(final_reduces), len(self.full_shape))):
      raise RuntimeError(f"reduces are not at the end of the shape {self.full_shape} -> {self.output_shape}")

  def copy(self):
    ret = type(self).__new__(type(self))

    # base linearizer params
    ret.opts, ret.ast = self.opts, self.ast

    # things downstream of the AST
    ret.reduceops, ret.vars, ret.bufs = self.reduceops, self.vars, self.bufs
    ret.sts = self.sts[:]
    ret.axis_types = self.axis_types[:]

    # parameters for optimizations
    ret.applied_opts, ret.dont_use_locals = self.applied_opts[:], self.dont_use_locals
    ret.tensor_core, ret.tensor_core_opts, ret.use_tensor_cores = self.tensor_core, self.tensor_core_opts, self.use_tensor_cores
    ret.finalized = self.finalized

    return ret

  @property
  def first_reduce(self) -> int:
    for i in range(self.first_upcast):
      if self.axis_types[i] in (AxisType.GROUP_REDUCE, AxisType.REDUCE): return i
    return self.first_upcast
  @property
  def first_upcast(self) -> int: return self.shape_len-self.upcasted

  @property
  def reduceop(self) -> UOp|None: return self.reduceops[0] if len(self.reduceops) > 0 else None
  @property
  def full_shape(self) -> tuple[sint, ...]: return self.sts[-1].shape

  @property
  def output_shape(self) -> tuple[sint, ...]: return self.sts[0].shape
  @property
  def shape_len(self) -> int: return len(self.sts[0].shape)

  @property
  def global_dims(self) -> int: return sum([1 for x in self.axis_types if x == AxisType.GLOBAL]) if hasattr(self, 'axis_types') else 0
  @property
  def local_dims(self) -> int: return sum([1 for x in self.axis_types if x == AxisType.LOCAL]) if hasattr(self, 'axis_types') else 0
  @property
  def upcasted(self) -> int: return sum([1 for x in self.axis_types if x in {AxisType.UPCAST, AxisType.UNROLL}]) if hasattr(self, 'axis_types') else 0
  @property
  def group_for_reduces(self) -> int: return sum([1 for x in self.axis_types if x == AxisType.GROUP_REDUCE]) if hasattr(self, 'axis_types') else 0

  # ******************** colors and names ********************

  def colors(self) -> list[str]:
    assert len(self.axis_types) == self.shape_len, "colors size mismatch"
    return [axis_colors[x] if not self.dont_use_locals or not x == AxisType.GLOBAL else "BLUE" for x in self.axis_types]

  def colored_shape(self, pad:Optional[int]=None, dense=False) -> str:
    shape_strs = [(s if dense else f"{s:4d}") if isinstance(s, int) else s.render() for s in self.full_shape]
    ret = ' '.join(colored(s, color) for s,color in zip(shape_strs, self.colors()))
    if pad: ret += ' '*(pad-ansilen(ret))
    return ret

  kernel_cnt: Final[defaultdict[str, int]] = defaultdict(int)
  @functools.cached_property
  def name(self) -> str:
    # kernel name (before late upcast)
    kernel_type = "r" if self.reduceop is not None else ("C" if all(x.op is Ops.SINK or x.op in GroupOp.Buffer for x in self.ast.toposort()) else "E")
    suffix = colored('_', 'BLACK').join([colored(x.render() if isinstance(x, UOp) else str(x), c) for x,c in zip(self.full_shape, self.colors())])
    name = kernel_type + (f"{len(self.ast.src)}" if len(self.ast.src) > 1 else "") + "_" + suffix

    # name the function something unique
    Kernel.kernel_cnt[(function_name := to_function_name(name))] += 1
    num = f"n{Kernel.kernel_cnt[function_name]-1}" if Kernel.kernel_cnt[function_name] > 1 else ""
    return name + colored(num, 'BLACK')

  # ******************** base simplifiers ********************

  # apply reshape and permute to all shapetrackers
  def reshape(self, new_shape_fxn:Callable[[tuple[sint, ...]], Sequence[sint]]):
    self.sts = [st.reshape(tuple(new_shape_fxn(st.shape))) for st in self.sts]
  def permute(self, new_axes:Sequence[int]): self.sts = [st.permute(tuple(new_axes)) for st in self.sts]

  # axis : the axis to pull from
  # amount : the amount to take
  # top : if you want to pull that amount from the top
  # insert_before : place to insert the new stuff
  def shift_to(self, axis:int, amount:int, new_type:AxisType, top:bool=False, insert_before:int|None=None):
    if insert_before is None: insert_before = self.shape_len
    self.axis_types.insert(insert_before, new_type)
    move_axis = axis if top else axis+1
    if move_axis < insert_before: insert_before += 1
    def new_shape_fxn(x): return x[0:axis] + (((amount,x[axis]//amount) if top else (x[axis]//amount,amount)) if x[axis] > 1 else (1,1)) + x[axis+1:]
    new_axes = [i for i in range(insert_before) if i != move_axis]+[move_axis]+[i for i in range(insert_before, self.shape_len+1) if i != move_axis]
    self.reshape(new_shape_fxn)
    self.permute(new_axes)

  # ******************** complex simplifiers ********************

  def simplify_ones(self) -> bool:
    # remove places where the shape is all ones
    if any(all_ones:=[s==1 for s in self.full_shape]):
      if hasattr(self, 'axis_types'):
        self.axis_types = [x for i,x in enumerate(self.axis_types) if not all_ones[i]]
      self.reshape(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]])
      return True
    return False

  def simplify_merge_adjacent(self):
    if self.shape_len == 0: return
    shapes, strides = [x.shape for x in self.sts], [x.real_strides() for x in self.sts]
    # NOTE: we can't use self.first_reduce yet
    first_reduce = [resolve(x!=y) for x,y in zip(self.sts[0].shape[:self.first_upcast]+(0,), self.full_shape[:self.first_upcast]+(1,))].index(True)

    # if it's an image, insert fake strides such that this fusion doesn't happen across image axes
    # TODO: remove membufs
    membufs = dedup([x.src[0].base for x in self.bufs if x.op in {Ops.LOAD, Ops.STORE}])
    if isinstance(membufs[0].base.dtype, ImageDType):
      base_shape = membufs[0].base.dtype.shape
      if shape_idx_groups := get_contraction(self.output_shape, base_shape):
        special_strides: tuple[sint, ...] = tuple()
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
      mergeable = all(can_merge) and i != first_reduce
      for j,(s,st) in enumerate(zip(shapes, strides)):
        if mergeable: rets[j][-1] = (rets[j][-1][0] * s[i], st[i])
        else: rets[j].append((s[i], st[i]))

    # do the reshapes
    for i,x in enumerate(rets[:len(self.sts)]): self.sts[i] = self.sts[i].reshape(tuple([y[0] for y in x]))

  # ******************** apply optimizations ********************

  def real_axis(self, opt:Opt):
    if opt.axis is None: return -1
    if opt.op is OptOps.UNROLL: return self.first_reduce+opt.axis
    if opt.op in {OptOps.GROUP, OptOps.GROUPTOP}: return self.first_reduce+self.group_for_reduces+opt.axis
    return opt.axis

  def apply_opt(self, opt:Opt, append_opt:bool=True):
    if self.finalized: raise RuntimeError("can't optimize Kernel after it's finalized")
    if self.dont_use_locals: check(opt.op not in {OptOps.LOCAL, OptOps.GROUP, OptOps.GROUPTOP}, "not using locals")

    if opt.op is OptOps.TC:
      check(len(self.applied_opts) == 0, "tensor core opts must be first") # TODO: things like PADTO might be fine
      check(len(self.opts.tensor_cores) > 0, "must have tensor cores")
      check(opt.axis is not None, "tensor core opts must have an axis")
      check(opt.arg is not None and isinstance(opt.arg, tuple) and len(opt.arg) == 3, "tensor core opts must have valid arg")
      check(-1 <= (tc_select:=cast(tuple, opt.arg)[0]) < len(self.opts.tensor_cores), "tensor core opts must have valid tc_select")
      check(0 <= (tc_opt:=cast(tuple, opt.arg)[1]) <= 2, "tensor core opts must have valid tc_opt")
      check(0 < (use_tensor_cores:=cast(tuple, opt.arg)[2]) <= 2, "use_tensor_cores value is not valid")
      check(self._apply_tc_opt(use_tensor_cores, cast(int, opt.axis), tc_select, tc_opt), "no tensor core available")
      self.applied_opts.append(opt)
      return

    axis = self.real_axis(opt)
    check(axis < len(self.full_shape), "invalid axis")

    if opt.op is OptOps.SWAP: amt = cast(int, opt.arg)  # arg is an axis in the SWAPs
    elif opt.arg is not None:
      check(isinstance(opt.arg, int), "arg should be int")
      amt = arg if (arg:=cast(int, opt.arg)) != 0 else self.full_shape[axis]
      check(isinstance(amt, int) and amt != 1, f"shift/padto of {amt=}, 1 or symbolic amount is meaningless")
      if opt.op is not OptOps.PADTO: check(self.full_shape[axis] % amt == 0, f"no longer valid shift {self.full_shape[axis]=}, {amt=}")
    else: amt = -1

    if self.reduceop is not None and (opt.op in {OptOps.GROUP, OptOps.GROUPTOP} or \
                                      (self.group_for_reduces and opt.op not in {OptOps.NOLOCALS, OptOps.PADTO})):
      acc_sz = self.reduceop.dtype.itemsize
      upcast_sz = prod([a for a,b in zip(self.full_shape[self.first_upcast:], self.sts[0].shape[self.first_upcast:]) if a == b])
      local_sz = prod(self.full_shape[self.first_reduce-self.local_dims:self.first_reduce+self.group_for_reduces])
      smem_sz = amt*acc_sz*upcast_sz*local_sz
      check(smem_sz <= self.opts.shared_max, f"exceeds maximum shared memory size: needs {smem_sz}, max {self.opts.shared_max}")

    if opt.op is OptOps.LOCAL:    # cyan
      # NOTE: LLVM/CPU can use locals too, but they are treated the same as globals (still helpful for L1 cache)
      # it's disabled for now since it makes BEAM slow for little gain
      check(self.opts.has_local, "target does not support local")
      check(axis < self.global_dims, "local is for globals")
      self.shift_to(axis, amt, AxisType.LOCAL, insert_before=self.first_reduce)
    elif opt.op in {OptOps.GROUP, OptOps.GROUPTOP}:   # green
      check(self.opts.has_local and self.opts.has_shared, "target does not support local or shared mem")
      check(self.first_reduce + self.group_for_reduces <= axis < self.first_upcast, "must be reduce axis to group")
      check(not self.tensor_core, "can't group with tensor cores")
      check(len(reduce_axes:=[i for r in self.reduceops for i in r.axis_arg]) == len(set(reduce_axes)), "can't group with parallel reduces")
      self.shift_to(axis, amt, AxisType.GROUP_REDUCE, top=(opt.op is OptOps.GROUPTOP), insert_before=self.first_reduce + self.group_for_reduces)
    elif opt.op is OptOps.UNROLL:                     # purple
      check(axis < self.first_upcast, "can't upcasted already upcasted")
      check(amt <= 32, "don't unroll more than 32")
      self.shift_to(axis, amt, AxisType.UNROLL, insert_before=None)
    elif opt.op is OptOps.UPCAST:                     # yellow
      check(axis < self.first_reduce, "upcast is for non-reduce")
      check(not (self.tensor_core and self.global_dims <= axis < self.global_dims+len(self.tensor_core.get_local_axes())), "can't upcast TC locals")
      check((self.opts is not None and self.opts.device == "DSP") or amt <= 16, "don't upcast more than 16")
      self.shift_to(axis, amt, AxisType.UPCAST, insert_before=None)
    elif opt.op is OptOps.NOLOCALS:
      check(self.opts.has_local and not self.dont_use_locals, "NOLOCALS is meaningless if target does not support local or already not using locals")
      check(self.local_dims == 0 and self.group_for_reduces == 0, "can't have no locals with locals")
      self.dont_use_locals = True
    elif opt.op is OptOps.SWAP:
      check(axis < amt < self.global_dims, f"swap is only for globals with axis < amt, getting {amt=}, {axis=}, {self.global_dims=}")
      permute = list(range(self.shape_len))
      permute[axis], permute[amt] = permute[amt], permute[axis]
      self.permute(tuple(permute))
    elif opt.op is OptOps.PADTO:
      check(not self.vars, "does not work with symbolic shape")
      check(axis < self.first_upcast, "cannot pad upcasted")
      # ok to pad SUM if all parent ALU ops have f(0) = 0
      if (r:=self.reduceop) is not None and self.first_reduce <= axis: check(r.arg[0] is Ops.ADD and can_pad(r, {}), f"cannot pad {r}")
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

  def apply_opts(self, opts:Sequence[Opt]) -> Kernel:
    for opt in opts: self.apply_opt(opt)
    return self

  # **** kernel outputs, mostly tensor cores ****

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
    if DEBUG >= 3: print("TENSOR CORES", axis_buf0, axis_buf1, tc)
    return TensorCoreOptions(axes=(s0, s1, s2), axes_exist=(True, True), axis_pads=axis_pads)

  def _apply_tc_opt(self, use_tensor_cores:int, axis:int, tc_select:int, opt_level:int) -> bool:
    if use_tensor_cores and self.reduceop is not None and self.reduceop.arg[0] is Ops.ADD:
      tensor_cores = self.opts.tensor_cores if tc_select == -1 else [self.opts.tensor_cores[tc_select]]
      for tc in tensor_cores:
        tensor_core_opts = [self._create_tc_opts(reduceop, tc, axis, opt_level) for reduceop in self.reduceops]
        # can only fuse reduces with the same tc options
        assert all_same(tensor_core_opts)
        if tensor_core_opts[0] is None: continue
        self.tensor_core_opts = tc_opts = tensor_core_opts[0]

        # attempt to pad the tensor axes that require it
        try:
          for axis, dim in tc_opts.axis_pads: self.apply_opt(Opt(OptOps.PADTO, axis, dim), append_opt=False) # PADTO might fail
        except KernelOptError: continue
        # tensor core -- unroll the reduce dim (K), upcast and local the inner and outer dims (N, M)
        for opt in tc.opts: self.apply_opt(Opt({"u":OptOps.UPCAST, "l":OptOps.LOCAL}[opt[0]], tc_opts.axes[int(opt[1])], 2), append_opt=False)
        for dim, amt in tc.get_reduce_axes(): self.apply_opt(Opt(OptOps.UNROLL, 0, amt), append_opt=False) # TODO: this should be the reduce, not 0
        self.tensor_core = tc
        self.use_tensor_cores = use_tensor_cores  # TC=2 will do the shape ops without the WMMA
        return True
    return False

  def apply_tensor_cores(self, use_tensor_cores=1, extra_opts:Optional[list[Opt]]=None, axis:int=0, tc_select:Optional[int]=None,
                         tc_opt:Optional[int]=None) -> bool:
    """ Attempts to apply a tensor core optimization to the kernel. If one exists and applies properly, return true, otherwise return false.
    Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).

    Keyword arguments:
    use_tensor_cores -- controls how tensor cores are applied (default 1)
      0: will disable any tensor core matching
      1: enable tensor cores
      2: apply tensor core shape but don't use UOp.WMMA
    extra_opts -- additional Opt's to apply after the tensor core instead of the hand-coded additional Opt's (default None)
    tc_select -- specifies which tensor core(s) to use for optimization (default -1)
      -1: iterates through all available tensor cores in order and uses the first one that matches the requirements (dims and dtypes)
      [0-N]: uses only the n'th tensor core available; useful for search
    tc_opt -- controls which kinds of kernels may be eligible for tensor cores application (default 2 during BEAM, 0 otherwise)
      0: applies to only kernels with a single reduce axis and direct Ops.LOAD into Ops.MUL
      1: allows kernels with multiple reduce axes and also multiplication of Ops.CAST'd buffers
      2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
    """
    if tc_select is None: tc_select = TC_SELECT.value
    if tc_opt is None: tc_opt = TC_OPT.value
    if not self.opts.tensor_cores: return False
    try: # check TC first and apply hand-coded opts if successful
      self.apply_opt(Opt(OptOps.TC, axis, (tc_select, tc_opt, use_tensor_cores)))

      if (tc_opts:=self.tensor_core_opts) is not None:
        if extra_opts is not None: self.apply_opts(extra_opts)
        else:
          if AMX: return True # skip hand-coded TC opts if AMX, upcasting will make kernel slower
          # hand-coded TC opts
          for tc_dim in [tc_dim for tc_dim in [1,0] if tc_opts.axes_exist[tc_dim]]: # attempt to upcast M and N
            szs = [sz for sz in [5,4,3,2] if self.full_shape[tc_opts.axes[tc_dim]] % sz == 0]
            if szs: self.apply_opt(Opt(OptOps.UPCAST, tc_opts.axes[tc_dim], szs[0]))

          if tc_opts.axes_exist[0] and (szs := [sz for sz in [4,2] if self.full_shape[tc_opts.axes[0]] % sz == 0]): # attempt to local N
            self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[0], szs[0]))
      return True
    except KernelOptError:
      return False

  # strings like ['g0', 'g1', 'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'R0', 'r0', 'r1', 'r2', 'u0', 'u1', 'u2']
  def shape_str(self) -> list[str]:
    ret: list[str] = []
    cnt: dict[AxisType, int] = {}
    for x in self.axis_types:
      cnt[x] = (cnt[x] + 1) if x in cnt else 0
      ret.append(f"{axis_letters[x]}{cnt[x]}")
    return ret
  def shape_str_to_axis(self, nms:list[str]) -> tuple[int, ...]: return tuple([self.shape_str().index(x) for x in nms])

  def get_optimized_ast(self, name_override:Optional[str]=None) -> UOp:
    @functools.cache
    def fixup_ast(op:UOp) -> UOp:
      ret = op.replace(src=tuple(fixup_ast(x) for x in op.src)) # noqa: F821
      if op.op in GroupOp.Buffer and op in self.bufs:
        st = self.sts[self.bufs.index(op)]
        # NOTE: if CONST got masked after applying opts, we create a new VALID
        if op.op is Ops.CONST and any(v.mask is not None for v in st.views): return op.view(st).valid()
        # otherwise we just replace the VIEW source
        return ret.replace(src=(ret.src[0].replace(arg=st),)+ret.src[1:])
      if op.op is Ops.SINK:
        # NOTE: should group_for_reduces be added to the local_dims?
        kernel_name = ret.arg.name if ret.arg is not None else self.name if name_override is None else name_override
        return ret.replace(arg=KernelInfo(kernel_name, tuple(self.axis_types), self.dont_use_locals, tuple(self.applied_opts)))
      if op.op is Ops.REDUCE_AXIS:
        reduce_idx = len(self.bufs) + self.reduceops.index(op) * 2
        axes = tuple(i for i in range(0, self.shape_len) if self.axis_types[i] in {AxisType.REDUCE, AxisType.UNROLL} and
                             resolve(self.sts[reduce_idx].shape[i] != self.sts[reduce_idx + 1].shape[i]))
        grouped_axes = tuple(i for i in range(0, self.shape_len) if self.axis_types[i] is AxisType.GROUP_REDUCE and
                             resolve(self.sts[reduce_idx].shape[i] != self.sts[reduce_idx + 1].shape[i]))
        if (tc := self.tensor_core) and self.use_tensor_cores == 1:
          # get reduce/upcast axes for the tensor cores
          tc_reduce_axes = self.shape_str_to_axis([f"r{i}" for i in range(len(tc.get_reduce_axes()))])
          base_upcast_axes = tuple([(s,2) for s in self.shape_str_to_axis([f"r{i}" for i in range(len(tc.get_reduce_axes()))] + \
                                                                          [f"u{i}" for i in range(len(tc.get_upcast_axes()))])])[::-1]
          tc_upcast_axes = tuple([base_upcast_axes[:int(math.log2(tc.elements_per_thread[i]))] for i in range(3)])

          # permute the srcs
          srcs = list((ret.src[0] if ret.src[0].op is not Ops.CAST else ret.src[0].src[0]).src)
          for i, (src, permaxis) in enumerate(zip(srcs, tc.permutes_for_shape_str(self.shape_str()))):
            src_st = (src if src.op is Ops.LOAD else src.src[0]).st_arg
            srcs[i] = src.view(ShapeTracker.from_shape(src_st.shape).permute(tuple(permaxis)))

          # construct the op
          wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, self.opts.device, tc.threads, tc_upcast_axes, tc_reduce_axes)
          wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
            UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0]),
            UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1]),
            UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg)
          tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2])

          # preserve any other reduce
          return ret.replace(src=(tc_uop,), arg=(Ops.ADD, new_axes)) if (new_axes := tuple(i for i in axes if i not in tc_reduce_axes)) else tc_uop

        ret = ret.replace(arg = (op.arg[0], axes))
        if self.group_for_reduces and grouped_axes:
          local_shape = tuple([s if self.axis_types[i] not in (AxisType.GLOBAL, AxisType.REDUCE, AxisType.UNROLL) and \
                               (self.axis_types[i] is not AxisType.GROUP_REDUCE or i in grouped_axes) else 1 for i,s in enumerate(self.full_shape)])
          st = ShapeTracker.from_shape(local_shape).expand(self.full_shape[:self.global_dims]+local_shape[self.global_dims:])
          local_size = st.real_size()
          local_buffer = UOp(Ops.DEFINE_LOCAL, op.dtype.ptr(local_size, local=True), (), f"temp{self.reduceops.index(op)}")
          local_load = local_buffer.view(st).load(local_buffer.view(st).store(ret))
          grouped_reduce = UOp(Ops.REDUCE_AXIS, op.dtype, (local_load,), arg=(op.arg[0], grouped_axes))
          if op is self.reduceops[-1]: return grouped_reduce
          st = ShapeTracker.from_shape(tuple([1 if i in grouped_axes else a for i,a in enumerate(local_shape)]))
          return local_buffer.view(st).load(local_buffer.view(st).store(grouped_reduce))

      return ret
    self.finalized = True
    fixed_ast = fixup_ast(self.ast)
    del fixup_ast
    return graph_rewrite(fixed_ast, view_left, name="fixup optimized AST")
