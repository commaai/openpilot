from __future__ import annotations
import os, math, itertools
from typing import NamedTuple, Optional, List, Tuple, cast, Dict, Union
from tinygrad.ops import LazyOp, FlopCounter, get_lazyop_info, UnaryOps, BinaryOps, ReduceOps, MemBuffer, ConstBuffer, BufferOps, Device, Compiled
from tinygrad.helpers import dedup, dtypes, colored, ImageDType, DType, all_int, ansilen, getenv, prod, DEBUG
from tinygrad.shape.shapetracker import ShapeTracker, get_contraction
from tinygrad.shape.symbolic import sint
from tinygrad.shape.view import View, strides_for_shape
from dataclasses import dataclass
from enum import Enum, auto

class OptOps(Enum):
  UPCAST = auto(); UPCASTMID = auto(); UNROLL = auto(); LOCAL = auto(); LASTLOCAL = auto(); GROUP = auto(); GROUPTOP = auto(); NOLOCALS = auto() # noqa: E702
  def __lt__(self, x:OptOps): return self.value < x.value

@dataclass(frozen=True, order=True)
class Opt:
  op: OptOps
  axis: Optional[int] = None
  amt: Optional[int] = None
  def __repr__(self): return f"Opt(op={self.op}, axis={self.axis}, amt={self.amt})"

@dataclass(frozen=True)
class TensorCore:
  device: str
  dims: List[int]
  dtype_in: DType
  dtype_out: DType
  threads: List[Tuple[int,int]] # list of (TC dim,amt) that construct the warp thread structure
  upcast_dim: int # which TC dim to upcast
  thread_local_aliases: List[List[List[int]]] # a list of [threads_1, ..., threads_n, upcast_1(unrolled), upcast_2(upcast)] defining the alias (-1 is upcast, 1-n is warp threads) for each TC dim
  thread_local_sizes: List[int] # in each thread, the number of elements stored in registers for each TC dim
  arch: Optional[str] = None
  def __str__(self): return f"tensor_core<{self.device}, {self.dims}, {self.dtype_in}, {self.dtype_out}>"

tensor_cores: Dict[str, List[TensorCore]] = {
  "METAL": [
    TensorCore(device="METAL", dims=[8,8,8], dtype_in=dtypes.float, dtype_out=dtypes.float, upcast_dim=0, threads=[(0,2),(1,4),(0,2),(1,2)], thread_local_sizes=[2,2,2], thread_local_aliases= [ [[4],[0],[2],[0],[-1, 1, 3],[0]], [[0],[3],[0],[1],[2, 4],[-1]], [[4],[3],[2],[1],[0],[-1]] ], arch="arm64"),
    TensorCore(device="METAL", dims=[8,8,8], dtype_in=dtypes.half,  dtype_out=dtypes.half,  upcast_dim=0, threads=[(0,2),(1,4),(0,2),(1,2)], thread_local_sizes=[2,2,2], thread_local_aliases= [ [[4],[0],[2],[0],[-1, 1, 3],[0]], [[0],[3],[0],[1],[2, 4],[-1]], [[4],[3],[2],[1],[0],[-1]] ], arch="arm64"),
  ],
  "HIP": [
    TensorCore(device="HIP", dims=[16,16,16], dtype_in=dtypes.half, dtype_out=dtypes.float, upcast_dim=1, threads=[(0,16),(1,2)], thread_local_sizes=[16,16,8], thread_local_aliases=[ [[0],[0],[-1],[1]], [[0],[1],[-1],[0]], [[0],[1],[0],[2,-1]] ]),
    TensorCore(device="HIP", dims=[16,16,16], dtype_in=dtypes.half, dtype_out=dtypes.half,  upcast_dim=1, threads=[(0,16),(1,2)], thread_local_sizes=[16,16,8], thread_local_aliases=[ [[0],[0],[-1],[1]], [[0],[1],[-1],[0]], [[0],[1],[0],[2,-1]] ]),
  ]
}

class LocalBuffer(NamedTuple):
  name: str
  size: int
  dtype: DType = dtypes.float32
  realized: None = None
  def __str__(self): return f"localbuffer<{self.name}[{self.size}]>"

class LinearizerOptions(NamedTuple):
  device: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  supports_float4_alu: bool = True
  has_local: bool = True
  has_shared: bool = True
  # NOTE: these two should be in z,y,x(reversed) order for cstyle backends, they are flipped when kernel is rendered
  global_max: Optional[List[int]] = None
  local_max: Optional[List[int]] = None

class Kernel:
  def __init__(self, ast:LazyOp, opts:Optional[LinearizerOptions]=None):
    self.opts = opts if opts else (cast(Compiled, Device[Device.DEFAULT]).linearizer_opts if isinstance(Device[Device.DEFAULT], Compiled) else LinearizerOptions())
    self.ast = ast

    # fetch lazyop info
    self.info: FlopCounter = get_lazyop_info(cast(LazyOp, self.ast))

    # there's only allowed to be one reduceop
    reduceops = [x for x in self.ast.get_lazyops() if x.op in ReduceOps]
    assert len(dedup(reduceops)) <= 1, "max one reduce op in an ast"
    self.reduceop = reduceops[0] if reduceops else None

    # create new shapetrackers inside this kernel, we will permute them
    self.bufs: List[Union[MemBuffer, ConstBuffer, LocalBuffer]] = [MemBuffer(0, self.info.dtype, ShapeTracker.from_shape(self.info.shape))] + dedup([x.arg for x in self.ast.get_lazyops() if x.op in BufferOps])

    # get earlybufs, before the one reduce op
    self.earlybufs = [x.arg for x in self.reduceop.get_lazyops() if x.op in BufferOps] if self.reduceop else []
    self.full_buf_index: int = self.bufs.index(self.earlybufs[0]) if self.earlybufs else 0

    # create the (permuted) shapetrackers
    self.sts: List[ShapeTracker] = [x.st for x in cast(List[Union[MemBuffer, ConstBuffer]], self.bufs)]

    # move all reduce axes to the end
    reduce = list(enumerate(zip(self.full_shape, self.sts[0].shape)))
    permute = tuple([i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])
    self.reshape_and_permute(None, permute)

    # parameters for optimization
    self.applied_opts: List[Opt] = []
    self.group_for_reduce: List[int] = []
    self.upcasted: int = 0
    self.local_dims: int = 0
    self.local_alias: Dict[int, LocalBuffer] = {}
    self.tensor_core: Optional[TensorCore] = None
    self.dont_use_locals: bool = False

    # group simplifies
    self.simplify_ones()
    self.simplify_merge_adjacent()

    # cache
    self.applied_opts_cache: Optional[List[Opt]] = None

  def copy(self):
    ret = type(self).__new__(type(self))

    # base linearizer params
    ret.opts, ret.ast = self.opts, self.ast

    # things downstream of the AST
    # NOTE: we copy bufs for local buffers and sts for optimizations
    ret.info, ret.reduceop, ret.bufs, ret.earlybufs, ret.full_buf_index, ret.sts = \
      self.info, self.reduceop, self.bufs[:], self.earlybufs, self.full_buf_index, self.sts[:]

    # parameters for optimizations
    ret.applied_opts, ret.group_for_reduce, ret.upcasted, ret.local_dims, ret.local_alias, ret.tensor_core, ret.dont_use_locals = \
      self.applied_opts[:], self.group_for_reduce[:], self.upcasted, self.local_dims, self.local_alias.copy(), self.tensor_core, self.dont_use_locals

    # uncached since linearize didn't run
    ret.applied_opts_cache = None

    return ret

  @property
  def membufs(self) -> List[MemBuffer]: return [x for x in self.bufs if isinstance(x, MemBuffer)]

  def has_variable_shape(self) -> bool:
    for b in self.bufs:
      if not isinstance(b, LocalBuffer) and not all_int(b.st.views[-1].shape): return True
    return False

  def shape_offsets(self, i): return itertools.product(*[list(range(s)) for s in self.sts[i].shape[self.shape_len-self.upcasted:][::-1]]) if self.upcasted > 0 else [tuple()]
  def float4_axis(self, i): return [x-(self.shape_len-self.upcasted) for x in self.sts[i].unit_stride_axes() if x >= self.shape_len-self.upcasted and self.sts[i].shape[x]%4 == 0]

  def upcasted_axis(self, i):
    return list(zip(self.sts[i].shape[self.shape_len-self.upcasted:],
                    self.sts[i].real_strides()[self.shape_len-self.upcasted:],
                    [x!=y for x,y in zip(self.sts[0].shape[self.shape_len-self.upcasted:], self.full_shape[self.shape_len-self.upcasted:])]))

  # TODO: is there a better way to write this?
  def acc_offsets(self, i):
    if self.upcasted == 0: return [0]
    upcasted_i = self.upcasted_axis(i)
    acc_strides = [x*(1-upcasted_i[::-1][i][2]) for i,x in enumerate(strides_for_shape(tuple(1 if r else s for s,_,r in upcasted_i[::-1])))]
    return [sum(t) for t in itertools.product(*[[y*acc_strides[i] for y in range(x[0])] for i,x in enumerate(upcasted_i[::-1])])]

  def get_upcast_dim(self, i) -> List[int]:
    should_upcast = self.opts.supports_float4 and (self.bufs[i].dtype in [dtypes.float32, dtypes.float16] or isinstance(self.bufs[i].dtype, ImageDType))
    return [x for x in self.sts[i].unit_stride_axes() if should_upcast and x >= self.shape_len-self.upcasted and self.sts[i].shape[x] > 1]

  @property
  def first_reduce(self) -> int: return [x!=y for x,y in zip(self.sts[0].shape[:self.shape_len-self.upcasted]+(0,), self.full_shape[:self.shape_len-self.upcasted]+(1,))].index(True)

  @property
  def output_shape(self) -> Tuple[sint, ...]: return self.sts[0].shape

  @property
  def full_shape(self) -> Tuple[sint, ...]: return self.sts[self.full_buf_index].shape

  @property
  def full_unupcasted_shape(self) -> Tuple[sint, ...]: return self.full_shape[:self.shape_len-self.upcasted]

  @property
  def shape_len(self) -> int: return len(self.sts[0].shape)

  @property
  def upcast_in_mid_reduce_axes(self) -> List[int]: return [j for j in range(self.first_reduce, self.first_reduce+len(self.group_for_reduce)) if self.full_shape[j] == self.sts[0].shape[j]]

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
    # between first_reduce and first_reduce + group_for_reduce, they are either upcast mid reduce (white), or late upcasted (green)
    colors += ["white" if i in self.upcast_in_mid_reduce_axes else "green" for i in range(self.first_reduce, self.first_reduce + len(self.group_for_reduce))]
    # between first_reduce + group_for_reduce and upcasted, they are reduce (red)
    colors += ["red"] * ((self.shape_len-self.upcasted) - (self.first_reduce + len(self.group_for_reduce)))
    # upcasted dimensions are reduce (magenta) or normal (yellow)
    colors += ["magenta" if self.full_shape[i] != self.sts[0].shape[i] else "yellow" for i in range(self.shape_len-self.upcasted, self.shape_len)]
    assert len(colors) == self.shape_len, "colors size mismatch"
    return colors

  def colored_shape(self, pad=None, dense=False) -> str:
    ret = ' '.join(colored(s, color) for s,color in zip([f"{s:4d}" if isinstance(s, int) and not dense else s for s in self.full_shape], self.colors()))
    if pad: ret += ' '*(pad-ansilen(ret))
    return ret

  # ******************** base simplifiers ********************

  # apply reshape and permute to all shapetrackers
  def reshape_and_permute(self, new_shape_fxn, axis):
    new_sts = []
    for st in self.sts:
      if new_shape_fxn is not None: st = st.reshape(tuple(new_shape_fxn(st.shape)))
      if axis is not None: st = st.permute(tuple(axis))
      new_sts.append(st)
    self.sts = new_sts

  # drops the final dimension
  def upcast(self):
    assert self.full_shape[-1] != 1, "can't upcast a dimension with size 1"
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
      lambda x: list(x[0:axis]) + (([amount, x[axis]//amount] if top else [x[axis]//amount, amount]) if x[axis] > 1 else [1,1]) + list(x[axis+1:]),
      [i for i in range(insert_before) if i != move_axis] + [move_axis] + [i for i in range(insert_before, self.shape_len+1) if i != move_axis])

  # ******************** complex simplifiers ********************

  def simplify_ones(self) -> bool:
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    if self.shape_len == 0: return False
    all_ones = [s==1 for s in self.full_shape]
    self.local_dims -= sum(all_ones[self.first_reduce-self.local_dims:self.first_reduce])
    self.upcasted -= sum(all_ones[self.shape_len-self.upcasted:])
    self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)
    return any(all_ones)

  def simplify_merge_adjacent(self):
    if self.shape_len == 0: return
    shapes, strides = [x.shape for x in self.sts], [x.real_strides() for x in self.sts]

    # if it's an image, insert fake strides such that this fusion doesn't happen across image axes
    if isinstance(self.bufs[0].dtype, ImageDType):
      base_shape = self.bufs[0].dtype.shape
      if shape_idx_groups := get_contraction(self.output_shape, base_shape):
        special_strides: Tuple[int, ...] = tuple()
        for i,g in enumerate(shape_idx_groups):
          shape_piece = tuple(self.output_shape[x] for x in g)
          assert prod(shape_piece) == base_shape[i], f"get_contraction was wrong? {shape_piece} != {base_shape[i]}"
          special_strides += strides_for_shape(shape_piece)
        # adding the fake image shape
        shapes.append(self.output_shape)
        strides.append(special_strides)

    # merge dimensions if we can, multi get_shape_strides
    # TODO: does this always preserve the reduce dimension, NO
    # TODO: move this into shapetracker, with tests!
    rets = [[(shapes[j][0], strides[j][0])] for j in range(len(shapes))]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for j in range(len(shapes)):
        # TODO: added the always mergeability of 1s, is this right? if so, add to shapetracker in the 1 case
        can_merge.append(strides[j][i] is not None and ((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*cast(int, strides[j][i])) or (strides[j][i] == 0 and rets[j][-1][1] == 0)))
      # more can merge than this
      mergeable = all(can_merge) and i != self.first_reduce
      for j in range(len(shapes)):
        if mergeable: rets[j][-1] = (rets[j][-1][0] * shapes[j][i], strides[j][i])
        else: rets[j].append((shapes[j][i], strides[j][i]))

    # do the reshapes
    for i,x in enumerate(rets[:len(self.sts)]): self.sts[i] = self.sts[i].reshape(tuple([y[0] for y in x]))

  # ******************** GPU simplifiers ********************
  def _limit_size(self, x: Tuple[int], max_size: List) -> Tuple[int, ...]:
    new_shape,dims = list(x), len(x)
    for i in range(dims):
      next_idx = (i + 1) % dims
      while new_shape[i] > max_size[i]:
        new_shape[i] = new_shape[i] // 2
        if (new_shape[next_idx] <= max_size[next_idx]):
          new_shape[next_idx] = new_shape[next_idx] * 2
        else:
          next_idx = (next_idx + 1) % dims
          new_shape[next_idx] = new_shape[next_idx] * 2
    return tuple(new_shape)

  def limit_dims_to_max(self, global_max: List[int], local_max: List[int]):
    # Check the global allocation limit, current the global_size will be flipped during codegen
    # and then padded right with 1s if its length < 3 which makes this part a bit awkward to write
    global_dims = self.first_reduce-self.local_dims
    if global_dims > 0:
      if global_max:
        tmp = global_max[:global_dims] + (local_max[:self.local_dims] if local_max else [])
        if max(global_max) < max(self.full_shape[:global_dims]): self.reshape_and_permute(lambda x: self._limit_size(x, tmp + [math.inf] * (len(self.full_shape)-len(tmp))), None)
        assert max(global_max) >= max(self.full_shape[:global_dims]), f"device max allocation {max(self.full_shape[:global_dims])} exceeds global dim maximum {max(global_max)}"
      for i in range(global_dims-1):
        if i < len(global_max) and self.full_shape[i] > global_max[i]:
          order = list(range(len(self.full_shape)))
          order[i], order[global_dims-1] = order[global_dims-1], order[i]
          self.reshape_and_permute(None, order)
          if DEBUG >= 3: print("permuted global dim", order, "due to allocation exceeds global limit")

  def alias_buffer(self, i, pattern):
    assert len(pattern) == len(self.sts[i].shape), f"must include a pattern for each shape {pattern} {self.sts[i].shape}"

    bst = 1
    real_strides = self.sts[i].real_strides()
    shp, stride = [(s if p != 0 else 1) for s,p in zip(self.sts[i].shape, pattern)], [0]*len(pattern)
    for priority in range(1, max(pattern)+1):  # priority. 0 is non local and ignored
      for j,p in enumerate(pattern):
        if priority == p and real_strides[j] != 0:
          stride[j] = bst
          bst *= shp[j]

    self.sts.append(ShapeTracker((View.create(tuple(shp), tuple(stride)),)))
    self.bufs.append(LocalBuffer(name=f"ldata{i}", size=self.sts[-1].size()))
    if DEBUG >= 4: print("aliasing buffer", self.sts[i])
    self.local_alias[i] = cast(LocalBuffer, self.bufs[-1])

  # ******************** high level optimizers ********************

  def apply_tensor_cores(self, use_tensor_cores=1, extra_opts:Optional[List[Opt]]=None):
    if use_tensor_cores and self.opts.has_local and self.reduceop and self.reduceop.op == ReduceOps.SUM and self.opts.device in tensor_cores:
      for tc in tensor_cores[self.opts.device]:
        if not((tc.arch is None or tc.arch == os.uname().machine) and isinstance(self.reduceop.src[0], LazyOp)): continue
        has_cast = tc.dtype_in != tc.dtype_out

        if has_cast and not(isinstance(self.reduceop.src[0], LazyOp) and self.reduceop.src[0].op == UnaryOps.CAST and self.reduceop.src[0].arg[0] == tc.dtype_out): continue
        mul_op = self.reduceop.src[0].src[0] if has_cast else self.reduceop.src[0]

        if not(isinstance(mul_op, LazyOp) and mul_op.op == BinaryOps.MUL): continue
        if not(isinstance(mul_op.src[0], LazyOp) and mul_op.src[0].op == BufferOps.MEM and mul_op.src[0].arg.dtype == tc.dtype_in): continue
        if not(isinstance(mul_op.src[1], LazyOp) and mul_op.src[1].op == BufferOps.MEM and mul_op.src[1].arg.dtype == tc.dtype_in): continue
        buf0, buf1 = self.bufs.index(cast(MemBuffer, mul_op.src[0].arg)), self.bufs.index(cast(MemBuffer, mul_op.src[1].arg))
        buf0_strides, buf1_strides = self.sts[buf0].real_strides(), self.sts[buf1].real_strides()
        axis_buf0 = [(i,self.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides[:self.first_reduce]) if s == 0 and self.full_shape[i]%tc.dims[0] == 0]
        axis_buf1 = [(i,self.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides[:self.first_reduce]) if s == 0 and self.full_shape[i]%tc.dims[1] == 0]

        if not(axis_buf0 and axis_buf1 and self.full_shape[self.first_reduce]%tc.dims[2] == 0 and self.full_shape[self.first_reduce] >= tc.dims[2] and (self.shape_len-self.first_reduce) == 1): continue

        if DEBUG >= 3: print("TENSOR CORES", axis_buf0, axis_buf1, tc)

        s0, s1 = axis_buf0[-1][0], axis_buf1[-1][0] # TODO: select axis in smart way
        s0_exists, s1_exists = True, True
        assert s0 != s1 and self.full_shape[s0]%tc.dims[0] == 0 and self.full_shape[s1]%tc.dims[1] == 0
        def fix(needed, ax):
          nonlocal s0, s1, s0_exists, s1_exists
          if not needed: return
          if s0_exists and ax == s0:
            if s1_exists and s0 < s1: s1 -= 1
            s0_exists = False
          elif s1_exists and ax == s1:
            if s0_exists and s1 < s0: s0 -= 1
            s1_exists = False

        # tensor core -- unroll the reduce dim, upcast input, then create the correct thread pattern
        self.apply_opt(Opt(OptOps.UNROLL, 0, tc.dims[2]))
        self.apply_opt(Opt(OptOps.UPCAST, s0 if tc.upcast_dim == 0 else s1, (tc.dims[0]*tc.dims[2])//prod([a[1] for a in tc.threads])))
        for (tc_dim, tc_amt) in tc.threads:
          fix(self.apply_opt(Opt(OptOps.LASTLOCAL, s0 if tc_dim == 0 else s1, tc_amt)), s0 if tc_dim == 0 else s1)

        # assert tensor core and prevent extra_opts from altering the key shape structure
        if use_tensor_cores == 1: self.tensor_core = tc # TC=2 will do the shape ops without the WMMA

        if extra_opts is not None:
          for opt in extra_opts:
            self.apply_opt(opt)
        else:
          # hand-coded TC opts
          if s1_exists:
            s1_div = [upc for upc in [5,4,3,2,1] if self.full_shape[s1]%upc == 0][0]
            if s1_div != 1: fix(self.apply_opt(Opt(OptOps.UPCAST, s1, s1_div)), s1)
          if s0_exists:
            s0_div = [upc for upc in [5,4,3,2,1] if self.full_shape[s0]%upc == 0][0]
            if s0_div != 1: fix(self.apply_opt(Opt(OptOps.UPCAST, s0, s0_div)), s0)
          if self.tensor_core and s0_exists:
            for upc in [4,2]:
              if self.full_shape[s0] % upc == 0:
                self.apply_opt(Opt(OptOps.LASTLOCAL, s0, upc))
                break

        # alias buffer
        alias_pattern = [0]*(self.global_dims+(self.local_dims-len(tc.threads))) + [2]*(len(tc.threads)) + [0]*(self.shape_len-self.upcasted-self.first_reduce) + [1,1] + [3]*(self.upcasted-2)
        self.alias_buffer(buf0, alias_pattern)
        self.alias_buffer(buf1, alias_pattern)
        return True
    return False

  def apply_opt(self, opt:Opt):
    assert not self.dont_use_locals or opt.op not in {OptOps.LOCAL, OptOps.LASTLOCAL, OptOps.GROUP, OptOps.GROUPTOP, OptOps.UPCASTMID}, "not using locals"
    self.applied_opts.append(opt)
    if opt.axis is not None:
      axis = opt.axis + (self.first_reduce if opt.op == OptOps.UNROLL else (self.first_reduce+len(self.group_for_reduce) if opt.op == OptOps.GROUP or opt.op == OptOps.GROUPTOP else 0))
    else:
      axis = -1
    if opt.amt is not None:
      amt = opt.amt if opt.amt != 0 else self.full_shape[axis]
      assert self.full_shape[axis] % amt == 0, "no longer valid shift"
      assert isinstance(amt, int) and amt != 1, "shift of amt 1 or Node is meaningless"
    else:
      amt = -1
    if opt.op == OptOps.LOCAL:        # cyan
      assert axis < self.first_reduce, "can't local a reduce"
      assert not(self.tensor_core), "can't local with tensor cores"
      self.shift_to(axis, amt, insert_before=self.first_reduce)
      self.local_dims += 1
    elif opt.op == OptOps.LASTLOCAL:  # cyan
      assert axis < self.first_reduce, "can't local a reduce"
      self.shift_to(axis, amt, insert_before=self.first_reduce-self.local_dims)
      self.local_dims += 1
    elif opt.op == OptOps.GROUP:      # green
      assert axis >= self.first_reduce + len(self.group_for_reduce) and axis < self.shape_len-self.upcasted, "must be reduce axis to group"
      assert not(self.tensor_core), "can't group with tensor cores"
      self.shift_to(axis, amt, insert_before=self.first_reduce + len(self.group_for_reduce))
      self.group_for_reduce.append(amt)
    elif opt.op == OptOps.GROUPTOP:   # green
      assert axis >= self.first_reduce + len(self.group_for_reduce) and axis < self.shape_len-self.upcasted, "must be reduce axis to group"
      assert not(self.tensor_core), "can't group with tensor cores"
      self.shift_to(axis, amt, top=True, insert_before=self.first_reduce + len(self.group_for_reduce))
      self.group_for_reduce.append(amt)
    elif opt.op == OptOps.UNROLL:     # purple
      assert axis < self.shape_len-self.upcasted, "can't upcasted already upcasted"
      assert amt <= 32, "don't unroll more than 32"
      self.shift_to(axis, amt, insert_before=None)
      self.upcast()
    elif opt.op == OptOps.UPCAST:     # yellow
      assert axis < self.first_reduce, "upcast is for non-reduce"
      assert amt <= 8, "don't upcast more than 8"
      self.shift_to(axis, amt, insert_before=None)
      self.upcast()
    elif opt.op == OptOps.UPCASTMID:  # white
      assert self.bufs[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduce and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1, "invalid upcast mid reduce"
      axes = self.sts[0].unit_stride_axes()
      assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
      assert axes[0] == axis, "wrong axis"
      assert amt == 4, "don't upcast mid anything but 4"
      self.shift_to(axis, amt, insert_before=self.first_reduce + len(self.group_for_reduce))
      self.group_for_reduce.append(amt)
    elif opt.op == OptOps.NOLOCALS:
      assert self.local_dims == 0 and len(self.group_for_reduce) == 0, "can't have no locals with locals"
      assert not self.dont_use_locals, "already not using locals"
      self.dont_use_locals = True
    return self.simplify_ones()

  def required_optimizations(self, early_only=False):
    for buf_index,buf in enumerate(self.bufs):
      unit_stride_axes_mul_4 = [i for i in self.sts[buf_index].unit_stride_axes(ignore_valid=True) if self.sts[buf_index].shape[i]%4 == 0]
      if (not early_only or buf in self.earlybufs) and self.bufs[buf_index].dtype.__class__ is ImageDType:
        assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {self.bufs[buf_index]}"
        if all(x < (self.shape_len-self.upcasted) for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in self.upcast_in_mid_reduce_axes:
          if unit_stride_axes_mul_4[0] < self.first_reduce:
            self.apply_opt(Opt(OptOps.UPCAST, unit_stride_axes_mul_4[0], 4))
          else:
            self.apply_opt(Opt(OptOps.UNROLL, unit_stride_axes_mul_4[0]-self.first_reduce, 4))

  def hand_coded_optimizations(self):
    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    self.required_optimizations(early_only=True)

    # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
    MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
    if self.opts.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
        self.reduceop and self.reduceop.op == ReduceOps.SUM and len(self.full_shape) >= 2 and self.opts.has_shared and \
        isinstance(self.reduceop.src[0], LazyOp) and self.reduceop.src[0].op == BinaryOps.MUL and \
        self.reduceop.src[0].src[0].op == BufferOps.MEM and self.reduceop.src[0].src[1].op == BufferOps.MEM:
      buf0 = self.bufs.index(cast(LazyOp, self.reduceop.src[0].src[0]).arg)
      buf1 = self.bufs.index(cast(LazyOp, self.reduceop.src[0].src[1]).arg)
      buf0_strides = self.sts[buf0].real_strides()
      buf1_strides = self.sts[buf1].real_strides()
      def has_expanded_axis(s, st): return any(x > 1 and y == 0 for x,y in zip(s,st))
      if buf0_strides[self.first_reduce] == 1 and not (has_expanded_axis(self.sts[buf0].shape, buf0_strides) and has_expanded_axis(self.sts[buf1].shape, buf1_strides)):
        for global_idx in range(self.global_dims):
          if self.full_shape[self.first_reduce]%MV_THREADS_PER_ROW == 0 and self.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
            if DEBUG >= 3: print(f"MATVEC: full_shape={self.full_shape} first_reduce={self.first_reduce} buf0_strides={buf0_strides} blocksize={MV_BLOCKSIZE} threads_per_row={MV_THREADS_PER_ROW} rows_per_thread={MV_ROWS_PER_THREAD}")
            if MV_THREADS_PER_ROW > 1:
              self.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
            if MV_BLOCKSIZE > 1:
              self.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
            if MV_ROWS_PER_THREAD > 1:
              self.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
            return

    if self.opts.has_local and self.opts.has_shared and all(isinstance(s, int) for s in self.sts[0].shape[:self.first_reduce]):
      # are we grouping? (requires local shape support)
      if not self.float4_axis(0) and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
        # TODO: use 1024 if it's allowed in a smarter way
        for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
          if all(st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts):
            self.apply_opt(Opt(OptOps.GROUPTOP, 0, sz))
            break

      # are we upcasting in mid reduce? (only for images)
      if self.bufs[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduce and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1:
        axes = self.sts[0].unit_stride_axes()
        assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
        if self.sts[0].shape[axes[0]]%4 == 0:
          self.apply_opt(Opt(OptOps.UPCASTMID, axes[0], 4))

    # now do everything required
    self.required_optimizations()

    # no more opt if we are grouping
    if self.group_for_reduce: return

    # **** below this line need to be optional and benchmarked ****

    # TODO: doing extra upcasts with images doesn't work for some reason (maybe has to do with to_image_idx)
    # to trigger the above bug, remove prod(self.full_shape[self.shape_len - self.upcasted:]) from the below
    # expression and run test/test_ops.py with IMAGE=2
    # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
    # this can be made much smarter
    to_upcast: List[int] = []
    # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
    for axis in range(self.first_reduce):
      # we might want to be able to split axes that are masked, or refuse to merge them in simplify_merge_adjacent
      # for now skip upcasting here if there is a symbolic axis
      if isinstance(self.full_shape[axis], int) and self.full_shape[axis] <= 7 and any(st.axis_is_masked(axis) for st in self.sts) and \
        prod(self.full_shape[self.shape_len - self.upcasted:]) * prod(self.full_shape[j] for j in to_upcast) * self.full_shape[axis] <= 7 * 7:
        if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
        to_upcast.append(axis)
    for axis in to_upcast[::-1]:
      self.apply_opt(Opt(OptOps.UPCAST, axis, 0))

    # potentially do more upcasts of non reduce axes based on a heuristic
    upcasted_axis = set()
    while prod(self.sts[0].shape[:self.first_reduce]) >= 1024:
      xb_choices = []
      for axis, upcast_amount in itertools.product(range(self.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
        # if we haven't upcasted it, it's not symbolic, it mods, and some buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
        if axis not in upcasted_axis and isinstance(self.full_shape[axis], int) and self.full_shape[axis]%upcast_amount == 0 and any(st.views[-1].strides[axis] == 0 and not any(x[1] == 0 for x in self.upcasted_axis(buf_index)) for buf_index, st in enumerate(self.sts)):
          xb_choices.append((sum(st.views[-1].strides[axis]>0 for st in self.sts), sum(st.views[-1].strides[axis] for st in self.sts), axis, upcast_amount))
      if xb_choices:
        xb_choices = sorted(xb_choices)
        if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
        self.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
        upcasted_axis.add(xb_choices[0][2])
      else:
        break

    # if last dim is small(ish) and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast. NOTE: careful, this has broken VALIDHACKS
    if self.first_reduce < (self.shape_len-self.upcasted) and (len(list(self.shape_offsets(self.full_buf_index))) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))) and (self.upcasted == 0 or prod(self.full_shape[-self.upcasted:]) < 64):
      if (s:=self.full_unupcasted_shape[-1]) <= 32 and isinstance(s, int):  # NOTE: cannot loop unroll symbolic axis
        self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, 0))
        # if it's small, upcast a second reduce dimension too
        if self.first_reduce < (self.shape_len-self.upcasted) and s <= 3 and (s2:=self.full_unupcasted_shape[-1]) <= 3 and isinstance(s2, int):
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
      if getenv("NOLOCALS") and self.local_dims == 0 and not self.group_for_reduce:
        self.apply_opt(Opt(OptOps.NOLOCALS))
      else:
        # prioritize making expand axes local
        local_axis_ranking = [(any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))), axis) for axis in range(len(self.full_shape[:self.first_reduce]))]
        to_local: List[Tuple[int, int]] = []
        for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
          local_size = prod(sz for _, sz in to_local)
          local_sz: Optional[int] = next((x for x in ([32] * (axis == 0) + [16, 8, 4, 3, 2]) if self.full_shape[axis] % x == 0 and local_size * x <= 128), None)
          if local_sz is not None: to_local.append((axis, local_sz))
        deleted_shape = 0
        for axis, local_sz in sorted(to_local[:3]):
          axis = axis - deleted_shape
          will_delete_shape = local_sz == self.full_shape[axis]
          self.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
          if will_delete_shape: deleted_shape += 1
