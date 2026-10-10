from __future__ import annotations
import itertools
from typing import cast
from tinygrad.uop.ops import Ops, UOp, KernelInfo, graph_rewrite, AxisType, ssimplify, identity_element
from tinygrad.uop.ops import axis_colors
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.helpers import colored, getenv, DEBUG, NOOPT, round_up, prod, get_single_element
from tinygrad.helpers import ALLOW_TF32, count, Context
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError, check
from tinygrad.codegen.simplify import pm_flatten_range
from tinygrad.renderer import Renderer

split_targets = {AxisType.UPCAST: (AxisType.GLOBAL, AxisType.LOCAL, AxisType.WEAK),
                 AxisType.LOCAL: (AxisType.GLOBAL, AxisType.WEAK)}

class Scheduler:
  def __init__(self, ast:UOp, ren:Renderer):
    self.ast, self.ren = graph_rewrite(ast, pm_flatten_range), ren
    self.applied_opts = list(self.ast.arg.applied_opts) if self.ast.arg is not None else []
    self.opt_range = count(start=max([x.axis_id[0] for x in self.ast.backward_slice if x.op is Ops.RANGE], default=0)+1)

  @property
  def rngs(self):
    # Keep serial/vectorized reduction axes last, deriving their role from REDUCE rather than the axis type.
    # void RANGEs are loops, not opt axes. the DEVICE axis is launched, not an opt axis
    red = self.reduce_ranges
    return sorted([u for u in self.ast.backward_slice if u.op is Ops.RANGE and u.dtype is not dtypes.void and u.vmax > 0
                   and u.axis_type is not AxisType.DEVICE],
                  key=lambda x: (x in red and x.axis_type in (AxisType.WEAK, AxisType.UPCAST), x.arg))
  @property
  def shape_len(self) -> int: return len(self.rngs)
  @property
  def full_shape(self): return [ssimplify(x.src[0]) for x in self.rngs]
  @property
  def axis_types(self) -> list[AxisType]: return [x.axis_type for x in self.rngs]

  def copy(self) -> Scheduler:
    ret = Scheduler(self.ast, self.ren)
    ret.applied_opts = self.applied_opts[:]
    return ret

  def get_optimized_ast(self, name_override:str|None=None) -> UOp:
    if name_override is not None: name = name_override
    else:
      k_type = "r" if self.reduceop is not None else "E"
      special_uops = sorted([x for x in self.ast.backward_slice if x.op is Ops.SPECIAL], key=lambda x: x.arg)
      special_ops = [colored(str(x.vmax+1), "blue" if x.arg[0] == "g" else "cyan") for x in special_uops]
      name = k_type + colored('_', 'BLACK').join(['']+special_ops+[colored(x.src[0].render(), color) for x,color in zip(self.rngs, self.colors())])
    return self.ast.replace(arg=KernelInfo(name=name, applied_opts=tuple(self.applied_opts)), tag=1)

  def convert_loop_to_global(self) -> None:
    if not self.ren.has_local: return
    red = self.reduce_ranges
    rngs = [r for s in self.ast.src if s.op is Ops.END for r in s.src[1:] if r.axis_type is AxisType.WEAK and r not in red]
    # exclude any output ranges from global that don't appear in all BUFFERIZE
    for x in self.ast.backward_slice:
      if x.op is Ops.STAGE:
        rngs = [r for r in rngs if r in x.ranges]
    self.ast = self.ast.substitute({r:r.replace(arg=(AxisType.GLOBAL,)+r.axis_id) for r in self.rngs if r in rngs})

  def colors(self) -> list[str]: return [axis_colors[t] for t in self.axis_types]
  def colored_shape(self) -> str: return ' '.join([colored(f'{x.src[0].render():>4s}', color) for x,color in zip(self.rngs, self.colors())])

  def shift_to(self, rng:UOp, amount:int, new_type:AxisType, top:bool=False, input_new_rng:UOp|None=None):
    check(rng.axis_type in split_targets[new_type], f"{new_type} is from {split_targets[new_type]}, not {rng.axis_type}")
    if (old_sz:=rng.src[0].divides(amount)) is None:
      raise KernelOptError(f"{amount} can't divide {rng.src[0]} in {self.colored_shape()}")
    new_rng = UOp.range(amount, next(self.opt_range), new_type, dtype=rng.dtype) if input_new_rng is None else input_new_rng
    replaced_rng = rng.replace(src=(old_sz,))
    sub_axis = (new_rng * old_sz + replaced_rng) if top else (replaced_rng * amount + new_rng)
    self.ast = self.ast.substitute({rng:sub_axis}, extra_pm=pm_flatten_range, name=f"shift {rng.axis_id} {amount} {new_type.name.lower()}")
    return replaced_rng, new_rng

  def axes_of(self, *axis_type:AxisType, reduce:bool|None=None) -> list[int]:
    red = self.reduce_ranges
    return [i for i,r in enumerate(self.rngs) if (not axis_type or r.axis_type in axis_type) and (reduce is None or (r in red) == reduce)]

  @property
  def reduce_ranges(self) -> set[UOp]: return {r for u in self.reduceops for r in u.src[1:]}

  def upcast_size(self): return prod(self.full_shape[a] for a in self.axes_of(AxisType.UPCAST))

  def upcastable_dims(self, reduce:bool|None=False) -> list[int]:
    return [i for i in self.axes_of(*split_targets[AxisType.UPCAST], reduce=reduce) if isinstance(self.full_shape[i], int)]

  def apply_opt(self, opt:Opt, append_opt:bool=True):
    if opt.op is OptOps.TC: rng = UOp(Ops.NOOP)
    else:
      check(type(opt.axis) is int and 0 <= opt.axis < self.shape_len, f"invalid axis on {opt.axis=} {opt.op=} {self.shape_len=}")
      rng = self.rngs[cast(int, opt.axis)]

    ret = None
    if opt.op is OptOps.SPLIT:
      check(isinstance(opt.arg, tuple) and len(opt.arg) in (2, 3), f"split arg is (amt, target) or (amt, target, top), not {opt.arg}")
      amt, new_type, top = (*cast(tuple, opt.arg), False)[0:3]
      check(type(amt) is int and (amt == 0 or amt > 1) and isinstance(new_type, AxisType) and new_type in split_targets and isinstance(top, bool),
            f"invalid split arg {opt.arg}")
      if new_type is AxisType.LOCAL: check(self.ren.has_local, "locals needed for opt")

      if amt == 0: amt = int(rng.vmax+1)
      is_reduce = rng in self.reduce_ranges
      if new_type is AxisType.UPCAST:
        if is_reduce: check(amt <= 32, "don't unroll more than 32")
        else: check(self.ren.target.device == "DSP" or amt <= 16, "don't upcast more than 16")
      # prevents METAL compiler hangs
      if self.reduceop is not None and ((new_type is AxisType.LOCAL and is_reduce) or self.group_for_reduces):
        upcast_local_sz = prod([self.full_shape[a] for a in self.axes_of(AxisType.UPCAST, reduce=False)+self.axes_of(AxisType.WARP, AxisType.LOCAL)])
        smem_sz = amt*upcast_local_sz*self.reduceop.dtype.itemsize
        check(smem_sz <= self.ren.shared_max, f"exceeds maximum shared memory size: needs {smem_sz}, max {self.ren.shared_max}")
      if is_reduce and new_type is AxisType.LOCAL:
        reduces = [u for u in self.reduceops if rng in u.src[1:]]
        # We currently don't support a group within another reduce, TODO: fix if-contexts
        check(not any(u in self.reduce_ranges and u.axis_type is AxisType.WEAK for u in reduces[0].ranges),
              "cannot have a group inside another reduce")
      ret = self.shift_to(rng, amt, new_type, top=top)
    elif opt.op is OptOps.TC:
      check(len(self.applied_opts) == 0, "tensor core opts must be first") # TODO: remove the need for this by having warps
      check(opt.axis is not None and opt.axis >= 0, "tensor core opts must have an axis")
      check(opt.arg is not None and isinstance(opt.arg, tuple) and len(opt.arg) == 3, "tensor core opts must have valid arg")
      check(-1 <= (tc_select:=cast(tuple, opt.arg)[0]) < len(self.ren.tensor_cores), "tensor core opts must have valid tc_select")
      check(0 <= (tc_opt:=cast(tuple, opt.arg)[1]) <= 2, "tensor core opts must have valid tc_opt")
      check(0 < (use_tensor_cores:=cast(tuple, opt.arg)[2]) <= 2, "use_tensor_cores value is not valid")
      try: ret = self._apply_tc_opt(use_tensor_cores, cast(int, opt.axis), tc_select, tc_opt)
      except ValueError as e: raise KernelOptError(str(e))
      check(ret is not None, "no tensor core available")
    elif opt.op is OptOps.PADTO:
      check(type(opt.arg) is int and opt.arg > 1, f"padto arg is a multiple > 1, not {opt.arg}")
      check(rng.src[0].op is Ops.CONST, "only pad const axes")
      # TODO: upcasted is only wrong for a range pinned in WMMA tc_upcast_axes
      check(rng.axis_type not in {AxisType.UPCAST, AxisType.WARP}, "cannot pad upcasted or warp")
      new_sz = round_up(int(rng.vmax+1), cast(int, opt.arg))
      check(rng.vmax+1 > new_sz//4, "pad adds more than quadruple the work")
      replaced_rng = rng.replace(src=(rng.src[0].const_like(new_sz),))
      replaces = {rng:replaced_rng}
      valid = replaced_rng < rng.vmax+1
      for b in self.bufs:
        if rng in (i:=b.src[1]).ranges: replaces[b] = b.replace(src=(b.src[0], i.get_idx().valid(valid&i.get_valid())))
      for r in self.reduceops:
        if rng in r.src[1:]:
          replaces[r] = r.replace(src=(valid.where(r.src[0], UOp.const(identity_element(r.arg[0], r.dtype), r.dtype)),)+r.src[1:])
      self.ast = self.ast.substitute(replaces, f"padto {rng.axis_id} {opt.arg}")
      ret = replaced_rng
    elif opt.op is OptOps.SWAP:
      check(type(opt.arg) is int and 0 <= opt.arg < self.shape_len, f"invalid swap axis on {opt.arg=} {self.shape_len=}")
      altrng:UOp = self.rngs[cast(int, opt.arg)]
      check(rng.axis_type == AxisType.GLOBAL and altrng.axis_type == AxisType.GLOBAL, "swap only for globals")
      self.ast = self.ast.substitute({rng:rng.replace(arg=(rng.axis_type, *altrng.axis_id)),
                                      altrng:altrng.replace(arg=(altrng.axis_type, *rng.axis_id))},
                                      name=f"swap {rng.axis_id} {altrng.axis_id}", walk=True)
    else:
      raise KernelOptError(f"unsupported opt {opt.op}")

    if append_opt: self.applied_opts.append(opt)
    return ret

  def _apply_tc_opt(self, use_tensor_cores:int, axis:int, tc_select:int, opt_level:int) -> None|list[UOp]:
    if not (reduceops := self.reduceops): raise KernelOptError("no reduce ops for TensorCore")
    reduceop = reduceops[0]
    if reduceop.arg[0] is Ops.ADD:
      mul = reduceop.src[0] if reduceop.src[0].op is not Ops.CAST else reduceop.src[0].src[0]
      if mul.op is not Ops.MUL: return None
      in0, in1 = mul.src
      for tc in self.ren.tensor_cores if tc_select == -1 else [self.ren.tensor_cores[tc_select]]:
        if self.ren.target.device in ("CUDA", "NV") and tc.dtype_in == dtypes.float and not ALLOW_TF32: continue
        if tc.dtype_in == in0.dtype and tc.dtype_in == in1.dtype and tc.dtype_out == reduceop.dtype:
          # tensor cores have three ranges. X, Y, and REDUCE
          in0_ranges = sorted([u for u in in0.ranges if u not in in1.ranges], key=lambda x: x.arg, reverse=True)
          in1_ranges = sorted([u for u in in1.ranges if u not in in0.ranges], key=lambda x: x.arg, reverse=True)
          red_ranges = sorted(reduceop.src[1:], key=lambda x: x.arg, reverse=True)
          if DEBUG >= 3:
            print(f"TC({axis}): {[(x.axis_id,x.vmax+1) for x in in0_ranges]}",
                              f"{[(x.axis_id,x.vmax+1) for x in in1_ranges]} {[(x.axis_id,x.vmax+1) for x in red_ranges]}")
          if not len(in0_ranges) or not len(in1_ranges) or not len(red_ranges): continue

          # pick ranges
          # NOTE: in1 and in0 are switched because tc.dims is (N, M, K)
          axis_choices = list(itertools.product(in1_ranges, in0_ranges, red_ranges))
          if not (axis < len(axis_choices)): continue
          axes = list(axis_choices[axis])
          check(not any(r in self.reduce_ranges for r in axes[:2]), "tensor core N and M can't be contracted")

          # do optimizations and save the ranges
          ast, warp, ne = self.ast, UOp.range(tc.threads, -1, AxisType.WARP), {}
          try:
            for i,a in enumerate(axes):
              if (a.vmax+1) % tc.dims[i] != 0:
                if opt_level < 2: raise KernelOptError("tc padding requires opt_level >= 2")
                axes[i] = self.apply_opt(Opt(OptOps.PADTO, self.rngs.index(a), tc.dims[i]), append_opt=False) # PADTO might fail
            # we create the warp as a whole thing, in case some of these ranges are moved/removed later
            for c in tc.axis_coords():
              d = "nmk".index(c[0])
              if c in tc.frag_c[0]: axes[d], ne[c] = self.shift_to(axes[d], 2, AxisType.LOCAL, input_new_rng=warp//2**tc.frag_c[0].index(c)%2)
              else: axes[d], ne[c] = self.shift_to(axes[d], 2, AxisType.UPCAST)
          except KernelOptError:
            self.ast = ast
            continue

          if use_tensor_cores != 2:
            reduceop = get_single_element([x for x in self.reduceops if axes[2] in x.src[1:]])
            gate, mul = (r0.src[0], r0.src[1]) if (r0:=reduceop.src[0]).op is Ops.WHERE else (None, r0)
            if mul.op is Ops.CAST: mul = mul.src[0]
            ins = mul.src if gate is None else tuple(gate.where(x, UOp.const(0, x.dtype)) for x in mul.src)
            srcs = [x.substitute({ne[a]: ne[b] for a,b in rl.items()}, walk=True) for x,rl in zip(ins, tc.relabel())]

            # get upcast axes for the tensor cores
            base_upcast_axes = [ne[c].arg for c in tc.base_upcast_axes()]
            upcast_cnt = [len(f[1]) for f in (tc.frag_a, tc.frag_b, tc.frag_c)]
            # each operand upcasts its first upcast_cnt axes, the axes only A or B upcast are size 1 so the operands broadcast
            tc_upcast_axes = tuple([tuple([(a, 2 if j < cnt else 1) for j,a in enumerate(base_upcast_axes[:max(cnt, *upcast_cnt[:2])])])
                                    for cnt in upcast_cnt])

            # construct the op
            # TODO: remove tc_upcast_axes from the arg
            tc_uop = UOp.wmma(srcs[0], srcs[1], UOp.const((0.0,)*2**upcast_cnt[2], tc.dtype_out),
                              tc.dims, tc.threads, tc_upcast_axes=tc_upcast_axes)

            # preserve extra reduces
            reduce_ranges = [x for x in reduceop.src[1:] if x not in [ne[c] for c in ne if c[0] == "k"]]
            if len(reduce_ranges): tc_uop = UOp(Ops.REDUCE, src=(tc_uop,)+tuple(reduce_ranges), arg=(Ops.ADD, 0))
            self.ast = self.ast.substitute({reduceop: tc_uop})
          return axes
    return None

  # helpers for hand_coded_optimizations
  @property
  def reduceops(self) -> list[UOp]: return [x for x in self.ast.backward_slice if x.op is Ops.REDUCE]
  @property
  def reduceop(self) -> UOp|None: return red[0] if (red:=self.reduceops) else None
  @property
  def bufs(self) -> list[UOp]: return [x for x in self.ast.backward_slice if x.op is Ops.INDEX][::-1]
  @property
  def group_for_reduces(self) -> int: return len(self.axes_of(AxisType.WARP, AxisType.LOCAL, reduce=True))

def args_from_ast(ast:UOp, dname:str) -> tuple[list[Buffer], dict[str, int]]:
  glbls = sorted([x for x in ast.backward_slice if x.op is Ops.PARAM and x.arg.slot >= 0], key=lambda x: x.arg.slot)
  return [Buffer(dname, x.max_numel() * x.dtype.itemsize) for x in glbls], {k.expr:int(k.vmax+k.vmin)//2 for k in ast.variables()}

def apply_opts(ast:UOp, ren:Renderer, beam:int=0) -> UOp:
  if ast.tag is not None: return ast
  k = Scheduler(ast, ren)
  k.convert_loop_to_global()
  if ast.arg is not None and ast.arg.opts_to_apply is not None:
    for opt in ast.arg.opts_to_apply: k.apply_opt(opt)
  elif beam >= 1:
    from tinygrad.codegen.opt.search import beam_search
    rawbufs, var_vals = args_from_ast(ast, ren.target.device)
    # beam search may open devices
    with Context(ALLOW_DEVICE_USAGE=1):
      k = beam_search(k, rawbufs, var_vals, beam, bool(getenv("BEAM_ESTIMATE", 1)))
  elif not NOOPT and (ast.arg is None or ast.arg.applied_opts == ()):
    from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
    # NOTE: hand_coded_optimizations doesn't support multiblock opts yet
    if not any(u.op is Ops.STAGE for u in ast.backward_slice):
      k = hand_coded_optimizations(k)
  return k.get_optimized_ast(name_override=ast.arg.name if ast.arg is not None and ast.arg.name != "test" else None)
