import itertools
from enum import Enum, auto
from typing import List, Tuple
from tinygrad.helpers import prod, dedup, all_same, colored
from tinygrad.ops import LazyOp, MovementOps, get_lazyop_info, get_buffers, ReduceOps, get_lazyops, map_buffers
from tinygrad.shape import ShapeTracker, View, strides_for_shape

def get_first_reduce(shapes):
  for i in range(len(shapes[0])):
    if not all_same([x[i] for x in shapes]): return i
  return len(shapes[0])  # off the end

# this will be removed soon anyway
class Types(Enum): FLOAT = auto(); FLOAT4 = auto() # noqa: E702
class Token:
  def __init__(self, tok:str, typ:Types, ptr:bool=False):
    assert isinstance(tok, str)
    self.tok, self.typ, self.ptr = tok, typ, ptr
    self.axis : List[Tuple[int, int, bool]] = []
  def array(self, length, stride, reduce): self.axis.append((length, stride, reduce))
  def size(self): return prod([x[0] for x in self.axis])
  def offsets(self): return [sum(t) for t in itertools.product(*[[y*x[1] for y in range(x[0])] for x in self.axis[::-1]])] if len(self.axis) else [0]
  def can_float4(self): return any(a[0:2] == (4,1) for a in self.axis)
  # TODO: this is sort of a hack, it gets the accumulator indices
  def acc_offsets(self):
    if len(self.axis) == 0: return [0]
    acc_strides = [x*(1-self.axis[::-1][i][2]) for i,x in enumerate(strides_for_shape(tuple(1 if r else s for s,_,r in self.axis[::-1])))]
    return [sum(t) for t in itertools.product(*[[y*acc_strides[i] for y in range(x[0])] for i,x in enumerate(self.axis[::-1])])]
  def decltype(self): return ('float' if self.typ == Types.FLOAT else 'float4') + ('*' if self.ptr else str())
  def __repr__(self): return f"<{self.typ}{'*' if self.ptr else str()} {self.tok}{f'[{self.axis}]' if len(self.axis) else str()}>"

# ast kernel can contain one ReduceOp with arbitrary Binary/Unary ops
class ASTKernel:
  def __init__(self, ast:LazyOp, output_buffer=None):
    self.input_ast = ast

    # if the AST ends with a RESHAPE, we remove it and create the buffer accordingly
    if ast.op == MovementOps.RESHAPE:
      output_shape = ast.arg
      ast = ast.src[0]
    else:
      output_shape = None

    self.info = get_lazyop_info(ast)
    self.bufs = dedup(get_buffers(ast))
    for b in self.bufs: b.st.simplify()
    self.ast = ast

    # check if the output buffer is allowed to be used
    # if it's aliased, don't use it
    if output_buffer is not None:
      for a in self.bufs:
        if a._buf == output_buffer._buf and not a.st.contiguous:
          output_buffer = None
          break

    # create the buffer we are returning (as the same type as the input buffers) and add it as the first buffer
    self.ret = output_buffer if output_buffer else type(self.bufs[0])(output_shape if output_shape else self.info.shape, force_create=True)
    self.bufs = ([type(self.ret)(self.info.shape, hostbuf=self.ret)] if output_shape else [self.ret]) + self.bufs

    # key for lookup in cache (can change, str might not be right)
    # bufs are needed because kernels like f(x) = x + x and f(x, y) = x + y have the same str(ast), but are different kernels.
    # mapping the buffers to integers is required because a-b != b-a (and how would you tell a and b apart?)
    self.key = f"ASTKernelKey ast={str(map_buffers({x:i for i,x in enumerate(self.bufs)}, ast))} bufs={self.bufs}"

  def process(self) -> None:
    if hasattr(self, "sts"): return   # already processed

    reduceops = [x for x in get_lazyops(self.ast) if x.op in ReduceOps]
    assert len(dedup(reduceops)) <= 1, "max one reduce op in an ast"
    self.reduceop = reduceops[0] if reduceops else None
    self.earlybufs = dedup(get_buffers(self.reduceop)) if self.reduceop else []

    self.buftokens = [Token(f"data{i}", Types.FLOAT, ptr=True) for i in range(len(self.bufs))]
    self.group_for_reduce : List[int] = []

    # check valid AST kernel
    assert all_same([x.shape for x in self.earlybufs]), "all earlybufs must have the same shape"
    assert all_same([x.shape for x in self.bufs if x not in self.earlybufs]), "all latebufs must have the same shape"
    assert all_same([len(x.shape) for x in self.bufs]), "all bufs must have the same shape size"

    # process
    self.sts : List[ShapeTracker] = [x.st.copy() for x in self.bufs]   # create new shapetrackers inside this kernel
    self.simplify_ones()
    self.simplify_merge_adjacent()

    # get full shape buf index (earlybufs if there are any, otherwise output)
    self.full_buf_index : int = self.bufs.index(self.earlybufs[0]) if len(self.earlybufs) > 0 else 0

  def print(self):
    buf_count, op_count, cache = -1, -1, {}
    def print_ast(x, name=None):
      nonlocal buf_count, op_count
      if x not in cache:
        if not isinstance(x, LazyOp):
          if name is None:
            buf_count += 1
            name = f"buf{buf_count}"
          print(f"buf{buf_count} = {x}")
          cache[x] = name
        else:
          srcs = [print_ast(y) for y in x.src]
          if name is None:
            op_count += 1
            name = f"op{op_count}"
          print(f"{name} = LazyOp({str(x.op)}, ({','.join(srcs)},), {x.arg})")
          cache[x] = name
      return cache[x]
    print_ast(self.input_ast, "ast")

  def printbufs(self, prefix="", print_shapetrackers=False):
    print(f"first_reduce: {self.first_reduce} shape_len: {self.shape_len} group_for_reduce: {self.group_for_reduce}")
    if print_shapetrackers:
      for st in self.sts: print(st)
    for i in range(len(self.sts)):
      print(prefix, self.buftokens[i], f"early:{'T' if i < len(self.bufs) and self.bufs[i] in self.earlybufs else 'F'}", self.sts[i].shape, self.sts[i].views[-1].strides, len(self.sts[i].views), type(self.bufs[i]._buf) if self.bufs[i] is not None else "FAKE")

  @property
  def shape_len(self) -> int: return len(self.sts[0].shape)

  @property
  def full_shape(self) -> Tuple[int, ...]: return self.sts[self.full_buf_index].shape

  @property
  def upcast_in_mid_reduce_axes(self): return [j for j in range(self.first_reduce, self.first_reduce+len(self.group_for_reduce)) if self.full_shape[j] == self.sts[0].shape[j]]

  def colorshape(self, pad=50) -> str:
    axis = [(f"{rs:4d}", (("green" if i in self.upcast_in_mid_reduce_axes else "cyan") if i < self.first_reduce + len(self.group_for_reduce) else "red") if i >= self.first_reduce else "blue") for i, rs in enumerate(self.full_shape)]
    axis += [(f"{s:4d}", 'magenta' if reduce else 'yellow') for s, _, reduce in self.buftokens[self.full_buf_index].axis[::-1]]
    return ' '.join([colored(*x) for x in axis])+(" "*(pad-len(' '.join([x[0] for x in axis]))))

  def simplify_ones(self):
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    all_ones = [all(st.shape[i]==1 for st in self.sts) for i in range(self.shape_len)]
    # keep at least 1 one
    if all(all_ones): all_ones[-1] = False
    self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)
    # find first mismatch, don't reduce this
    self.first_reduce = get_first_reduce([x.shape for x in self.sts])

  def simplify_merge_adjacent(self):
    shapes, strides = [x.shape for x in self.sts], [x.views[-1].strides for x in self.sts]

    # merge dimensions if we can, multi get_shape_strides
    # TODO: does this always preserve the reduce dimension, NO
    # TODO: move this into shapetracker, with tests!
    rets = [[(shapes[j][0], strides[j][0])] for j in range(len(shapes))]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for j in range(len(shapes)):
        # TODO: added the always mergeability of 1s, is this right? if so, add to shapetracker in the 1 case
        can_merge.append((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*strides[j][i]) or (strides[j][i] == 0 and rets[j][-1][1] == 0))
      # more can merge than this
      mergeable = all(can_merge) and i != self.first_reduce
      for j in range(len(shapes)):
        if mergeable: rets[j][-1] = (rets[j][-1][0] * shapes[j][i], strides[j][i])
        else: rets[j].append((shapes[j][i], strides[j][i]))

    for i,x in enumerate(rets): self.sts[i].reshape(tuple(y[0] for y in x))
    self.first_reduce = get_first_reduce([x.shape for x in self.sts])

  # this should be aware of the three parts to the shape
  #  * the input/output dimensions
  #  * the reduce dimensions
  #  * the size outputted by each kernel
  def reshape_and_permute(self, new_shape_fxn, axis):
    for st in self.sts:
      if new_shape_fxn is not None: st.reshape(tuple(new_shape_fxn(st.shape)))
      if axis is not None: st.permute(tuple(axis))

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

  # drops the final dimension
  def upcast(self):
    upcasted = [x.shape[-1] for x in self.sts if x.shape[-1] != 1]
    assert len(upcasted) >= 1 and all_same(upcasted), f"can't upcast mismatch {upcasted}"
    for st,buftoken in zip(self.sts, self.buftokens):
      # add last axis to the buftoken (if it's not a 1)
      if st.shape[-1] == upcasted[0]: buftoken.array(st.shape[-1], st.views[-1].strides[-1], len(upcasted) != len(self.sts))
      # remove the last axis (unless it's the only dimension, then make it a 1)
      st.views[-1] = View(st.shape[0:-1], st.views[-1].strides[0:-1], st.views[-1].offset) if len(st.shape) > 1 else View((1,), (0,), st.views[-1].offset)
