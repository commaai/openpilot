from __future__ import annotations
import heapq
from collections import defaultdict
from dataclasses import dataclass, replace
from tinygrad.ops import UOp, Ops, PatternMatcher, UPat, GroupOp
from tinygrad.helpers import dedup, partition, all_same, flatten
from tinygrad.spec import type_verify

# NOTE: any toposort should be valid here, unlike last time this isn't required, it's just for speed
def block_reorder(lst:list[UOp]) -> list[UOp]:
  in_this_block = set(lst)
  local_children: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree:dict[UOp, int] = {}
  priorities:dict[UOp, int] = {}

  # get local children and assign priorities
  # NOTE: this requires the lst be locally toposorted
  for u in reversed(lst):
    in_degree[u] = 0
    for s in u.src:
      if s in in_this_block:
        local_children[s].append(u)
        in_degree[u] += 1
    # put loads in the beginning of the block and prevent priority inversion. hack for BARRIER grouping too
    priority = [0] + [priorities[x] for x in local_children[u]]
    if u.op is Ops.LOAD: priority.append(-1000)
    if u.op is Ops.BARRIER: priority.append(-1500)
    priorities[u] = min(priority)

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: (priorities[x],)+x.tuplize))}

  # then force then to be toposorted in as close to the ideal order as possible
  heapq.heapify(heap:=[(nkey[u],u) for u in lst if in_degree[u] == 0])
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in local_children[u]:
      in_degree[v] -= 1
      if in_degree[v] == 0: heapq.heappush(heap, (nkey[v],v))

  assert len(newlst) == len(lst), f"len mismatch {len(newlst)} != {len(lst)}"
  return newlst

# ***** basic block *****

def disp(y:UOp) -> str:
  if y.op is Ops.IF: return f'IF{id(y)}'
  if y.op is Ops.RANGE: return str(y.arg)
  return "<NONE>"

@dataclass(frozen=True, eq=False)
class BasicBlock2:
  lst: tuple[UOp, ...]
  ctx: tuple[UOp, ...] = ()
  end: UOp|None = None
  cnt: int = 0
  child_ctx: tuple[UOp, ...]|None = None
  def __lt__(self, _:BasicBlock2): raise RuntimeError("no comparing basic blocks")
  def __repr__(self):
    return f"{(str(disp(self.end))+' ') if self.end is not None else ''}"+f'f{self.cnt} '+\
           f"{[disp(y) for y in self.ctx]} {[disp(y) for y in self.child_ctx] if self.child_ctx is not None else '-'} "+\
           f"{len(self.lst)}" + "\n" + '\n'.join([str(x.op) for x in self.lst])
  def last_ctx(self): return self.child_ctx if self.child_ctx is not None else self.ctx

def _sort_ctx(inp): return tuple(sorted(dedup(inp), key=lambda x: x.tuplize))

# ***** block context *****

@dataclass
class BlockContext:
  child_count: dict[UOp, int]
  block_ctxs: dict[UOp, tuple[UOp, ...]]
  child_ctxs: dict[UOp, tuple[UOp, ...]]
  def last_ctx(self, u): return ret if (ret:=self.child_ctxs.get(u)) is not None else self.block_ctxs[u]
  @staticmethod
  def from_sink(sink:UOp) -> BlockContext:
    # get children and all block contexts
    ctx = BlockContext({}, {}, {})
    for u in sink.toposort():
      this_block_ctx: list[UOp] = []
      ctx.child_count[u] = 0

      # get children and accumulate the last_ctx
      for s in u.src:
        # NOTE: if a parent appears multiple times in the src, it counts multiple times as a child
        ctx.child_count[s] += 1
        this_block_ctx += ctx.last_ctx(s)

      # save the block ctx
      ctx.block_ctxs[u] = _sort_ctx(this_block_ctx)

      # RANGE/IF add to the next ctx
      # STORE/ASSIGN subtract from the next ctx
      if u.op in {Ops.RANGE, Ops.IF}: ctx.child_ctxs[u] = _sort_ctx(ctx.block_ctxs[u] + (u,))
      elif u.op is Ops.STORE:
        # ugh, deal with non-reduce locals. probably wrong
        if any(x.op is Ops.DEFINE_LOCAL for x in u.src[0].toposort()):
          idx_context, store_context = ctx.last_ctx(u.src[0]), ctx.last_ctx(u.src[1])
          ctx.child_ctxs[u] = tuple([y for y in store_context if y not in idx_context and y.op is Ops.RANGE])
        else: ctx.child_ctxs[u] = ()
      elif u.op is Ops.ASSIGN:
        assert u.src[0].op is Ops.DEFINE_ACC
        ctx.child_ctxs[u] = tuple([y for y in ctx.last_ctx(u.src[1]) if y not in u.src[0].src[1:]])
    return ctx

# ***** make blocks *****

DONT_PLACE_IN_BLOCK = {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.CONST}

def add_blockends(base_block:UOp, new_ctx:tuple[UOp, ...], current_ctx:tuple[UOp, ...], cnt:int=1) -> UOp:
  ends_to_add = [z for z in new_ctx if z not in current_ctx]
  while len(ends_to_add):
    r:UOp = ends_to_add.pop(-1)
    new_ctx = tuple([z for z in new_ctx if z is not r])
    end_uop = UOp(Ops.ENDIF if r.op is Ops.IF else Ops.ENDRANGE, src=(r,))
    base_block = UOp(Ops.BLOCKEND, src=(base_block,)*cnt, arg=BasicBlock2((end_uop,), tuple(new_ctx), end=r, cnt=cnt))
  return base_block

def make_block_bottom_up(ctx:BlockContext, x:UOp):
  if x.op is Ops.BLOCKSTART:
    current_ctx, child_ctx = x.arg
    lst = list(x.src)
    child_count = 1
  else:
    current_ctx, child_count, child_ctx = ctx.block_ctxs[x], ctx.child_count[x], ctx.child_ctxs.get(x, None)
    lst = [x]

  # count of times we've seen this block, or a seed for a new block if we can't merge it
  unmergable: defaultdict[UOp, int] = defaultdict(int)
  blockseeds = defaultdict(list)

  # add the srcs of this to the frontier
  # NOTE: things may be in here multiple times, that's okay
  frontier_nodes = list(flatten(y.src[::-1] for y in lst))
  while len(frontier_nodes):
    u = frontier_nodes.pop(0)
    if u.op not in DONT_PLACE_IN_BLOCK and ctx.child_count[u] == unmergable[u]+1:
      # count is correct
      if (newctx:=ctx.block_ctxs[u]) == current_ctx:
        # block has same context, merge it, and put the srcs on the frontier
        lst.append(u)
        frontier_nodes.extend(u.src[::-1])
      else:
        # block has different context, add it to blockseeds
        blockseeds[(newctx, ctx.child_ctxs.get(u, None))].append(u)
      del unmergable[u]
    else:
      # count is incorrect (or it's DONT_PLACE_IN_BLOCK), add it to unmergable
      unmergable[u] += 1

  # add unmergables to sources
  srcs = []
  for u,cnt in unmergable.items(): srcs += [add_blockends(u, ctx.block_ctxs[u], current_ctx, cnt=cnt)]*cnt

  # add blockseeds, with blockends as needed
  for (new_ctx, new_child_ctx), v in blockseeds.items():
    base_block = UOp(Ops.BLOCKSTART, src=tuple(v), arg=(new_ctx, new_child_ctx))
    srcs.append(add_blockends(base_block, new_ctx, current_ctx))

  lst = block_reorder(lst[::-1])
  bb = BasicBlock2(tuple(lst), ctx=current_ctx, cnt=child_count, child_ctx=child_ctx)
  return UOp(Ops.BLOCK, src=tuple(srcs), arg=bb)

block_create = PatternMatcher([
  (UPat(GroupOp.All-DONT_PLACE_IN_BLOCK.union({Ops.BLOCK, Ops.BLOCKEND}), name="x"), make_block_bottom_up),
])

# ***** blockend merging ****

def merge_blockends(sink:UOp) -> UOp|None:
  # only run on the final BLOCK with the SINK in it
  if sink.arg.lst[-1].op is not Ops.SINK: return None
  # combine matching BLOCKENDS, the keys of this dictionary are the RANGE UOps, values are the BLOCKENDs
  blockends_to_arg: dict[UOp, list[UOp]] = {}
  for be in sink.toposort():
    if be.op is Ops.BLOCKEND: blockends_to_arg.setdefault(be.arg.end, []).append(be)
  new_forks = {}
  for k,v in blockends_to_arg.items():
    # NOTE: if any BLOCKEND is the parent of any other with the same arg, this algo fails
    if len(v) > 1:
      bb = BasicBlock2(v[0].arg.lst, _sort_ctx(flatten([y.arg.ctx for y in v])), k, cnt=sum(y.arg.cnt for y in v))
      out = UOp(Ops.BLOCKEND, src=tuple(flatten([x.src for x in v])), arg=bb)
      # NOTE: bb.ctx != u.arg.ctx can cause problems here
      for u in v: new_forks[u] = out
  if len(new_forks) == 0: return None
  return sink.substitute(new_forks)

pm_blockend_merge = PatternMatcher([(UPat(Ops.BLOCK, name="sink"), merge_blockends)])

# ***** block merging ****

def merge_block(x:UOp):
  unmergable_blocks, mergable_blocks = [], []
  mergable_dict: defaultdict[UOp, int] = defaultdict(int)
  for y in x.src:
    if y.op is Ops.BLOCK and x.op is Ops.BLOCK and x.arg.ctx == y.arg.ctx: mergable_dict[y] += 1
    elif y.op is Ops.BLOCK and x.op is Ops.BLOCKEND and x.arg.end in y.arg.ctx: mergable_dict[y] += 1
    else: unmergable_blocks.append(y)
  for k,v in mergable_dict.items():
    if v == k.arg.cnt: mergable_blocks.append(k)
    else: unmergable_blocks.extend([k]*v)
  if len(mergable_blocks) == 0: return None
  del mergable_dict

  # create the block
  arg = replace(x.arg, lst=tuple(flatten([y.arg.lst for y in mergable_blocks]))+x.arg.lst)
  return UOp(x.op, src=tuple(flatten([y.src for y in mergable_blocks])+unmergable_blocks), arg=arg)

def remove_blockend(x:UOp):
  # if there's any remaining blocks that need to go in this BLOCKEND, we don't remove it
  if any(x.arg.end in y.arg.ctx for y in x.src if y.op in {Ops.BLOCK, Ops.BLOCKEND}): return None

  parent_blocks = [y for y in x.src if y.op is Ops.BLOCK and y.arg.child_ctx is not None and x.arg.end in y.arg.child_ctx]
  assert all_same(parent_blocks), f"should never have two parent blocks (has {len(parent_blocks)})"
  if len(parent_blocks) > 0:
    parent_block = parent_blocks[0]
    assert len(parent_blocks) == parent_block.arg.cnt
    # range needs DEFINE_ACC to be before the range (never in DEFINE_ACC for if)
    early_ops, late_ops = partition(x.arg.lst, lambda y: y.op is Ops.DEFINE_ACC and x.arg.end in y.src)
    # NOTE: we have to add a barrier at the start if barrier is used in the range
    if x.op is Ops.BLOCKEND and any(y.op is Ops.BARRIER for y in late_ops) and late_ops[-1].op is Ops.ENDRANGE:
      late_ops = [UOp(Ops.BARRIER)] + late_ops
    arg = BasicBlock2(tuple(early_ops)+parent_block.arg.lst+tuple(late_ops), tuple([y for y in x.arg.ctx if y is not x.arg.end]), cnt=x.arg.cnt)
    return UOp(Ops.BLOCK, src=tuple(y for y in x.src if y is not parent_block)+parent_block.src, arg=arg)

block_merge = PatternMatcher([
  (UPat((Ops.BLOCK, Ops.BLOCKEND), name="x"), merge_block),
  (UPat(Ops.BLOCKEND, name="x"), remove_blockend),
])

# ****** finalize ******

def finalize(sink:UOp) -> UOp:
  if sink.op is not Ops.BLOCK or not all(x.op in DONT_PLACE_IN_BLOCK for x in sink.src):
    raise RuntimeError("linearize failure")

  # place the early things
  lst = sorted(dedup(sink.src), key=lambda x: x.tuplize) + list(sink.arg.lst)

  if __debug__: type_verify(lst)

  return UOp(Ops.BLOCKFINAL, arg=BasicBlock2(tuple(lst)))

pm_finalize = PatternMatcher([(UPat(Ops.BLOCK, name="sink"), finalize)])
