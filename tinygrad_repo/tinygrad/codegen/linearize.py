from __future__ import annotations
import collections, heapq
from dataclasses import dataclass
from tinygrad.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite, GroupOp
from tinygrad.spec import type_verify
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import dedup, flatten, partition

DONT_PLACE_IN_BLOCK = {Ops.NAME, Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.CONST, *GroupOp.Block}

def disp(y:UOp) -> str:
  if y.op is Ops.BLOCKSTART: return "w"+disp(y.src[0])
  if y.op is Ops.IF: return f'IF{id(y)}'
  if y.op is Ops.RANGE: return str(y.arg)
  return "<NONE>"

@dataclass(frozen=True)
class BasicBlock:
  ctx: tuple[UOp, ...]
  lst: tuple[UOp, ...]
  end: UOp|None = None
  def __lt__(self, o:BasicBlock): return tuple(x.tuplize for x in self.ctx+self.lst) < tuple(x.tuplize for x in o.ctx+o.lst)
  def __repr__(self):
    return f"{(str(disp(self.end))+' ') if self.end is not None else ''}"+\
           f"{[disp(y) for y in self.ctx]} {len(self.lst)}" + "\n" + '\n'.join([str(x.op) for x in self.lst])

def append_to_block(ctx:tuple[dict[UOp, tuple[UOp, ...]], dict[UOp, list[UOp]]], x:UOp):
  block_ctxs, children = ctx
  in_this_block = set(x.arg.lst)

  # collections to build
  new_srcs: list[UOp] = []
  to_append: list[UOp] = []
  old_blocks: dict[tuple[UOp, ...], UOp] = {}
  new_blocks: dict[tuple[UOp, ...], list[UOp]] = {}

  for u in x.src:
    if u.op is Ops.BLOCK:
      # merge sibling blocks. NOTE: blocks must only have one output source
      assert u.arg.ctx not in old_blocks, "sibling should never have been created"
      old_blocks[u.arg.ctx] = u
    elif u.op not in DONT_PLACE_IN_BLOCK and set(children[u]).issubset(in_this_block):
      # if it can go in blocks and all its children are in the block, we add it to the block
      if (block_ctx:=block_ctxs[u]) == x.arg.ctx:
        # if it's the same context, we place the UOp in this block and append the parents to its srcs
        new_srcs.extend(u.src)
        to_append.append(u)
      else:
        # if it's a different context, we create a new block with this UOp
        new_blocks.setdefault(block_ctx, []).append(u)
    else:
      # otherwise, we keep it in the srcs
      new_srcs.append(u)
  if len(to_append) == 0 and len(new_blocks) == 0: return None

  for rng,lst in new_blocks.items():
    srcs = flatten(y.src for y in lst)
    if (old_block:=old_blocks.pop(rng, None)) is not None:
      # NOTE: order shouldn't matter here
      srcs.extend(old_block.src)
      lst.extend(old_block.arg.lst)
    new_block = UOp(Ops.BLOCK, dtypes.void, tuple(dedup(srcs)), BasicBlock(rng, tuple(lst)))
    lrng = list(rng)
    for r in rng[::-1]:
      if r not in x.arg.ctx and r.op is not Ops.BLOCKSTART:
        lrng.remove(r)
        new_block = UOp(Ops.BLOCKEND, src=(new_block,),
                        arg=BasicBlock(tuple(lrng), (UOp(Ops.ENDIF if r.op is Ops.IF else Ops.ENDRANGE, src=(r,)),), r))
    new_srcs.append(new_block)
  return UOp(Ops.BLOCK, dtypes.void, tuple(dedup(list(old_blocks.values())+new_srcs)), BasicBlock(x.arg.ctx, tuple(to_append)+x.arg.lst))

make_basic_blocks = PatternMatcher([
  (UPat(Ops.SINK, name="x"),
    lambda x: UOp(Ops.BLOCK, src=x.src+((UOp(Ops.NAME, arg=x.arg.name),) if x.arg is not None else ()), arg=BasicBlock((), (x,)))),
  (UPat(Ops.BLOCK, name="x"), append_to_block),
])

def block_merge(ctx, x:UOp):
  # ctx is children here
  if x.op is Ops.BLOCKEND:
    # if it's a BLOCKEND, see if we are done with placement. if all the children of the range are in here
    in_this_block = set(x.arg.lst)
    if len([y for y in ctx[x.arg.end] if y not in in_this_block]) == 0:
      # find the parent block that has the BLOCKSTART in the ctx
      parent_blocks = [y for y in x.src if y.op is Ops.BLOCK and UOp(Ops.BLOCKSTART, src=(x.arg.end,)) in y.arg.ctx]
      assert len(parent_blocks) <= 1, "should never have two parent blocks"
      if len(parent_blocks) == 1:
        parent_block = parent_blocks[0]
        # range needs DEFINE_ACC to be before the range (never in DEFINE_ACC for if)
        early_ops, late_ops = partition(x.arg.lst, lambda y: y.op is Ops.DEFINE_ACC and x.arg.end in y.src)
        return UOp(Ops.BLOCK, dtypes.void, tuple(y for y in x.src if y is not parent_block)+parent_block.src,
                  BasicBlock(tuple(y for y in x.arg.ctx if y is not x.arg.end), tuple(early_ops)+parent_block.arg.lst+tuple(late_ops)))

  new_srcs: list[UOp] = []
  to_append: list[UOp] = []
  new_ctx = x.arg.ctx
  placed = set()
  for u in x.src:
    if u.op is Ops.BLOCK and (tuple(u.arg.ctx) == tuple(x.arg.ctx) or (x.arg.end is not None and x.arg.end in u.arg.ctx)):
      # NOTE: this can't appear in srcs twice or it would be a BLOCKFORK
      new_ctx += tuple(y for y in u.arg.ctx if y not in x.arg.ctx)
      new_srcs.extend(u.src)
      to_append.extend(u.arg.lst)
    elif u.op is Ops.BLOCKFORK and x.src.count(u) == u.arg: # block fork appears # of times in srcs
      if u not in placed:
        new_srcs.extend(u.src)
        placed.add(u)
    else:
      # keep it in srcs
      new_srcs.append(u)
  if len(to_append) == 0 and len(placed) == 0: return None
  return UOp(x.op, dtypes.void, tuple(new_srcs), BasicBlock(tuple(sorted(new_ctx, key=lambda x: x.tuplize)), tuple(to_append)+x.arg.lst, x.arg.end))

pm_block_merge = PatternMatcher([(UPat((Ops.BLOCKEND, Ops.BLOCK), name="x"), block_merge),])

def block_finalize(block:UOp):
  if len(block.src) == 0: return None
  _uops = sorted(dedup(block.src), key=lambda x: x.tuplize)
  assert all(len(x.src) == 0 and x.op not in {Ops.BLOCK, Ops.BLOCKSTART, Ops.BLOCKEND, Ops.BLOCKFORK} for x in _uops)
  _uops += block.arg.lst
  # strip the SINK
  assert _uops[-1].op is Ops.SINK, "doesn't end with SINK"
  return UOp(Ops.BLOCK, arg=BasicBlock((), tuple(_uops[:-1])))

pm_block_finalize = PatternMatcher([(UPat(Ops.BLOCK, name="block"), block_finalize)])

# NOTE: any toposort should be valid here, unlike last time this isn't required, it's just for speed
def block_reorder(in_block:UOp):
  in_this_block = set(in_block.arg.lst)
  local_children: collections.defaultdict[UOp, list[UOp]] = collections.defaultdict(list)
  in_degree: collections.defaultdict[UOp, int] = collections.defaultdict(int)
  priorities:dict[UOp, int] = {}

  # get local children and assign priorities
  for u in reversed(in_block.arg.lst):
    for s in u.src:
      if s in in_this_block:
        local_children[s].append(u)
        in_degree[u] += 1
    # put loads in the beginning of the block and prevent priority inversion
    priorities[u] = min([-1000 if u.op is Ops.LOAD else 0] + [priorities[x] for x in local_children[u]])

  # placement queue
  queue:list[tuple[int, tuple, UOp]] = []
  def push(u:UOp): heapq.heappush(queue, (priorities[u], u.tuplize, u))

  # place the first ones that don't have deps
  for u in in_block.arg.lst:
    if u not in in_degree: push(u)

  newlst = []
  while queue:
    _,_,x = heapq.heappop(queue)
    newlst.append(x)
    for u in local_children[x]:
      in_degree[u] -= 1
      if in_degree[u] == 0: push(u)

  assert len(newlst) == len(in_block.arg.lst), f"len mismatch {len(newlst)} != {len(in_block.arg.lst)}"
  return in_block.replace(arg=BasicBlock(in_block.arg.ctx, tuple(newlst)))

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> list[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  # get children and all block contexts
  temp_block_ctxs: dict[UOp, list[UOp]] = {}
  children: dict[UOp, list[UOp]] = {}
  for u in sink.toposort:
    this_block_ctx: list[UOp] = []
    for s in u.src:
      # save children
      children.setdefault(s, []).append(u)
      # compute block ctx
      if s.op in {Ops.RANGE, Ops.IF}: this_block_ctx.append(s)
      # don't flow (fully) through assign and store
      elif s.op is Ops.STORE:
        # ugh, deal with non-reduce locals. probably wrong
        if isinstance(s.src[0].dtype, PtrDType) and s.src[0].dtype.local:
          idx_context, store_context = temp_block_ctxs[s.src[0]], temp_block_ctxs[s]
          this_block_ctx += [x for x in store_context if x not in idx_context and x.op is Ops.RANGE]
      elif s.op is Ops.ASSIGN:
        # flow though assign, but remove the ranges used in the assign
        assert s.src[0].op is Ops.DEFINE_ACC
        this_block_ctx += [x for x in temp_block_ctxs[s.src[1]] if x not in s.src[0].src[1:]]
      else:
        # flow though everything else
        this_block_ctx += temp_block_ctxs[s]
    temp_block_ctxs[u] = sorted(dedup(this_block_ctx), key=lambda x: x.tuplize)

  # make final block_ctxs, add BLOCKSTART to block_ctxs for IF and RANGE
  block_ctxs: dict[UOp, tuple[UOp, ...]] = {}
  for u in sink.toposort:
    block_ctxs[u] = ((UOp(Ops.BLOCKSTART, src=(u,)),) + tuple(temp_block_ctxs[u])) if u.op in {Ops.IF, Ops.RANGE} else tuple(temp_block_ctxs[u])

  # TODO: there's probably a clever way to remove this while loop
  while 1:
    sink = graph_rewrite(sink, make_basic_blocks, ctx=(block_ctxs, children))

    # add BLOCKFORK (slow!)
    block_parent_count = collections.Counter(flatten([x.src for x in sink.toposort if x.op is Ops.BLOCK]))
    non_block_parents = set(flatten([x.src for x in sink.toposort if x.op is not Ops.BLOCK]))
    forks = {u:UOp(Ops.BLOCKFORK, src=(UOp(Ops.BLOCK, src=u.src, arg=BasicBlock(block_ctxs[u], (u,))),), arg=child_count)
      for u,child_count in block_parent_count.items() if u.op not in DONT_PLACE_IN_BLOCK and child_count > 1 and u not in non_block_parents}

    if not len(forks): break
    sink = sink.substitute(forks)

  # combine matching BLOCKENDS
  blockends_to_arg: dict[UOp, list[UOp]] = {}
  for be in sink.toposort:
    if be.op is Ops.BLOCKEND: blockends_to_arg.setdefault(be.arg.end, []).append(be)
  new_forks = {}
  for k,v in blockends_to_arg.items():
    # NOTE: if any BLOCKEND is the parent of any other with the same arg, this algo fails
    if len(v) > 1:
      out = UOp(Ops.BLOCKFORK, src=(UOp(Ops.BLOCKEND, src=tuple(flatten(x.src for x in v)),
                                        arg=BasicBlock(tuple(dedup(flatten([y.arg.ctx for y in v]))), v[0].arg.lst, k)),), arg=len(v))
      for u in v: new_forks[u] = out
  sink = sink.substitute(new_forks)

  # reorder ops in block for speed
  sink = sink.substitute({u:newu for u in sink.toposort if u.op is Ops.BLOCK and (newu:=block_reorder(u)) is not u})

  # final rewrite to merge all blocks into one
  sink = graph_rewrite(sink, pm_block_merge, ctx=children)

  # there should just be one block left, with a few parents with 0 srcs (now done in a rewriter)
  sink = graph_rewrite(sink, pm_block_finalize)

  # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  if not skip_check: type_verify(sink.arg.lst)

  # return the list. TODO: refactor to return the UOp
  return list(sink.arg.lst)
