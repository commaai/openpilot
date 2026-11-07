# this is a temporary intermediate step while we remove this index style
from tinygrad.uop.ops import PatternMatcher, UPat, Ops
from tinygrad.dtype import Invalid, dtypes

def move_where_load(gate, l, a): return l.replace(src=(l.src[0], a.cast(l.dtype), l.src[2]))
def ungate(x, gate): return x.replace(src=tuple(s.src[1] if s.op is Ops.WHERE and s.src[0] is gate and s.src[2].is_invalid else s for s in x.src))
gated_mop = UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(), UPat.var("gate").where(UPat(), UPat(arg=Invalid))), name="mop", allow_any_len=True)

pm_move_gates_from_index = PatternMatcher([
  # here we create the alt value for load to be 0s and remove the where Invalid
  (gated_mop.load(name="l"), lambda mop,gate,l: ungate(mop, gate).load(l.vconst_like(0), gate)),
  (gated_mop.store(UPat.var("data")), lambda mop,gate,data: ungate(mop, gate).store(data, gate)),

  # Where after gated load becomes alt value
  (UPat.var("gate").where(UPat().load(UPat(), UPat.var("gate", dtype=dtypes.bool), name="l"), UPat.var("a")), move_where_load),
  (UPat.var("gate").where(UPat.var("a"), UPat().load(UPat(), ~UPat.var("gate", dtype=dtypes.bool), name="l")), move_where_load),
])
