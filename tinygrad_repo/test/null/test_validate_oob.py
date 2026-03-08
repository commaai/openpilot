import unittest
from tinygrad import dtypes, Variable
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, AxisType
from test.test_uops import to_uops_list

class TestValidateOOB(unittest.TestCase):
  """Test z3 validation of index bounds for different ALU ops and patterns."""

  # basic index patterns
  def test_const_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      to_uops_list([buf.index(UOp.const(dtypes.int, 0), ptr=True).load(dtype=dtypes.int)])  # valid
      to_uops_list([buf.index(UOp.const(dtypes.int, 15), ptr=True).load(dtype=dtypes.int)])  # valid (last element)
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.const(dtypes.int, 16), ptr=True).load(dtype=dtypes.int)])  # off by one
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.const(dtypes.int, 42), ptr=True).load(dtype=dtypes.int)])  # way out

  def test_variable_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      to_uops_list([buf.index(Variable("i", 0, 15), ptr=True).load(dtype=dtypes.int)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("i", 0, 20), ptr=True).load(dtype=dtypes.int)])  # oob
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("i", -5, 10), ptr=True).load(dtype=dtypes.int)])  # negative

  def test_range_with_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      r = UOp.range(42, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r.valid(r < 16), ptr=True).load(dtype=dtypes.int)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r.valid(r < 17), ptr=True).load(dtype=dtypes.int)])  # oob

  def test_variable_with_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      v = Variable("v", -5, 80)
      to_uops_list([buf.index(v.valid((v >= 0) & (v < 16)), ptr=True).load(dtype=dtypes.int)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(v.valid(v < 20), ptr=True).load(dtype=dtypes.int)])  # negative not masked

  def test_gated_store(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      v = Variable("v", 0, 20)
      to_uops_list([buf.index(v.valid(v < 16)).store(0)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(v.valid(v < 20)).store(0)])  # oob

  # ALU ops in index
  def test_idiv(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      to_uops_list([buf.index(UOp.range(32, 0, AxisType.GLOBAL) // 2, ptr=True).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.range(34, 0, AxisType.GLOBAL) // 2, ptr=True).load(dtype=dtypes.int)])  # 0..16 oob

  def test_mod(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      r = UOp.range(100, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r % 16, ptr=True).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r % 20, ptr=True).load(dtype=dtypes.int)])  # 0..19 oob

  def test_shr(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      to_uops_list([buf.index(UOp.range(64, 0, AxisType.GLOBAL) >> 2, ptr=True).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.range(128, 0, AxisType.GLOBAL) >> 2, ptr=True).load(dtype=dtypes.int)])  # 0..31 oob

  def test_shl(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(64), (), 0)
      r = UOp.range(8, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r << 2, ptr=True).load(dtype=dtypes.int)])  # 0..28 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r << 4, ptr=True).load(dtype=dtypes.int)])  # 0..112 oob

  def test_and(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      r = UOp.range(100, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r & 15, ptr=True).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r & 31, ptr=True).load(dtype=dtypes.int)])  # 0..31 oob

  def test_max(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      to_uops_list([buf.index(Variable("v", -10, 15).maximum(0), ptr=True).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("v2", -10, 20).maximum(0), ptr=True).load(dtype=dtypes.int)])  # 0..20 oob

  def test_xor_in_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      r = UOp.range(32, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r.valid((r < 8) ^ ((r >= 8) & (r < 16))), ptr=True).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r.valid((r < 10) ^ (r >= 20)), ptr=True).load(dtype=dtypes.int)])  # 0..9,20..31 oob

  # cast patterns
  def test_float_cast_in_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      r = UOp.range(20, 0)
      i = (r.cast(dtypes.float) * 0.68).trunc().cast(dtypes.int)
      to_uops_list([buf.index(i.valid((i >= 0) & (i < 16)), ptr=True).load(dtype=dtypes.int)])

  def test_bool_cast_in_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp(Ops.PARAM, dtypes.int.ptr(1), (), 0)
      r = UOp.range(20, 0)
      to_uops_list([buf.index(r.valid(r.cast(dtypes.bool).logical_not()), ptr=True).load(dtype=dtypes.int)])  # only r=0 valid

  # load result as index/mask
  def test_load_as_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf0 = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      buf1 = UOp(Ops.PARAM, dtypes.int.ptr(64), (), 1)
      r = UOp.range(42, 0, AxisType.GLOBAL)
      ld0 = buf0.index(r.valid(r < 8), ptr=True).load(dtype=dtypes.int).cast(dtypes.index)
      to_uops_list([buf1.index((ld0 * 2).valid((ld0 >= 0) & (ld0 < 32)), ptr=True).load(dtype=dtypes.int)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf1.index((ld0 * 2).valid((ld0 >= 0) & (ld0 < 64)), ptr=True).load(dtype=dtypes.int)])  # oob

  def test_load_bool_as_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf_bool = UOp(Ops.PARAM, dtypes.bool.ptr(16), (), 0)
      buf_int = UOp(Ops.PARAM, dtypes.int.ptr(8), (), 1)
      gidx = UOp(Ops.SPECIAL, dtypes.index, (UOp.const(dtypes.index, 16),), "gidx0")
      ld_bool = buf_bool.index(gidx, ptr=True).load()
      with self.assertRaises(RuntimeError):
        to_uops_list([buf_int.index(gidx.valid(ld_bool), ptr=True).load()])  # gidx 0..15, buf_int size 8

  # skipped tests (moved from test_uop_graph.py)
  @unittest.skip("if not allowed in graph")
  def test_in_bounds_access_gated_local(self):
    with Context(CHECK_OOB=1):
      # Define buffers
      gbuf = UOp(Ops.PARAM, dtypes.uint.ptr(400), (), 0)
      sbuf = UOp(Ops.DEFINE_LOCAL, dtypes.uint.ptr(8, addrspace=AddrSpace.LOCAL), (), "temp0")

      # Define indices, valids and barrier
      gidx = UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 416),), "gidx0")
      lidx = UOp(Ops.SPECIAL, dtypes.int, (UOp.const(dtypes.int, 10),), "lidx0")

      gate = (gidx<400) & (lidx<8)

      local_store = UOp(Ops.STORE, dtypes.void, (sbuf.index(lidx, lidx<8), UOp.const(dtypes.uint, 1)))

      barrier = UOp(Ops.BARRIER, dtypes.void, (local_store,))
      if_barrier = UOp(Ops.IF, dtypes.void, (gate, barrier))

      # Load from local memory (after the IF/barrier)
      local_load = UOp(Ops.LOAD, dtypes.uint, (sbuf.index(lidx, ptr=True), if_barrier))

      # Store to global memory
      global_store = UOp(Ops.STORE, dtypes.void, (gbuf.index(gidx), local_load))
      to_uops_list([global_store])

  @unittest.skip("Bool load is not supported yet")
  def test_load_mask(self):
    with Context(CHECK_OOB=1):
      glbl0 = UOp(Ops.PARAM, dtypes.int.ptr(16), (), 0)
      mask = UOp(Ops.PARAM, dtypes.bool.ptr(16), (), 0)
      ridx = UOp.range(20, 0)
      ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(UOp.const(ridx, ridx<16&mask), ptr=True)))
      to_uops_list([ld0])

if __name__ == "__main__":
  unittest.main()
