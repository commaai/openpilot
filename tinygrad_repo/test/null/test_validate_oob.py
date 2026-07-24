import unittest
from tinygrad import dtypes, Variable
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, AxisType
from test.helpers import to_uops_list

class TestValidateOOB(unittest.TestCase):
  """Test z3 validation of index bounds for different ALU ops and patterns."""

  # basic index patterns
  def test_const_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      to_uops_list([buf.index(UOp.const(dtypes.int, 0)).load(dtype=dtypes.int)])  # valid
      to_uops_list([buf.index(UOp.const(dtypes.int, 15)).load(dtype=dtypes.int)])  # valid (last element)
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.const(dtypes.int, 16)).load(dtype=dtypes.int)])  # off by one
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.const(dtypes.int, 42)).load(dtype=dtypes.int)])  # way out

  def test_variable_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      to_uops_list([buf.index(Variable("i", 0, 15)).load(dtype=dtypes.int)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("i", 0, 20)).load(dtype=dtypes.int)])  # oob
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("i", -5, 10)).load(dtype=dtypes.int)])  # negative

  def test_range_with_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      r = UOp.range(42, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r.valid(r < 16)).load(dtype=dtypes.int)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r.valid(r < 17)).load(dtype=dtypes.int)])  # oob

  def test_variable_with_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      v = Variable("v", -5, 80)
      to_uops_list([buf.index(v.valid((v >= 0) & (v < 16))).load(dtype=dtypes.int)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(v.valid(v < 20)).load(dtype=dtypes.int)])  # negative not masked

  def test_gated_store(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      v = Variable("v", 0, 20)
      to_uops_list([buf.index(v.valid(v < 16)).store(0)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(v.valid(v < 20)).store(0)])  # oob

  # ALU ops in index
  def test_floordiv(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      to_uops_list([buf.index(UOp.range(32, 0, AxisType.GLOBAL) // 2).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.range(34, 0, AxisType.GLOBAL) // 2).load(dtype=dtypes.int)])  # 0..16 oob

  def test_mod(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      r = UOp.range(100, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r % 16).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r % 20).load(dtype=dtypes.int)])  # 0..19 oob

  def test_shr(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      to_uops_list([buf.index(UOp.range(64, 0, AxisType.GLOBAL) >> 2).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.range(128, 0, AxisType.GLOBAL) >> 2).load(dtype=dtypes.int)])  # 0..31 oob

  def test_shl(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (64,))
      r = UOp.range(8, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r << 2).load(dtype=dtypes.int)])  # 0..28 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r << 4).load(dtype=dtypes.int)])  # 0..112 oob

  def test_and(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      r = UOp.range(100, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r & 15).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r & 31).load(dtype=dtypes.int)])  # 0..31 oob

  def test_max(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      to_uops_list([buf.index(Variable("v", -10, 15).maximum(0)).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("v2", -10, 20).maximum(0)).load(dtype=dtypes.int)])  # 0..20 oob

  def test_xor_in_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      r = UOp.range(32, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r.valid((r < 8) ^ ((r >= 8) & (r < 16)))).load(dtype=dtypes.int)])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r.valid((r < 10) ^ (r >= 20))).load(dtype=dtypes.int)])  # 0..9,20..31 oob

  # cast patterns
  def test_float_cast_in_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (16,))
      r = UOp.range(20, 0)
      i = (r.cast(dtypes.float) * 0.68).trunc().cast(dtypes.int)
      to_uops_list([buf.index(i.valid((i >= 0) & (i < 16))).load(dtype=dtypes.int)])

  def test_bool_cast_in_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, (1,))
      r = UOp.range(20, 0)
      to_uops_list([buf.index(r.valid(r.cast(dtypes.bool).logical_not())).load(dtype=dtypes.int)])  # only r=0 valid

  # load result as index/mask
  def test_load_as_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf0 = UOp.param(0, dtypes.int, (16,))
      buf1 = UOp.param(1, dtypes.int, (64,))
      r = UOp.range(42, 0, AxisType.GLOBAL)
      ld0 = buf0.index(r.valid(r < 8)).load(dtype=dtypes.int).cast(dtypes.weakint)
      to_uops_list([buf1.index((ld0 * 2).valid((ld0 >= 0) & (ld0 < 32))).load(dtype=dtypes.int)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf1.index((ld0 * 2).valid((ld0 >= 0) & (ld0 < 64))).load(dtype=dtypes.int)])  # oob

  def test_load_bool_as_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf_bool = UOp.param(0, dtypes.bool, (16,))
      buf_int = UOp.param(1, dtypes.int, (8,))
      gidx = UOp(Ops.SPECIAL, src=(UOp.const(dtypes.weakint, 16),), arg="gidx0")
      ld_bool = buf_bool.index(gidx).load()
      with self.assertRaises(RuntimeError):
        to_uops_list([buf_int.index(gidx.valid(ld_bool)).load()])  # gidx 0..15, buf_int size 8

  # skipped tests (moved from test_uop_graph.py)
  @unittest.skip("if not allowed in graph")
  def test_in_bounds_access_gated_local(self):
    with Context(CHECK_OOB=1):
      # Define buffers
      gbuf = UOp.param(0, dtypes.uint, (400,))
      sbuf = UOp.placeholder((8,), dtypes.uint, slot=0, addrspace=AddrSpace.LOCAL)

      # Define indices, valids and barrier
      gidx = UOp(Ops.SPECIAL, src=(UOp.const(dtypes.int, 416),), arg="gidx0")
      lidx = UOp(Ops.SPECIAL, src=(UOp.const(dtypes.int, 10),), arg="lidx0")

      gate = (gidx<400) & (lidx<8)

      local_store = sbuf.index(lidx.valid(lidx<8)).store(UOp.const(dtypes.uint, 1))

      barrier = UOp(Ops.BARRIER, src=(local_store,))
      if_barrier = UOp(Ops.IF, src=(gate, barrier))

      # Load from local memory (after the IF/barrier)
      local_load = UOp(Ops.LOAD, src=(sbuf.index(lidx), if_barrier))

      # Store to global memory
      global_store = UOp(Ops.STORE, src=(gbuf.index(gidx), local_load))
      to_uops_list([global_store])

  @unittest.skip("Bool load is not supported yet")
  def test_load_mask(self):
    with Context(CHECK_OOB=1):
      glbl0 = UOp.param(0, dtypes.int, (16,))
      mask = UOp.param(0, dtypes.bool, (16,))
      ridx = UOp.range(20, 0)
      ld0 = UOp(Ops.LOAD, src=(glbl0.index(UOp.const(ridx, ridx<16&mask))))
      to_uops_list([ld0])

if __name__ == "__main__":
  unittest.main()
