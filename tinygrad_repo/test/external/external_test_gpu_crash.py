# ruff: noqa: F405
"""Tests for GPU crash scenarios using AMD assembly to trigger invalid operations.

These tests intentionally cause GPU faults to verify error handling.
Run with: DEV=AMD python -m pytest test/external/external_test_gpu_crash.py -v
"""
import unittest, re, importlib
from tinygrad.device import Device
from tinygrad.renderer.amd.dsl import s, v, Inst, NULL

RDNA3_CDNA3_MAP = {"v_mov_b32_e32": "v_mov_b32_e32", "s_mov_b32": "s_mov_b32", "s_waitcnt": "s_waitcnt", "s_endpgm": "s_endpgm",
                   "global_load_b32": "global_load_dword", "global_store_b32": "global_store_dword",
                   "global_atomic_add_u32": "global_atomic_add", "flat_load_b32": "flat_load_dword",
                   "flat_store_b32": "flat_store_dword", "flat_atomic_add_u32": "flat_atomic_add", "s_load_b32": "s_load_dword"}

def assemble(code:str, name:str="test", is_cdna:bool=False) -> str:
  kd = {"next_free_vgpr": 8, "next_free_sgpr": 8, "user_sgpr_kernarg_segment_ptr": 1, "kernarg_size": 8}
  if is_cdna: kd["accum_offset"] = 8
  else: kd["wavefront_size32"] = 1
  return f".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:\n{code}\n.rodata\n.p2align 6\n.amdhsa_kernel {name}\n" + \
         "\n".join(f".amdhsa_{k} {v}" for k,v in kd.items()) + "\n.end_amdhsa_kernel"

@unittest.skipIf(Device.DEFAULT != "AMD", "AMD required")
class TestGPUCrash(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    cls.dev = Device["AMD"]
    cls.compiler = HIPCompiler(cls.dev.arch)
    cls.is_cdna = cls.dev.target[0] < 10
    ins = importlib.import_module('tinygrad.runtime.autogen.amd.' + ('cdna' if cls.is_cdna else 'rdna3') + '.ins')
    for rdna3_name, cdna3_name in RDNA3_CDNA3_MAP.items():
      setattr(cls, rdna3_name, getattr(ins, cdna3_name if cls.is_cdna else rdna3_name))

  def setUp(self):
    # Verify device works before each test
    from tinygrad import Tensor
    try:
      t = Tensor([1.0, 2.0], device="AMD").realize()
      assert (t + 1).numpy().tolist() == [2.0, 3.0]
    except Exception:
      self.fail("Device not working before test")

  def _run(self, code: str):
    from tinygrad.runtime.ops_amd import AMDProgram
    prg = AMDProgram(self.dev, "test", self.compiler.compile(assemble(code, is_cdna=self.is_cdna)))
    prg(self.dev.allocator.alloc(64), global_size=(1,1,1), local_size=(1,1,1), wait=True)

  def _run_insts(self, insts: list[Inst]):
    from test.amd.disasm import disasm
    self._run("\n".join(disasm(i) for i in insts))

  def _assert_gpu_fault(self, func):
    """Assert that func raises a RuntimeError indicating a GPU fault (not a setup error)."""
    with self.assertRaises(RuntimeError) as cm:
      func()
    err_msg = str(cm.exception).lower()
    # Verify it's a GPU fault, not a setup/device initialization error
    self.assertTrue(
      re.search(r'fault|hang|timeout|illegal|memviol', err_msg),
      f"Expected GPU fault error, got: {cm.exception}"
    )


class TestOutOfBoundsMemoryAccess(TestGPUCrash):
  """Tests for out-of-bounds memory accesses."""

  def test_global_load_null_ptr(self):
    """Global load from NULL pointer."""
    insts = [self.v_mov_b32_e32(v[0], 0), self.v_mov_b32_e32(v[1], 0),
             self.global_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))

  def test_global_store_null_ptr(self):
    """Global store to NULL pointer."""
    insts = [self.v_mov_b32_e32(v[0], 0), self.v_mov_b32_e32(v[1], 0), self.v_mov_b32_e32(v[2], 0xDEADBEEF),
             self.global_store_b32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))

  def test_global_load_unmapped_high_address(self):
    """Global load from high unmapped address (0xDEAD00000000)."""
    insts = [self.v_mov_b32_e32(v[0], 0x00000000), self.v_mov_b32_e32(v[1], 0xDEAD),
             self.global_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))

  def test_global_store_unmapped_high_address(self):
    """Global store to high unmapped address."""
    insts = [self.v_mov_b32_e32(v[0], 0x00000000), self.v_mov_b32_e32(v[1], 0xDEAD), self.v_mov_b32_e32(v[2], 0x12345678),
             self.global_store_b32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))

  def test_global_atomic_unmapped(self):
    """Atomic operation on unmapped memory."""
    insts = [self.v_mov_b32_e32(v[0], 0xBEEF0000), self.v_mov_b32_e32(v[1], 0xDEAD), self.v_mov_b32_e32(v[2], 1),
             self.global_atomic_add_u32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))


class TestSMEMFaults(TestGPUCrash):
  """Tests for scalar memory (SMEM) faults."""

  def test_smem_load_null(self):
    """SMEM load from NULL base."""
    insts = [self.s_mov_b32(s[2], 0), self.s_mov_b32(s[3], 0),
             self.s_load_b32(s[4], s[2:3], 0, soffset=NULL), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))

  def test_smem_load_unmapped(self):
    """SMEM load from unmapped address."""
    insts = [self.s_mov_b32(s[2], 0xBEEF0000), self.s_mov_b32(s[3], 0xDEAD),
             self.s_load_b32(s[4], s[2:3], 0, soffset=NULL), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))


class TestFlatMemoryFaults(TestGPUCrash):
  """Tests for FLAT memory instruction faults."""

  def test_flat_load_null(self):
    """FLAT load from NULL address."""
    insts = [self.v_mov_b32_e32(v[0], 0), self.v_mov_b32_e32(v[1], 0),
             self.flat_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))

  def test_flat_store_null(self):
    """FLAT store to NULL address."""
    insts = [self.v_mov_b32_e32(v[0], 0), self.v_mov_b32_e32(v[1], 0), self.v_mov_b32_e32(v[2], 0xDEADBEEF),
             self.flat_store_b32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))

  def test_flat_atomic_null(self):
    """FLAT atomic on NULL address."""
    insts = [self.v_mov_b32_e32(v[0], 0), self.v_mov_b32_e32(v[1], 0), self.v_mov_b32_e32(v[2], 1),
             self.flat_atomic_add_u32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), self.s_waitcnt(0), self.s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_insts(insts))


if __name__ == "__main__":
  unittest.main()
