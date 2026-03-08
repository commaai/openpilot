# ruff: noqa: F405
import unittest, subprocess, os
from extra.assembly.amd.autogen.rdna3.ins import *  # noqa: F403
from extra.assembly.amd.dsl import s, v, Inst, NULL

def assemble_kernel(insts:list[Inst], name:str="test") -> str:
  kd = {"next_free_vgpr": 8, "next_free_sgpr": 8, "wavefront_size32": 1, "user_sgpr_kernarg_segment_ptr": 1, "kernarg_size": 8}
  disasm = "\n".join(inst.disasm() for inst in insts)
  hsasrc = f".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:\n{disasm}\n"
  return hsasrc + f".rodata\n.p2align 6\n.amdhsa_kernel {name}\n" + "\n".join(f".amdhsa_{k} {v}" for k, v in kd.items()) + "\n.end_amdhsa_kernel"

def _run(code:str, timeout:float=15.0) -> subprocess.CompletedProcess:
  # TODO: AM_RESET is required for now, so subprocesses
  return subprocess.run(["python", "-c", code], env={**os.environ, "AMD": "1"}, capture_output=True, text=True, timeout=timeout)

def _run_asm(asm_src:str) -> subprocess.CompletedProcess:
  return _run('from tinygrad.device import Device; from tinygrad.runtime.ops_amd import AMDProgram; '
              'from tinygrad.runtime.support.compiler_amd import HIPCompiler; dev = Device["AMD"]; '
              f'AMDProgram(dev, "test", HIPCompiler(dev.arch).compile("""{asm_src}"""))('
              'dev.allocator.alloc(64), global_size=(1,1,1), local_size=(1,1,1), wait=True)')

def _verify_recovery() -> subprocess.CompletedProcess:
  return _run('from tinygrad import Tensor; t = Tensor([1.0, 2.0], device="AMD").realize(); assert (t + 1).numpy().tolist() == [2.0, 3.0]')

_ILLEGAL_INST_ASM = ".text\n.globl test\n.p2align 8\n.type test,@function\ntest:\n.byte 0xff,0xff,0xff,0xff\ns_endpgm\n" \
  ".rodata\n.p2align 6\n.amdhsa_kernel test\n.amdhsa_next_free_vgpr 8\n.amdhsa_next_free_sgpr 8\n" \
  ".amdhsa_wavefront_size32 1\n.amdhsa_user_sgpr_kernarg_segment_ptr 1\n.amdhsa_kernarg_size 8\n.end_amdhsa_kernel"

@unittest.skipIf(os.environ.get("AMD") != "1" or os.environ.get("MOCKGPU") == "1", "AMD with AM driver required")
class TestAMFaultRecovery(unittest.TestCase):
  def _run_kernel(self, insts: list[Inst]) -> subprocess.CompletedProcess: return _run_asm(assemble_kernel(insts))

  def _assert_fault_and_recovery(self, result:subprocess.CompletedProcess):
    if result.stdout.strip(): print(f"\nstdout: {result.stdout.strip()}")
    if result.stderr.strip(): print(f"\nstderr: {result.stderr.strip()}")
    self.assertNotEqual(result.returncode, 0, f"Expected fault but succeeded: {result.stdout}")
    self.assertEqual(_verify_recovery().returncode, 0)


class TestGlobalMemoryFaults(TestAMFaultRecovery):
  def test_global_load_unmapped(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD),
             global_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(vmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_global_store_unmapped(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD), v_mov_b32_e32(v[2], 0x12345678),
             global_store_b32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(vmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_global_null_ptr(self):
    insts = [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 0),
             global_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(vmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_global_misaligned_b64(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0001), v_mov_b32_e32(v[1], 0xDEAD),
             global_load_b64(v[2:3], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(vmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_global_misaligned_b128(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0004), v_mov_b32_e32(v[1], 0xDEAD),
             global_load_b128(v[2:5], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(vmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))


class TestSMEMFaults(TestAMFaultRecovery):
  def test_smem_null_base(self):
    insts = [s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
             s_load_b32(s[4], s[2:3], 0, soffset=NULL), s_waitcnt(lgkmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_smem_unmapped_address(self):
    insts = [s_mov_b32(s[2], 0xBEEF0000), s_mov_b32(s[3], 0xDEAD),
             s_load_b32(s[4], s[2:3], 0, soffset=NULL), s_waitcnt(lgkmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_smem_misaligned_b64(self):
    insts = [s_mov_b32(s[2], 0xBEEF0004), s_mov_b32(s[3], 0xDEAD),
             s_load_b64(s[4:5], s[2:3], 0, soffset=NULL), s_waitcnt(lgkmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_smem_misaligned_b128(self):
    insts = [s_mov_b32(s[2], 0xBEEF0004), s_mov_b32(s[3], 0xDEAD),
             s_load_b128(s[4:7], s[2:3], 0, soffset=NULL), s_waitcnt(lgkmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))


class TestIllegalInstruction(TestAMFaultRecovery):
  def test_malformed_encoding(self):
    self._assert_fault_and_recovery(_run_asm(_ILLEGAL_INST_ASM))


class TestFlatFaults(TestAMFaultRecovery):
  def test_flat_load_unmapped(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD),
             flat_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(vmcnt=0, lgkmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_flat_store_unmapped(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD), v_mov_b32_e32(v[2], 0x12345678),
             flat_store_b32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(vmcnt=0, lgkmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))


class TestAtomicFaults(TestAMFaultRecovery):
  def test_global_atomic_unmapped(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD), v_mov_b32_e32(v[2], 1),
             global_atomic_add_u32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(vmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))

  def test_flat_atomic_unmapped(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD), v_mov_b32_e32(v[2], 1),
             flat_atomic_add_u32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(vmcnt=0, lgkmcnt=0), s_endpgm()]
    self._assert_fault_and_recovery(self._run_kernel(insts))


class TestRecovery(TestAMFaultRecovery):
  def test_recovery_after_memviol(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD),
             global_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(vmcnt=0), s_endpgm()]
    self.assertNotEqual(self._run_kernel(insts).returncode, 0)
    self.assertEqual(_verify_recovery().returncode, 0)

  def test_recovery_after_illegal_inst(self):
    self.assertNotEqual(_run_asm(_ILLEGAL_INST_ASM).returncode, 0)
    self.assertEqual(_verify_recovery().returncode, 0)

  def test_multiple_faults_recovery(self):
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD),
             global_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(vmcnt=0), s_endpgm()]
    for _ in range(3):
      self.assertNotEqual(self._run_kernel(insts).returncode, 0)
      self.assertEqual(_verify_recovery().returncode, 0)

if __name__ == "__main__":
  unittest.main()
