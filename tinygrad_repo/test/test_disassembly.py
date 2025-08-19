import unittest, io
from tinygrad import Tensor, dtypes
from contextlib import redirect_stdout
from tinygrad.device import Device
from tinygrad.helpers import OSX
from tinygrad.engine.realize import get_program

class TestDisassembly(unittest.TestCase):
  # TODO: fails on llvm. llvm.LLVMGetHostCPUName() returns "generic"
  @unittest.skipUnless(Device.DEFAULT in ("CPU",) and OSX, "m series cpus support fp16 arithmetic")
  def test_float16_alu(self):
    c = Tensor([1], dtype=dtypes.float16) + Tensor([1], dtype=dtypes.float16)
    s = c.schedule()[-1]
    p = get_program(s.ast, Device[Device.DEFAULT].renderer)
    lib = Device[Device.DEFAULT].compiler.compile(p.src)
    out = io.StringIO()
    with redirect_stdout(out): Device[Device.DEFAULT].compiler.disassemble(lib)
    assert "fcvt" not in out.getvalue()

if __name__ == "__main__":
  unittest.main()