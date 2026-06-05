import unittest
from tinygrad import Device, Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.runtime.graph.hcq import HCQGraph
from tinygrad.runtime.support.hcq import HCQCompiled
from tinygrad.runtime.support.usb import USBMMIOInterface
from test.mockgpu.usb import MockUSB

@unittest.skipUnless(issubclass(type(Device[Device.DEFAULT]), HCQCompiled), "HCQ device required to run")
class TestHCQUnit(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT == "CPU", "requires non-CPU HCQ device")
  def test_supports_uop(self):
    d0, cpu_dev = Device[Device.DEFAULT], Device["CPU"]

    @TinyJit
    def f(inp, inp_cpu):
      return (inp + 1.0).contiguous().realize(), (inp_cpu + 1.0).contiguous().realize()
    inp, inp_cpu = Tensor.randn(10, 10, device=Device.DEFAULT).realize(), Tensor.randn(10, 10, device="CPU").realize()
    for _ in range(5): f(inp, inp_cpu)

    # construct minimal CALL UOps for supports_uop (graphs only see PROGRAMs after compile_linear)
    gpu_call = UOp(Ops.PROGRAM, src=(UOp.sink(), UOp(Ops.DEVICE, arg=Device.DEFAULT))).call(UOp.new_buffer(Device.DEFAULT, 1, dtypes.float))
    cpu_call = UOp(Ops.PROGRAM, src=(UOp.sink(), UOp(Ops.DEVICE, arg="CPU"))).call(UOp.new_buffer("CPU", 1, dtypes.float))
    gpu_devs = [d0]

    # local MMIO: GPU works alone and with CPU in batch (cpu_support=True)
    assert HCQGraph.supports_uop(gpu_devs, gpu_call) is True
    assert HCQGraph.supports_uop(gpu_devs, cpu_call) is True
    assert HCQGraph.supports_uop(gpu_devs + [cpu_dev], gpu_call) is True

    # USB MMIO: GPU-only still works, but CPU batching must be rejected (cpu_support=False)
    orig_view = d0.timeline_signal.base_buf.view
    try:
      d0.timeline_signal.base_buf.view = USBMMIOInterface(MockUSB(bytearray(256)), 0, 16, fmt='B')
      assert HCQGraph.supports_uop(gpu_devs, gpu_call) is True
      assert HCQGraph.supports_uop(gpu_devs, cpu_call) is False
      assert HCQGraph.supports_uop(gpu_devs + [cpu_dev], gpu_call) is False
    finally:
      d0.timeline_signal.base_buf.view = orig_view

if __name__ == "__main__":
  unittest.main()
