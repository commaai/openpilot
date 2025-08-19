import ctypes, unittest
from tinygrad.helpers import init_c_struct_t
from tinygrad.device import Device, Buffer, BufferXfer
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.hsa import AQLQueue
from tinygrad.runtime.graph.hsa import VirtAQLQueue, HSAGraph
from tinygrad.engine.realize import ExecItem

def get_hsa_inc_prog(dev, inc=1):
  prg = f"""
extern "C" __attribute__((global)) void test_inc(int* data0) {{
  data0[0] = (data0[0]+{inc});
}}
"""
  return dev.runtime("test_inc", dev.compiler.compile(prg))

def get_hsa_buffer_and_kernargs(dev):
  test_buf = Buffer(Device.DEFAULT, 1, dtypes.int)
  test_buf.copyin(memoryview(bytearray(4))) # zero mem
  assert test_buf.as_buffer().cast('I')[0] == 0 # check mem is visible + sync to exec

  args_struct_t = init_c_struct_t(tuple([('f0', ctypes.c_void_p)]))
  kernargs = dev.alloc_kernargs(8)
  args_st = args_struct_t.from_address(kernargs)
  args_st.__setattr__('f0', test_buf._buf)
  dev.flush_hdp()
  return test_buf, kernargs

@unittest.skipUnless(Device.DEFAULT == "HSA", "only run on HSA")
class TestHSADriver(unittest.TestCase):
  def test_hsa_simple_enqueue(self):
    dev = Device[Device.DEFAULT]
    queue = AQLQueue(dev, sz=256)

    clprg = get_hsa_inc_prog(dev, inc=1)
    test_buf, kernargs = get_hsa_buffer_and_kernargs(dev)

    queue.submit_kernel(clprg, [1,1,1], [1,1,1], kernargs)
    queue.wait()

    assert test_buf.as_buffer().cast('I')[0] == 1, f"{test_buf.as_buffer().cast('I')[0]} != 1, all packets executed?"
    del queue

  def test_hsa_ring_enqueue(self):
    dev = Device[Device.DEFAULT]

    queue_size = 256
    exec_cnt = int(queue_size * 1.5)
    queue = AQLQueue(dev, sz=queue_size)

    clprg_inc1 = get_hsa_inc_prog(dev, inc=1)
    clprg_inc2 = get_hsa_inc_prog(dev, inc=2)
    test_buf, kernargs = get_hsa_buffer_and_kernargs(dev)

    for _ in range(exec_cnt):
      queue.submit_kernel(clprg_inc1, [1,1,1], [1,1,1], kernargs)
    for _ in range(exec_cnt):
      queue.submit_kernel(clprg_inc2, [1,1,1], [1,1,1], kernargs)
    queue.wait()

    expected = exec_cnt + exec_cnt * 2
    assert test_buf.as_buffer().cast('I')[0] == expected, f"{test_buf.as_buffer().cast('I')[0]} != {expected}, all packets executed?"
    del queue

  def test_hsa_blit_enqueue(self):
    dev = Device[Device.DEFAULT]

    queue_size = 256
    exec_cnt = 178
    queue = AQLQueue(dev, sz=queue_size)

    test_buf, kernargs = get_hsa_buffer_and_kernargs(dev)

    # Using VirtAQLQueue to blit them
    virt_queue_packets_cnt = 31
    virt_queue = VirtAQLQueue(dev, sz=virt_queue_packets_cnt)

    clprogs = []
    sum_per_blit = 0
    for i in range(virt_queue_packets_cnt):
      sum_per_blit += i+1
      clprogs.append(get_hsa_inc_prog(dev, inc=i+1))

    for i in range(virt_queue_packets_cnt):
      virt_queue.submit_kernel(clprogs[i], [1,1,1], [1,1,1], kernargs)

    for _ in range(exec_cnt):
      queue.blit_packets(virt_queue.queue_base, virt_queue.packets_count)
    queue.wait()

    expected = exec_cnt * sum_per_blit
    assert test_buf.as_buffer().cast('I')[0] == expected, f"{test_buf.as_buffer().cast('I')[0]} != {expected}, all packets executed?"
    del queue, clprogs

  def test_hsa_copies_sync(self):
    d0, d1 = f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"

    test_buf0 = Buffer(d0, 1, dtypes.int)
    test_buf1 = Buffer(d0, 1, dtypes.int)
    test_buf2 = Buffer(d1, 1, dtypes.int)
    test_buf0.copyin(memoryview(bytearray(1*4)))
    test_buf1.copyin(memoryview(bytearray(1*4)))
    test_buf2.copyin(memoryview(bytearray(1*4)))

    jit_cache = [ExecItem(BufferXfer(), [test_buf0, test_buf2]), ExecItem(BufferXfer(), [test_buf2, test_buf1])]
    graph = HSAGraph(jit_cache, [], {})

    for i in range(10000):
      test_buf0.copyin(memoryview(bytearray(1*4)))
      test_buf2.copyin(memoryview(bytearray(int.to_bytes(4, length=1*4, byteorder='little'))))
      graph([], {})
      assert test_buf0.as_buffer().cast('I')[0] == 4
      assert test_buf2.as_buffer().cast('I')[0] == 0

if __name__ == '__main__':
  unittest.main()
