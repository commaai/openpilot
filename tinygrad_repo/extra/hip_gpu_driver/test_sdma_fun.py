import ctypes, mmap, time
from tinygrad.runtime.ops_amd import AMDDevice, kio, sdma_pkts, libc
import tinygrad.runtime.autogen.amd_sdma as amd_sdma
import tinygrad.runtime.autogen.kfd as kfd
from tinygrad.helpers import to_mv

if __name__ == "__main__":
  dev = AMDDevice()

  sdma_ring = dev._gpu_alloc(1 << 22, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)
  gart = dev._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
  sdma_queue = kio.create_queue(AMDDevice.kfd,
    ring_base_address=sdma_ring.va_addr, ring_size=sdma_ring.size, gpu_id=dev.gpu_id,
    queue_type=kfd.KFD_IOC_QUEUE_TYPE_SDMA, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
    write_pointer_address=gart.va_addr + 0x100, read_pointer_address=gart.va_addr + 0x108)

  doorbells_base = sdma_queue.doorbell_offset & (~0xfff)
  doorbells = libc.mmap(0, 8192, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, AMDDevice.kfd, doorbells_base)

  sdma_read_pointer = to_mv(sdma_queue.read_pointer_address, 8).cast("Q")
  sdma_write_pointer = to_mv(sdma_queue.write_pointer_address, 8).cast("Q")
  sdma_doorbell = to_mv(doorbells + sdma_queue.doorbell_offset - doorbells_base, 4).cast("I")

  test_write_page = dev._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR, uncached=True)
  cmd = sdma_pkts.timestamp(op=amd_sdma.SDMA_OP_TIMESTAMP, sub_op=amd_sdma.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL, addr=test_write_page.va_addr)

  sdma_doorbell_value = 0
  def blit_sdma_command(cmd):
    ctypes.memmove(sdma_ring.va_addr + (sdma_doorbell_value % sdma_ring.size), ctypes.addressof(cmd), sz:=ctypes.sizeof(cmd))
    return sz

  while True:
    sdma_doorbell_value += blit_sdma_command(cmd)
    sdma_write_pointer[0] = sdma_doorbell_value
    sdma_doorbell[0] = sdma_doorbell_value
    while sdma_read_pointer[0] != sdma_write_pointer[0]: continue
    tm = to_mv(test_write_page.va_addr, 0x1000).cast("Q")[0]/1e8
    print(f"{tm:.3f} s @ 0x{sdma_ring.va_addr + (sdma_doorbell_value % sdma_ring.size):X} R:0x{sdma_queue.read_pointer_address:X} W:0x{sdma_queue.write_pointer_address:X}")
    time.sleep(0.01)




