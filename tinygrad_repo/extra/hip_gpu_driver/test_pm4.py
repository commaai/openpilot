import time
from hexdump import hexdump
from tinygrad import Tensor, Device
import tinygrad.runtime.autogen.amd_gpu as amd_gpu
import tinygrad.runtime.autogen.kfd as kfd
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.runtime.ops_amd import kio, AMDProgram
from tinygrad.helpers import to_mv

DISPATCH_INIT_VALUE = 0x21 | 0x8000

#mmCOMPUTE_START_X = 0x2e04
#mmCOMPUTE_PGM_LO = 0x2e0c

BASE_ADDR = 0x00001260
PACKET3_SET_SH_REG_START = 0x2c00
SUB = PACKET3_SET_SH_REG_START - BASE_ADDR

regCOMPUTE_PGM_LO = 0x1bac - SUB
regCOMPUTE_START_X = 0x1ba4 - SUB
regCOMPUTE_NUM_THREAD_X = 0x1ba7 - SUB
regCOMPUTE_USER_DATA_0 = 0x1be0 - SUB
regCOMPUTE_USER_DATA_8 = 0x1be8 - SUB

regCOMPUTE_PGM_RSRC1 = 0x1bb2 - SUB
regCOMPUTE_PGM_RSRC2 = 0x1bb3 - SUB

# DEBUG=6 python3 extra/hip_gpu_driver/test_pm4.py
# sudo umr -i 1 -s amd744c.gfx1100 --sbank 1 1 2 | grep regCOMPUTE

# 0x00009025

COMPUTE_SHADER_EN = 1
USE_THREAD_DIMENSIONS = 1 << 5
CS_W32_EN = 1 << 15

def format_struct(s):
  sdats = []
  for field_name, field_type in s._fields_:
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

if __name__ == "__main__":
  dev = Device["KFD"]

  a = Tensor([0.,1.,2.], device="KFD").realize()
  b = a + 7
  b.uop.buffer.allocate()
  si = b.schedule()[-1]
  runner = dev.get_runner(*si.ast)
  prg: AMDProgram = runner.clprg
  print("device initted")

  # Compute Queue

  gart_compute = dev._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
  eop_buffer = dev._gpu_alloc(0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
  compute_ring = dev._gpu_alloc(0x800000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT, uncached=True)
  ctx_save_restore_address = dev._gpu_alloc(0x2C02000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
  compute_queue = kio.create_queue(dev.kfd, ring_base_address=compute_ring.va_addr, ring_size=compute_ring.size, gpu_id=dev.gpu_id,
    queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
    #eop_buffer_address=eop_buffer.va_addr, eop_buffer_size=eop_buffer.size,
    #ctx_save_restore_address=ctx_save_restore_address.va_addr, ctx_save_restore_size=ctx_save_restore_address.size,
    #ctl_stack_size = 0xa000,
    write_pointer_address=gart_compute.va_addr, read_pointer_address=gart_compute.va_addr+8)
  compute_doorbell = to_mv(dev.doorbells + compute_queue.doorbell_offset - dev.doorbells_base, 4).cast("I")

  #scratch = dev._gpu_alloc(0x10000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)
  ka = to_mv(dev.kernargs_ptr, 0x10).cast("Q")
  ka[0] = b.uop.buffer._buf.va_addr
  ka[1] = a.uop.buffer._buf.va_addr

  compute_read_pointer = to_mv(compute_queue.read_pointer_address, 8).cast("Q")
  compute_write_pointer = to_mv(compute_queue.write_pointer_address, 8).cast("Q")

  hexdump(to_mv(prg.handle, 0x40))
  code = hsa.amd_kernel_code_t.from_address(prg.handle)

  #print(format_struct(code))
  #print("code")
  #hexdump(to_mv(code_ptr, 0x100))
  #runner.local_size = [2,1,1]

  print(runner.local_size, runner.global_size)

  #pm4_cmd += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 6), mmCOMPUTE_PGM_LO,
  #  prg.handle&0xFFFFFFFF, prg.handle>>32, 0, 0, (scratch.va_addr>>8)&0xFFFFFFFF, scratch.va_addr>>40]
  code_ptr = (prg.handle + code.kernel_code_entry_byte_offset) >> 8
  pm4_cmd  = [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 6), regCOMPUTE_PGM_LO, code_ptr&0xFFFFFFFF, code_ptr>>32, 0, 0, 0, 0]
  pm4_cmd += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), regCOMPUTE_PGM_RSRC1, code.compute_pgm_rsrc1, code.compute_pgm_rsrc2]
  pm4_cmd += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), regCOMPUTE_USER_DATA_0, dev.kernargs_ptr&0xFFFFFFFF, dev.kernargs_ptr>>32]
  #pm4_cmd += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), regCOMPUTE_USER_DATA_0, 0, 0]
  pm4_cmd += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 8), regCOMPUTE_START_X, 0,0,0,
              runner.local_size[0],runner.local_size[1],runner.local_size[2],0,0]
  # disabled USE_THREAD_DIMENSIONS
  pm4_cmd += [amd_gpu.PACKET3(amd_gpu.PACKET3_DISPATCH_DIRECT, 3),
              runner.global_size[0],runner.global_size[1],runner.global_size[2], CS_W32_EN | COMPUTE_SHADER_EN]

  #pm4_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_NOP, 0x3fff)]*0x200

  """
  addr=0x0
  sz=(1 << 64)-1
  gli=0
  glv=0
  glk=0
  gl1=0
  gl2=0
  pm4_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_ACQUIRE_MEM, 6), 0,
              sz & 0xffffffff, (sz >> 32) & 0xff, addr & 0xffffffff, (addr >> 32) & 0xffffff, 0,
              amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | \
              amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) | \
              amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2)]
  print(pm4_cmd)
  """

  wptr = 0
  pm4_buffer_view = to_mv(compute_ring.va_addr, compute_ring.size).cast("I")

  for j in range(0x80000):
    for i, value in enumerate(pm4_cmd): pm4_buffer_view[wptr+i] = value
    wptr += len(pm4_cmd)

    compute_write_pointer[0] = wptr
    compute_doorbell[0] = wptr
    for k in range(10):
      done = compute_read_pointer[0] == compute_write_pointer[0]
      print(compute_read_pointer[0], compute_write_pointer[0], done)
      if done: break
      time.sleep(0.01)
    break
    #break

    #print(compute_read_pointer[0])
    #time.sleep(0.05)
    #print(compute_read_pointer[0])

  #time.sleep(100)

  print(a.numpy())
  print(b.numpy())
  exit(0)

  #pm4_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 8), mmCOMPUTE_PGM_LO, 0,0,0,1,1,1,0,0]


  #pm4_cmd += [amd_gpu.PACKET3(amd_gpu.PACKET3_DISPATCH_DIRECT, )]


  #pm4_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_ACQUIRE_MEM, 6), 0,
  #          sz & 0xffffffff, (sz >> 32) & 0xff, addr & 0xffffffff, (addr >> 32) & 0xffffff, 0,
  #          amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | \
  #          amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) | \
  #          amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2)]
