import os, ctypes, pathlib, re, fcntl, functools, mmap, time
import tinygrad.runtime.autogen.kfd as kfd
from tinygrad.helpers import to_mv, getenv
from extra.hip_gpu_driver import hip_ioctl
import tinygrad.runtime.autogen.hsa as hsa
from hexdump import hexdump

libc = ctypes.CDLL("libc.so.6")
libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_char, ctypes.c_int]
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
MAP_NORESERVE = 0x4000
MAP_FIXED = 0x10

def kfd_ioctl(idir, nr, user_struct, fd, **kwargs):
  made = user_struct(**kwargs)
  ret = fcntl.ioctl(fd, (idir<<30) | (ctypes.sizeof(user_struct)<<16) | (ord('K')<<8) | nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def format_struct(s):
  sdats = []
  for field_name, field_type in s._fields_:
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

idirs = {"IOW": 1, "IOR": 2, "IOWR": 3}
def ioctls_from_header():
  hdr = pathlib.Path("/usr/include/linux/kfd_ioctl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(AMDKFD_IOC_[A-Z0-9_]+)\s+AMDKFD_(IOW?R?)\((0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)

  fxns = {}
  for name, idir, nr, sname in matches:
    fxns[name.replace("AMDKFD_IOC_", "").lower()] = functools.partial(kfd_ioctl, idirs[idir], int(nr, 0x10), getattr(kfd, "struct_"+sname))
  return type("KIO", (object, ), fxns)
kio = ioctls_from_header()

# sudo su -c "echo 'file drivers/gpu/drm/amd/* +p' > /sys/kernel/debug/dynamic_debug/control"

def gpu_alloc_userptr(fd, size, flags):
  addr = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
  assert addr != 0xffffffffffffffff
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=size, gpu_id=GPU_ID, flags=flags, mmap_offset=addr)
  return mem

def gpu_alloc(fd, size, flags):
  addr = libc.mmap(0, size, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  assert addr != 0xffffffffffffffff
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=size, gpu_id=GPU_ID, flags=flags)
  buf = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, mem.mmap_offset)
  assert buf != 0xffffffffffffffff
  assert addr == buf == mem.va_addr
  return mem

if __name__ == "__main__":
  fd = os.open("/dev/kfd", os.O_RDWR)
  gpu_num = getenv("GPU", 0)
  drm_fd = os.open(f"/dev/dri/renderD{128+gpu_num}", os.O_RDWR)
  with open(f"/sys/devices/virtual/kfd/kfd/topology/nodes/{1+gpu_num}/gpu_id", "r") as f: GPU_ID = int(f.read())

  #ver = kio.get_version(fd)
  st = kio.acquire_vm(fd, drm_fd=drm_fd, gpu_id=GPU_ID)
  #exit(0)

  # 0xF0000001 = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0xD6000002 = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0xD6000004 = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0x94000010 = KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, -1, 0)
  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
  #mem = kio.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xD6000004)

  #mem = gpu_alloc(fd, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
  #                            kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
  #                            kfd.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
  #arr = (ctypes.c_int32 * 1)(GPU_ID)
  #stm = kio.map_memory_to_gpu(fd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)

  arr = (ctypes.c_int32 * 1)(GPU_ID)
  rw_ptr = gpu_alloc(fd, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                 kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                 kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
  stm = kio.map_memory_to_gpu(fd, handle=rw_ptr.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1
  event_page = gpu_alloc(fd, 0x8000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                 kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                 kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
  stm = kio.map_memory_to_gpu(fd, handle=event_page.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1
  ring_base = gpu_alloc_userptr(fd, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                    kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                    kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
  stm = kio.map_memory_to_gpu(fd, handle=ring_base.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1
  signals = gpu_alloc_userptr(fd, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                    kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                    kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
  stm = kio.map_memory_to_gpu(fd, handle=signals.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1
  eop_buffer = gpu_alloc(fd, 0x1000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                                     kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                                     kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
  stm = kio.map_memory_to_gpu(fd, handle=eop_buffer.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1
  ctx_save_restore_address = gpu_alloc(fd, 0x2C02000, kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                                     kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                                     kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
  stm = kio.map_memory_to_gpu(fd, handle=ctx_save_restore_address.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1

  #113.00 ms +   0.00 ms :  0 = AMDKFD_IOC_CREATE_QUEUE                  ring_base_address:0x797465200000 write_pointer_address:0x79751C068038 read_pointer_address:0x79751C068080 doorbell_offset:0x0 ring_size:0x800000 gpu_id:0x433D queue_type:0x2 queue_per
  #centage:0x64 queue_priority:0x7 queue_id:0x0 eop_buffer_address:0x79751C064000 eop_buffer_size:0x1000 ctx_save_restore_address:0x796E52400000 ctx_save_restore_size:0x2BEA000 ctl_stack_size:0xA000

  #113.84 ms +   0.59 ms :  0 = AMDKFD_IOC_CREATE_QUEUE                  ring_base_address:0x71AC3F600000 write_pointer_address:0x71B302AB0038 read_pointer_address:0x71B302AB0080 doorbell_offset:0xD0CF400000000008 ring_size:0x800000 gpu_id:0x433D queue_typ
  #e:0x2 queue_percentage:0x64 queue_priority:0x7 queue_id:0x1 eop_buffer_address:0x71B302AAC000 eop_buffer_size:0x1000 ctx_save_restore_address:0x71AC3C800000 ctx_save_restore_size:0x2BEA000 ctl_stack_size:0xA000

  #define KFD_MMAP_TYPE_SHIFT	62
  #define KFD_MMAP_TYPE_DOORBELL	(0x3ULL << KFD_MMAP_TYPE_SHIFT)
  evt = kio.create_event(fd, event_page_offset=event_page.handle, auto_reset=1)

  nq = kio.create_queue(fd, ring_base_address=ring_base.va_addr, ring_size=0x1000, gpu_id=GPU_ID,
                        queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE,
                        queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
                        eop_buffer_address=eop_buffer.va_addr, eop_buffer_size=0x1000,
                        ctx_save_restore_address=ctx_save_restore_address.va_addr, ctx_save_restore_size=0x2C02000,
                        ctl_stack_size = 0xa000,
                        # write_pointer_address and read_pointer_address are on GART
                        #write_pointer_address=0xaaaabbbb, read_pointer_address=0xaaaacccc)
                        write_pointer_address=rw_ptr.va_addr+0, read_pointer_address=rw_ptr.va_addr+0x8)
  doorbell = libc.mmap(0, 8192, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, fd, nq.doorbell_offset)
  print("doorbell", hex(doorbell))

  to_mv(signals.va_addr, 0x40)

  """
  hexdump(to_mv(event_page.va_addr, 0x40))
  kio.set_event(fd, event_id=evt.event_id)
  hexdump(to_mv(event_page.va_addr, 0x40))
  kio.reset_event(fd, event_id=evt.event_id)
  hexdump(to_mv(event_page.va_addr, 0x40))
  """

  # KFD_EVENT_TYPE_SIGNAL

  BARRIER_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
  BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
  BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
  BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

  AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)
  EMPTY_SIGNAL = hsa.hsa_signal_t()

  ds = to_mv(rw_ptr.va_addr, 0x100).cast("Q")
  ds[0] = 1 #ring_base.va_addr + AQL_PACKET_SIZE
  ds[1] = 0 #ring_base.va_addr
  #libc.memset(rw_ptr.va_addr, 0xaa, 0x100)
  #hexdump(to_mv(rw_ptr.va_addr, 0x100))

  #packet = hsa.hsa_barrier_and_packet_t.from_address(rw_ptr.va_addr+0x38)
  packet = hsa.hsa_barrier_and_packet_t.from_address(ring_base.va_addr)
  packet.reserved0 = 0
  packet.reserved1 = 0
  for i in range(5): packet.dep_signal[i] = EMPTY_SIGNAL
  #packet.dep_signal[0] = hsa.hsa_signal_t(evt.event_id)
  packet.reserved2 = 0
  #packet.completion_signal = EMPTY_SIGNAL
  packet.completion_signal = hsa.hsa_signal_t(signals.va_addr)
  packet.header = BARRIER_HEADER
  hexdump(to_mv(ring_base.va_addr, AQL_PACKET_SIZE))

  # _HsaEventData
  to_mv(signals.va_addr, 0x40).cast("Q")[0] = 1
  to_mv(signals.va_addr, 0x40).cast("Q")[1] = 1
  #to_mv(signals.va_addr, 0x40).cast("Q")[2] = event_page
  to_mv(signals.va_addr, 0x40).cast("Q")[2] = event_page.va_addr + evt.event_slot_index*8  # HWData2=HWAddress
  to_mv(signals.va_addr, 0x40).cast("Q")[3] = evt.event_trigger_data # HWData3=HWData
  print(hex(ds[0]), hex(ds[1]), hex(ds[2]))
  hexdump(to_mv(signals.va_addr, 0x40))

  # 10 08 49 3E 46 77 00 00


  # ring doorbell
  print(hex(to_mv(doorbell, 0x10).cast("I")[0]))
  #to_mv(doorbell, 0x10).cast("I")[0] = 0xffffffff
  to_mv(doorbell, 0x10).cast("I")[0] = 0

  evt_arr = (kfd.struct_kfd_event_data * 1)()
  evt_arr[0].event_id = evt.event_id
  kio.wait_events(fd, events_ptr=ctypes.addressof(evt_arr), num_events=1, wait_for_all=0, timeout=1000)

  print(hex(ds[0]), hex(ds[1]), hex(ds[2]))
  hexdump(to_mv(signals.va_addr, 0x40))

  #nq = kio.create_queue(fd, ring_base_address=buf, ring_size=0x1000, gpu_id=GPU_ID,
  #                      queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE,
  #                      queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY, write_pointer_address=buf+8, read_pointer_address=buf+0x10)
  #print(nq)

  #mv = to_mv(buf, 0x1000)
  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, -1, 0)

  #print('\n'.join(format_struct(ver)))
  #print('\n'.join(format_struct(st)))
