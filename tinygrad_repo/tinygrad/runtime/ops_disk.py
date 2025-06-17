import os, sys, mmap, io, ctypes, ctypes.util, contextlib
from typing import Optional, Generator, Callable
from tinygrad.helpers import OSX, round_up
from tinygrad.device import Compiled, Allocator
with contextlib.suppress(ImportError):
  import _posixshmem
  from tinygrad.runtime.autogen import io_uring, libc

class DiskDevice(Compiled):
  _tried_io_uring_init = False

  def __init__(self, device:str):
    if not DiskDevice._tried_io_uring_init: self._iouring_setup()

    self.size: Optional[int] = None
    self.fd: Optional[int] = None
    self.count = 0
    super().__init__(device, DiskAllocator(self), None, None, None)
  def _might_open(self, size):
    self.count += 1
    assert self.size is None or size <= self.size, f"can't reopen Disk tensor with larger size, opened with {self.size}, tried to open with {size}"
    if self.size is not None: return
    filename = self.device[len("disk:"):]
    self.size = size

    if sys.platform != "win32" and filename.startswith("shm:"):
      fd = _posixshmem.shm_open("/"+filename[4:].lstrip("/"), os.O_RDWR, 0o600)
      self.mem = mmap.mmap(fd, self.size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
    else:
      try: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT|getattr(os, "O_DIRECT", 0))
      except OSError: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT)
      if os.fstat(self.fd).st_size < self.size: os.ftruncate(self.fd, self.size)
      self.mem = mmap.mmap(self.fd, self.size)
    if hasattr(self.mem, 'madvise') and (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None:
      with contextlib.suppress(OSError): self.mem.madvise(hp) # some systems have transparent_hugepage disabled
  def _might_close(self):
    self.count -= 1
    if self.count == 0:
      if self.fd is not None: os.close(self.fd)
      self.size = None
  def _iouring_setup(self):
    DiskDevice._tried_io_uring_init = True

    if sys.platform == 'linux' and not hasattr(sys, "getandroidapilevel"):
      fd = libc.syscall(io_uring.NR_io_uring_setup, 4096, ctypes.byref(p:=io_uring.struct_io_uring_params()))
      if fd < 0: return

      sq_ptr = libc.mmap(0, p.sq_off.array + p.sq_entries * 4, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_POPULATE, fd, 0)
      cq_ptr = libc.mmap(0, p.cq_off.cqes + p.cq_entries * ctypes.sizeof(io_uring.struct_io_uring_cqe),
                        mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_POPULATE, fd, io_uring.IORING_OFF_CQ_RING)
      sqes = libc.mmap(0, p.sq_entries * ctypes.sizeof(io_uring.struct_io_uring_sqe),
                      mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_POPULATE, fd, io_uring.IORING_OFF_SQES)

      def u32ptr(val): return ctypes.cast(val, ctypes.POINTER(ctypes.c_uint32))
      sqdesc = io_uring.struct_io_uring_sq(khead=u32ptr(sq_ptr+p.sq_off.head), ktail=u32ptr(sq_ptr+p.sq_off.tail),
                                           array=u32ptr(sq_ptr+p.sq_off.array),
        kring_mask=u32ptr(sq_ptr+p.sq_off.ring_mask), sqes=ctypes.cast(sqes, ctypes.POINTER(io_uring.struct_io_uring_sqe)))

      cqdesc = io_uring.struct_io_uring_cq(khead=u32ptr(cq_ptr+p.cq_off.head), ktail=u32ptr(cq_ptr+p.cq_off.tail),
        kring_mask=u32ptr(sq_ptr+p.cq_off.ring_mask), cqes=ctypes.cast(cq_ptr+p.cq_off.cqes, ctypes.POINTER(io_uring.struct_io_uring_cqe)))

      DiskDevice.io_uring = io_uring.struct_io_uring(ring_fd=fd, sq=sqdesc, cq=cqdesc) # type: ignore

class DiskBuffer:
  def __init__(self, device:DiskDevice, size:int, offset=0):
    self.device, self.size, self.offset = device, size, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} offset={self.offset}>"
  def _buf(self) -> memoryview:
    assert hasattr(self.device, "mem"), f"DiskBuffer wasn't opened: {self.device.device}"
    return memoryview(self.device.mem)[self.offset:self.offset+self.size]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  def __init__(self, dev:DiskDevice): self.dev = dev
  def _alloc(self, size:int, options):
    self.dev._might_open(size)
    return DiskBuffer(self.dev, size)
  def _free(self, opaque, options): self.dev._might_close()
  def _as_buffer(self, src:DiskBuffer): return src._buf()
  def _copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def _copyout(self, dest:memoryview, src:DiskBuffer):
    if OSX and self.dev.fd is not None:
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(self.dev.fd, "a+b", closefd=False) as fo:
        fo.seek(src.offset)
        bytes_read = 0
        while (n := fo.readinto(dest[bytes_read:])) is not None and n > 0: bytes_read += n
    else:
      dest[:] = src._buf()

  def _copyout_sharded(self, src:DiskBuffer, size:int, _get_free_buf:Callable, seg_len:int) -> Generator[tuple[int, int, int, int], None, None]:
    assert hasattr(DiskDevice, 'io_uring'), "function requires io uring support"

    fd_offset = src.offset - (minor_offset := src.offset % mmap.PAGESIZE)
    processed_reqs_cnt, copied_in, next_read_offset, total_copy_size = 0, 0, 0, round_up(size + minor_offset, mmap.PAGESIZE)
    reqs: list[tuple[int, int, int, int]] = []

    while next_read_offset < total_copy_size or len(reqs) != processed_reqs_cnt:
      if next_read_offset < total_copy_size and (copy_batch := _get_free_buf()) is not None:
        # Prepare sqe
        sqe_index = (tail:=DiskDevice.io_uring.sq.ktail[0]) & DiskDevice.io_uring.sq.kring_mask[0]
        sqe = DiskDevice.io_uring.sq.sqes[sqe_index]
        sqe.opcode, sqe.fd, sqe.off = io_uring.IORING_OP_READ, self.dev.fd, fd_offset + next_read_offset
        sqe.addr, sqe.len, sqe.user_data = copy_batch[0], min(seg_len, total_copy_size - next_read_offset), len(reqs)

        # Send sqe
        DiskDevice.io_uring.sq.array[sqe_index] = sqe_index
        DiskDevice.io_uring.sq.ktail[0] = tail + 1
        libc.syscall(io_uring.NR_io_uring_enter, DiskDevice.io_uring.ring_fd, 1, 1, io_uring.IORING_ENTER_GETEVENTS)

        reqs.append((copy_batch, copied_in, minor_offset, real_copy_size:=min(sqe.len - minor_offset, size - copied_in)))
        next_read_offset += sqe.len
        copied_in += real_copy_size
        minor_offset = 0

      if (head:=DiskDevice.io_uring.cq.khead[0]) != DiskDevice.io_uring.cq.ktail[0]:
        cqe = DiskDevice.io_uring.cq.cqes[head & DiskDevice.io_uring.cq.kring_mask[0]]
        assert cqe.res >= 0, f"read from disk failed, err: {cqe.res}"
        yield reqs[cqe.user_data]
        DiskDevice.io_uring.cq.khead[0] = head + 1 # advance
        processed_reqs_cnt += 1

  def _offset(self, buf:DiskBuffer, size:int, offset:int): return DiskBuffer(buf.device, size, offset)
