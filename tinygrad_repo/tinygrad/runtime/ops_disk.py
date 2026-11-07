import os, sys, mmap, io, ctypes, contextlib, pathlib, functools, collections, struct, itertools, errno
from dataclasses import replace
from typing import cast
from tinygrad.helpers import OSX, mv_address, flatten, to_tuple, unwrap, ceildiv
from tinygrad.device import BufferStorage, MMIOInterface, Compiled, Allocator, Buffer, BufferSpec, Device, HCQ_RUNTIME_DEV
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, uopfunc
from tinygrad.runtime.support.hcq2 import ccall, ins, patch
with contextlib.suppress(ImportError):
  import _posixshmem
  from tinygrad.runtime.autogen import io_uring, libc

class DiskDevice(Compiled):
  def synchronize(self, timeout:int|None=None): pass

  def __init__(self, device:str):
    self.size: int|None = None
    self.fd: int|None = None
    self.refcount, self.info = 0, Buffer(HCQ_RUNTIME_DEV.device, 16, options=BufferSpec(nolru=True)) # info: [fd, mmap address] for batch reads
    super().__init__(device, DiskAllocator(self), [], None)
  def _might_open(self, size:int):
    assert self.size is None or size <= self.size, f"can't reopen Disk tensor with larger size, opened with {self.size}, tried to open with {size}"
    if self.size is not None and hasattr(self, "mem"):
      self.refcount += 1
      return
    filename = self.device[len("disk:"):]

    if filename.startswith("shm:"):
      fd = _posixshmem.shm_open("/"+filename[4:].lstrip("/"), os.O_RDWR, 0o600)
      self.mem = mmap.mmap(fd, size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
    else:
      try: self.fd, rdonly = os.open(filename, os.O_RDWR|os.O_CREAT|getattr(os, "O_DIRECT", 0)), False
      except OSError as e: self.fd = os.open(filename, os.O_RDONLY if (rdonly:=e.errno in (errno.EROFS, errno.EACCES)) else os.O_RDWR|os.O_CREAT)
      if not pathlib.Path(filename).is_block_device() and os.fstat(self.fd).st_size < size: os.ftruncate(self.fd, size)
      self.mem = mmap.mmap(self.fd, size, access=mmap.ACCESS_COPY if rdonly else mmap.ACCESS_DEFAULT)
      self.info.host.view()[:16] = struct.pack("2Q", self.fd, mv_address(memoryview(self.mem)))
    self.size = size
    if hasattr(self.mem, 'madvise') and (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None:
      with contextlib.suppress(OSError): self.mem.madvise(hp) # some systems have transparent_hugepage disabled
    self.refcount += 1
  def _might_close(self):
    self.refcount -= 1
    if self.refcount == 0:
      if self.fd is not None: os.close(self.fd)
      if hasattr(self, "mem"):
        try: self.mem.close()
        except BufferError: pass
      self.size = None

class DiskBuffer:
  def __init__(self, device:DiskDevice, size:int, offset=0):
    self.device, self.size, self.offset = device, size, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} offset={self.offset}>"
  def _buf(self) -> memoryview:
    assert hasattr(self.device, "mem"), f"DiskBuffer wasn't opened: {self.device.device}"
    return memoryview(self.device.mem)[self.offset:self.offset+self.size]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  lru = False
  def _alloc(self, size:int, options) -> BufferStorage:
    self.dev._might_open(size)
    return BufferStorage(opaque:=DiskBuffer(self.dev, size), None, MMIOInterface(mv_address(opaque._buf()), size))

  def _free(self, storage:BufferStorage, options): self.dev._might_close()
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

  def _offset(self, buf:DiskBuffer, size:int, offset:int): return DiskBuffer(buf.device, size, offset)

# *****************
# UOps implementation

CHUNK_SZ, READ_SZ, SLOTS, RING_ENTRIES = 64 << 20, 1 << 20, 2, 256
SLOT_SZ, STAGE_SZ = CHUNK_SZ + 4096, 4096 + SLOTS * (CHUNK_SZ + 4096) # +4096: reads are page aligned. stage: a page of counts, then the slots

@uopfunc
def disk_read(file:UOp, srcs:UOp, counts:UOp, rings:UOp, sqes:UOp) -> UOp: # srcs: [address in the file's mmap, bytes] per copy
  fd, p = unwrap(uring())[:2]
  file, srcs, counts, rings, sqes = (b.replace(arg=replace(b.arg, volatile=True, device=None)) for b in (file, srcs, counts, rings, sqes))

  # for each copy
  src_va, nbytes = srcs[(copy:=UOp.range(srcs.max_numel() // 2, 0, dtype=dtypes.uint64)) * 2], srcs[copy * 2 + 1]

  # for each chunk: the next chunk n waits for the gpu to free its slot
  n = counts.after(chunk:=UOp.range(ceildiv(nbytes, CHUNK_SZ), 1, dtype=dtypes.uint64))[0]
  free = (copied:=counts.after(n, loop:=UOp.loop(3))[1]).backedge(loop, copied + SLOTS <= n)
  pos = src_va + chunk * CHUNK_SZ - file[1]
  span = (pos % 4096 + (nbytes - chunk * CHUNK_SZ).minimum(CHUNK_SZ) + 4095) // 4096 * 4096

  # for each read: queues READ_SZ of the chunk's pages. an sqe is 8 words: fd << 32 | flags << 8 | opcode, file offset, address, bytes
  tail, read = rings.after(free)[p.sq_off.tail // 4].cast(dtypes.uint64), UOp.range(nreads:=ceildiv(span, READ_SZ), 2, dtype=dtypes.uint64)
  sqe = [file[0] << 32 | io_uring.IOSQE_ASYNC << 8 | io_uring.IORING_OP_READ, pos - pos % 4096 + read * READ_SZ,
         counts.getaddr(HCQ_RUNTIME_DEV.device) + 4096 + n % SLOTS * SLOT_SZ + read * READ_SZ, (span - read * READ_SZ).minimum(READ_SZ)]
  queued = UOp.group(*[sqes[(tail + read) % RING_ENTRIES * 8 + j].store(v) for j, v in enumerate(sqe)]).end(read)

  # submits the reads, waits for all, marks the chunk read
  to_submit = nreads.cast(dtypes.int).after(rings.after(queued)[p.sq_off.tail // 4].store((tail + nreads).cast(dtypes.uint32)))
  done = ccall(libc.syscall, io_uring.NR_io_uring_enter, fd, to_submit, to_submit, io_uring.IORING_ENTER_GETEVENTS, 0, 0)
  reaped = rings.after(done)[p.cq_off.head // 4].store(rings.after(done)[p.cq_off.tail // 4])
  return counts.after(reaped)[0].store(n + 1).end(chunk).end(copy).sink()

# *****************
# 2. rewriter

def is_disk_read(c:UOp) -> bool:
  dev, src = [to_tuple(b.device)[0] for b in c.src[1:3]] if c.op is Ops.CALL and c.body.op is Ops.STORE else ("", "")
  return src.startswith("DISK:") and not src.startswith("DISK:shm:") and Device[dev].has_copy_queue and not dev.startswith("NULL") and bool(uring()) \
    and not getattr(Device[dev], "is_usb", False)

def disk_copy_rewriter(s:UOp) -> UOp|None:
  lin = s.without_after.body
  if not (copies:=[c for c in lin.src if is_disk_read(c)]): return None
  rings = UOp.alloc((unwrap(uring())[2].nbytes // 4,), dtypes.uint32, 0, device=HCQ_RUNTIME_DEV.device, name="uring_rings")
  sqes = UOp.alloc((8 * RING_ENTRIES,), dtypes.uint64, 0, device=HCQ_RUNTIME_DEV.device, name="uring_sqes")

  # [mmap address, bytes] per copy in a table: an arg each would hit ctypes' 1024 limit
  words = flatten((c.src[2].getaddr(HCQ_RUNTIME_DEV.device), UOp.const(c.src[2].nbytes(), dtypes.uint64)) for c in copies)
  table = UOp.alloc((8 * len(words),), dtypes.uint8, device=HCQ_RUNTIME_DEV.device)
  srcs = patch(table, [(8 * i, w) for i, w in enumerate(words)]).bitcast(dtypes.uint64)

  # [chunks read, chunks copied] at the start of the gpu's stage
  dev = s.without_after.src[1].device
  counts = UOp.alloc((STAGE_SZ,), dtypes.uint8, 0, device=dev, name="disk_stage")[:16].bitcast(dtypes.uint64)

  # cpu part
  done, off, gpu_ops = s, 0, collections.defaultdict[UOp, list[UOp]](list)
  for disk, grp in itertools.groupby(range(len(copies)), lambda k: to_tuple(copies[k].src[2].device)[0]):
    file, ks = UOp.alloc((2,), dtypes.uint64, 0, device=disk, name="disk_info"), list(grp)
    done = disk_read(file, srcs[2 * ks[0]:2 * ks[-1] + 2], counts.after(done), rings, sqes)

  # gpu part
  for k, c in enumerate(copies):
    first, slots = counts.index(0).load() + off, counts.getaddr(dev) + 4096 + srcs.index(2 * k).load() % 4096
    off += ceildiv(c.src[2].nbytes(), CHUNK_SZ)
    for chunk in range(ceildiv(c.src[2].nbytes(), CHUNK_SZ)):
      n, dst, nb = first + chunk, c.src[1].getaddr() + chunk * CHUNK_SZ, min(CHUNK_SZ, c.src[2].nbytes() - chunk * CHUNK_SZ)
      gpu_ops[c] += [ins("wait", counts, n + 1), ins("copy", dst, slots + n % SLOTS * SLOT_SZ, nb), ins("store", counts[1:], n + 1)]
  return done.substitute({lin: lin.replace(src=tuple(flatten(gpu_ops.get(c, [c]) for c in lin.src)))}, enter_calls=True)
Compiled.pm_encode = PatternMatcher([(UPat.sink().call(UPat(Ops.PARAM)).after(allow_any_len=True, name="s"),
  disk_copy_rewriter)]) + Compiled.pm_encode

# *****************
# 3. bufferize

def shared(fd:int, at:int, n:int) -> Buffer: # n bytes of memory the kernel shares at offset at
  return Buffer(HCQ_RUNTIME_DEV.device, n, options=BufferSpec(external_ptr=libc.mmap(0, n, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, fd, at)))

@functools.cache
def uring() -> tuple[int, io_uring.struct_io_uring_params, Buffer, Buffer]|None: # fd, offsets, the rings (heads, tails, cqes), the sqes
  if sys.platform != "linux" or hasattr(sys, "getandroidapilevel"): return None
  p = io_uring.struct_io_uring_params(flags=io_uring.IORING_SETUP_NO_SQARRAY)  # type: ignore[unreachable]
  if (fd:=libc.syscall(io_uring.NR_io_uring_setup, RING_ENTRIES, ctypes.byref(p))) < 0: return None
  return fd, p, shared(fd, io_uring.IORING_OFF_SQ_RING, p.cq_off.cqes + 16 * p.cq_entries), shared(fd, io_uring.IORING_OFF_SQES, 64 * RING_ENTRIES)

@functools.cache
def _stage(dev) -> Buffer: return Buffer(Device[dev].host, STAGE_SZ, options=BufferSpec(cpu_access=True), initial_value=bytes(STAGE_SZ))

Compiled.link_bufs |= {"disk_info": lambda b: cast(DiskDevice, Device[b.device]).info, "disk_stage": lambda b: _stage(b.device),
                       "uring_rings": lambda b: unwrap(uring())[2], "uring_sqes": lambda b: unwrap(uring())[3]}
