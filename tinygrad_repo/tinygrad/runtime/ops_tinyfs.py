import socket, json, asyncio, threading, math
from contextlib import asynccontextmanager
from tinygrad.device import Compiled, Allocator
from tinygrad.helpers import DEBUG, getenv
from tinygrad import Tensor

TINYFS_ENDPOINT = getenv("TINYFS_ENDPOINT", "localhost:6767")
TINYFS_TIMEOUT = getenv("TINYFS_TIMEOUT", 60)

class TinyFSDevice(Compiled):
  def __init__(self, device:str):
    self.op = device[len("tinyfs:"):].upper()
    super().__init__(device, TinyFSAllocator(self), None, None, None)

    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect((TINYFS_ENDPOINT.rsplit(":", 1)[0], int(TINYFS_ENDPOINT.rsplit(":", 1)[1])))
    self.sock.settimeout(TINYFS_TIMEOUT)
    self.sfile = self.sock.makefile("rwb")

    # fetch node info
    self.sfile.write(b"INFO\r\n")
    self.sfile.flush()
    info = self.sfile.readline()
    self.node_info = json.loads(info)
    if DEBUG >= 2: print(f"nodes: {self.node_info}")

    # spawn thread for async copyout
    self.start_event = threading.Event()
    self.t = threading.Thread(target=self._start_thread, daemon=True)
    self.t.start()
    self.start_event.wait()

    # connection pools
    self.conn_pools: dict[str, asyncio.Queue] = {}
    self.conn_pools_lock = asyncio.Lock()

  def finalize(self):
    self.sfile.close()

    for pool in self.conn_pools.values():
      while not pool.empty():
        _, w = pool.get_nowait()
        w.close()
        asyncio.run_coroutine_threadsafe(w.wait_closed(), self.loop).result()

    if hasattr(self, "loop"):
      self.loop.call_soon_threadsafe(self.loop.stop)
    self.t.join()

  def _start_thread(self):
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

    self.start_event.set()
    self.loop.run_forever()
    self.loop.close()

  @asynccontextmanager
  async def connection(self, loc):
    if loc not in self.conn_pools:
      await self.conn_pools_lock.acquire()
      if loc not in self.conn_pools:
        self.conn_pools[loc] = asyncio.Queue(nw:=getenv("ASYNC_COPY_WORKERS", 4))
        conn_tasks = [asyncio.open_connection(*self.node_info[loc][-1].rsplit(":", 1)) for _ in range(nw)]
        connections = await asyncio.gather(*conn_tasks)
        for reader, writer in connections: self.conn_pools[loc].put_nowait((reader, writer))
      self.conn_pools_lock.release()

    reader, writer = await self.conn_pools[loc].get()
    try:
      yield reader, writer
    finally:
      await self.conn_pools[loc].put((reader, writer))

class TinyFSBuffer:
  def __init__(self, device:TinyFSDevice, size:int, offset=0, copyout_queue=None, hash_buf=None):
    self.device, self.size, self.offset = device, size, offset
    self.copyout_queue = copyout_queue or []
    self.hash_buf = hash_buf or bytearray()
  def __repr__(self): return f"<TinyFSBuffer size={self.size} offset={self.offset}>"

class TinyFSAllocator(Allocator[TinyFSDevice]):
  def _alloc(self, size, options):
    return TinyFSBuffer(self.dev, size)

  def _copyin(self, dest:TinyFSBuffer, src:memoryview):
    if DEBUG >= 2: print(f"Copying in {dest.size} bytes to TINYFS:{dest.device.op}")
    self.dev.sfile.write(f"{dest.device.op}_IN {dest.size}\r\n".encode())

    self.dev.sfile.write(src)
    self.dev.sfile.flush()

    if dest.device.op == "LOAD":
      locs = self.dev.sfile.readline()
      dest.copyout_queue = json.loads(locs)
      dest.hash_buf = src.tobytes()
    elif dest.device.op == "STORE":
      expected_hashes = math.ceil(dest.size / Tensor.CHUNK_SIZE)
      dest.hash_buf = bytearray(expected_hashes * 16)
      self.dev.sfile.readinto(dest.hash_buf)

  def _copyout(self, dest:memoryview, src:TinyFSBuffer):
    if DEBUG >= 2: print(f"Copying out {src.size} bytes from TINYFS:{src.device.op}")
    if src.device.op == "LOAD":
      asyncio.run_coroutine_threadsafe(self._copyout_async(dest, src), src.device.loop).result()
    elif src.device.op == "STORE":
      dest[:] = src.hash_buf

  async def _copyout_async(self, dest:memoryview, src:TinyFSBuffer):
    async def _worker(i, loc):
      async with self.dev.connection(loc) as (reader, writer):
        ptr = i * Tensor.CHUNK_SIZE
        size = min(len(dest[ptr:ptr+Tensor.CHUNK_SIZE]), Tensor.CHUNK_SIZE)

        writer.write(f"CHUNK_OUT {size}\r\n".encode())
        writer.write(src.hash_buf[i*16:(i+1)*16])
        await asyncio.wait_for(writer.drain(), timeout=TINYFS_TIMEOUT)

        chunk = await asyncio.wait_for(reader.readexactly(size), timeout=TINYFS_TIMEOUT)

        view = dest[ptr:ptr+len(chunk)]
        view[:] = chunk
        del view

    workers = [asyncio.create_task(_worker(i, loc)) for i, loc in enumerate(src.copyout_queue)]
    await asyncio.gather(*workers)

  def _offset(self, buf:TinyFSBuffer, size:int, offset:int):
    return TinyFSBuffer(buf.device, size, offset, buf.copyout_queue, buf.hash_buf)
