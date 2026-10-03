from __future__ import annotations
import os, mmap, array, functools, contextlib, itertools, struct, socket, subprocess, time, enum, atexit
from tinygrad.helpers import getenv, temp, ceildiv, unwrap, fetch, system, _ensure_downloads_dir, DEBUG, flatten
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.support.system import PCIDevice, System

class RemoteCmd(enum.IntEnum):
  PROBE,MAP_BAR,MAP_SYSMEM_FD,CFG_READ,CFG_WRITE,RESET,MMIO_READ,MMIO_WRITE,MAP_SYSMEM,SYSMEM_READ,SYSMEM_WRITE,RESIZE_BAR,PING = range(13)

class RemoteMMIOInterface(MMIOInterface):
  def __init__(self, dev:RemotePCIDevice, residx:int, nbytes:int, fmt='B', off=0, rd_cmd=RemoteCmd.MMIO_READ, wr_cmd=RemoteCmd.MMIO_WRITE):
    self.dev, self.residx, self.nbytes, self.fmt, self.off, self.el_sz = dev, residx, nbytes, fmt, off, struct.calcsize(fmt)
    self.rd_cmd, self.wr_cmd = rd_cmd, wr_cmd

  def __getitem__(self, index):
    sl = index if isinstance(index, slice) else slice(index, index + 1)
    start, stop = (sl.start or 0) * self.el_sz, (sl.stop or len(self)) * self.el_sz
    data = self.dev._bulk_read(self.rd_cmd, self.residx, self.off + start, stop - start)
    result = data if self.fmt == 'B' else list(struct.unpack(f'<{(stop - start) // self.el_sz}{self.fmt}', data))
    return result if isinstance(index, slice) else result[0]

  def __setitem__(self, index, val):
    start = (index.start or 0) * self.el_sz if isinstance(index, slice) else index * self.el_sz
    data = (val if self.fmt == 'B' else struct.pack(f'<{len(val)}{self.fmt}', *val)) if isinstance(index, slice) else struct.pack(f'<{self.fmt}', val)
    self.dev._bulk_write(self.wr_cmd, self.residx, self.off + start, data)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return RemoteMMIOInterface(self.dev, self.residx, size or (self.nbytes - offset), fmt or self.fmt, self.off + offset, self.rd_cmd, self.wr_cmd)

class RemotePCIDevice(PCIDevice):
  _bulk_sent:int = 0
  _bulk_recv:int = 0
  _rpc_count:int = 0
  _start_time:float = 0.0

  @staticmethod
  @functools.cache
  def remote_sock(host:str, port:int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(getenv("REMOTE_TIMEOUT", 3))
    sock.connect((host, port))
    sock.settimeout(None)
    if DEBUG >= 1 and RemotePCIDevice._start_time == 0.0:
      RemotePCIDevice._start_time = time.perf_counter()
      def _print_stats():
        dt = time.perf_counter() - RemotePCIDevice._start_time
        sent_mb, recv_mb = RemotePCIDevice._bulk_sent / 1e6, RemotePCIDevice._bulk_recv / 1e6
        print(f"remote: sent {sent_mb:,.2f} MB ({sent_mb/dt:,.2f} MB/s), recv {recv_mb:,.2f} MB ({recv_mb/dt:,.2f} MB/s), "
              f"{RemotePCIDevice._rpc_count:,} roundtrips in {dt:.2f}s")
      atexit.register(_print_stats)
    return sock

  @staticmethod
  @functools.cache
  def remote_list(vendor:int, devices:tuple[tuple[int, tuple[int, ...]], ...], base_class:int|None) -> list[tuple[socket.socket, str]]:
    payload = array.array('I', itertools.chain.from_iterable((m, d) for m, ds in devices for d in ds)).tobytes()
    def q(r:str) -> list[tuple[socket.socket, str]]:
      sock = RemotePCIDevice.remote_sock((host:=r.strip().split(":")[0]), (port:=int(r.strip().split(":")[1]) if ":" in r else 6667))
      data_len, _, _, _ = RemotePCIDevice._rpc(sock, 0, RemoteCmd.PROBE, base_class or 0, len(payload), vendor, payload=payload)
      return [(sock, f"remote:{host}:{port}:{d}") for d in RemotePCIDevice._recvall(sock, data_len).decode().split('\n')]
    return flatten([q(r) for r in getenv("REMOTE", "").split(",") if r.strip()])

  @staticmethod
  def _recvall(sock:socket.socket, n:int) -> bytes:
    data = b''
    while len(data) < n and (chunk:=sock.recv(n - len(data))): data += chunk
    if len(data) < n: raise RuntimeError("Connection closed")
    return data

  @staticmethod
  def _rpc(sock:socket.socket, dev_id:int, cmd:int, *args:int, bar:int=0, readout_size:int=0, payload:bytes=b'', has_fd=False):
    sock.sendall(struct.pack('<BIIQQQ', cmd, dev_id, bar, *(*args, 0, 0, 0)[:3]) + payload)
    if has_fd:
      msg, anc, _, _ = sock.recvmsg(17, socket.CMSG_LEN(4))
      fd = struct.unpack('<i', anc[0][2][:4])[0]
    else: msg, fd = RemotePCIDevice._recvall(sock, 17), None
    if (resp:=struct.unpack('<BQQ', msg))[0] != 0:
      raise RuntimeError(f"RPC failed: {RemotePCIDevice._recvall(sock, resp[1]).decode('utf-8') if resp[1] > 0 else 'unknown error'}")
    RemotePCIDevice._rpc_count += 1
    return (resp[1], resp[2]) + ((RemotePCIDevice._recvall(sock, readout_size) if readout_size > 0 else None),) + (fd,)

  def __init__(self, devpref:str, pcibus:str, sock:socket.socket):
    self.sock, self.pcibus, self.dev_id = sock, pcibus, int(pcibus.split(':')[-1]) if ':' in pcibus else 0
    self.peer_group = sock.getpeername()[0]
    for buft in [socket.SO_SNDBUF, socket.SO_RCVBUF]: self.sock.setsockopt(socket.SOL_SOCKET, buft, 64 << 20)

    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")

  def _bulk_read(self, cmd:int, idx:int, offset:int, size:int) -> bytes:
    RemotePCIDevice._bulk_recv += size
    return unwrap(self._rpc(self.sock, self.dev_id, cmd, offset, size, bar=idx, readout_size=size)[2])
  def _bulk_write(self, cmd:int, idx:int, offset:int, data:bytes):
    RemotePCIDevice._bulk_sent += len(data)
    self.sock.sendall(struct.pack('<BIIQQQ', cmd, self.dev_id, idx, offset, len(data), 0) + data)

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False) -> tuple[MMIOInterface, list[int]]:
    paddrs_len, handle, _, _ = self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_SYSMEM, size, int(contiguous))
    paddrs = list(struct.unpack(f'<{paddrs_len // 8}Q', self._recvall(self.sock, paddrs_len)))
    return RemoteMMIOInterface(self, handle, size, fmt='B', rd_cmd=RemoteCmd.SYSMEM_READ, wr_cmd=RemoteCmd.SYSMEM_WRITE), paddrs

  def reset(self): self._rpc(self.sock, self.dev_id, RemoteCmd.RESET)
  def read_config(self, offset:int, size:int): return self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_READ, offset, size)[0]
  def write_config(self, offset:int, value:int, size:int): self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_WRITE, offset, size, value)

  @functools.cache
  def bar_info(self, bar_idx:int) -> tuple[int, int]: return self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_BAR, bar=bar_idx)[:2]
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    return RemoteMMIOInterface(self, bar, size or self.bar_info(bar)[1], fmt).view(off, size, fmt)
  def resize_bar(self, bar_idx:int): self._rpc(self.sock, self.dev_id, RemoteCmd.RESIZE_BAR, bar=bar_idx)

class APLRemotePCIDevice(RemotePCIDevice):
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"

  @classmethod
  def ensure_app(cls):
    commit = "c0d024f9ff0e1dc8fdf217f255da7101d91e8323"
    app_name = f"TinyGPU_{commit}.zip"
    if (_ensure_downloads_dir() / app_name).is_file() and os.path.exists(cls.APP_PATH): return
    print("Downloading TinyGPU.app...")
    with contextlib.suppress(RuntimeError): system("pkill -f TinyGPU")
    system(f"ditto -xk {fetch(f'https://github.com/tinygrad/tinygpu_releases/raw/{commit}/TinyGPU.zip', name=app_name)} /Applications")
    print(system(f"{cls.APP_PATH} install"))

  def __init__(self, devpref:str, pcibus:str):
    self.ensure_app()
    sock_path, sock = getenv("APL_REMOTE_SOCK", temp("tinygpu.sock")), socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for i in range(100):
      with contextlib.suppress(ConnectionRefusedError, FileNotFoundError):
        sock.connect(sock_path)
        break
      if i == 0: subprocess.Popen([self.APP_PATH, "server", sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      time.sleep(0.05)
    else: raise RuntimeError(f"Failed to connect to TinyGPU server at {sock_path}.")
    super().__init__(devpref, "usb4", sock=sock)

  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False) -> tuple[MMIOInterface, list[int]]:
    mapped_size, _, _, fd = self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)
    memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), mapped_size, fmt='B')

    # paddrs are returned as (paddr, size) pairs until a (paddr=0, size=0) terminator in the beginning of the mapping.
    paddrs_raw = list(itertools.takewhile(lambda p: p[1] != 0, zip(memview.view(fmt='Q')[0::2], memview.view(fmt='Q')[1::2])))
    return memview, [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]
