# the CLOUD=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with CLOUD tinygrad is  frontend <-> middleware <-> CloudDevice ///HTTP/// cloud_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import multiprocessing, functools, http.client, hashlib, json, time, os, binascii, struct, ast, contextlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from tinygrad.renderer import Renderer
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, DEBUG, fromimport, unwrap, Timing
from tinygrad.device import Compiled, Allocator, Compiler, Device, BufferSpec

# ***** API *****

class CloudRequest: pass

@dataclass(frozen=True)
class BufferAlloc(CloudRequest): buffer_num: int; size: int; options: BufferSpec # noqa: E702

@dataclass(frozen=True)
class BufferFree(CloudRequest): buffer_num: int # noqa: E702

@dataclass(frozen=True)
class CopyIn(CloudRequest): buffer_num: int; datahash: str # noqa: E702

@dataclass(frozen=True)
class CopyOut(CloudRequest): buffer_num: int

@dataclass(frozen=True)
class ProgramAlloc(CloudRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramFree(CloudRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramExec(CloudRequest):
  name: str; datahash: str; bufs: tuple[int, ...]; vals: tuple[int, ...] # noqa: E702
  global_size: Optional[tuple[int, ...]]; local_size: Optional[tuple[int, ...]]; wait: bool # noqa: E702

# for safe deserialization
whitelist = {x.__name__:x for x in [BufferAlloc, BufferFree, CopyIn, CopyOut, ProgramAlloc, ProgramFree, ProgramExec, BufferSpec]}
eval_fxns = {ast.Constant: lambda x: x.value, ast.Tuple: lambda x: tuple(map(safe_eval, x.elts)), ast.List: lambda x: list(map(safe_eval, x.elts)),
  ast.Call: lambda x: safe_eval(x.func)(*[safe_eval(arg) for arg in x.args], **{kwarg.arg: safe_eval(kwarg.value) for kwarg in x.keywords}),
  ast.Name: lambda x: whitelist[x.id], ast.Attribute: lambda x: {"imagef": dtypes.imagef, "imageh": dtypes.imageh}[x.attr]}
def safe_eval(node): return eval_fxns[node.__class__](node)

class BatchRequest:
  def __init__(self):
    self._q: list[CloudRequest] = []
    self._h: dict[str, bytes] = {}
  def h(self, d:bytes) -> str:
    binhash = hashlib.sha256(d).digest()
    self._h[datahash:=binascii.hexlify(binhash).decode()] = binhash+struct.pack("<Q", len(d))+d
    return datahash
  def q(self, x:CloudRequest): self._q.append(x)
  def serialize(self) -> bytes:
    self.h(repr(self._q).encode())
    return b''.join(self._h.values())
  def deserialize(self, dat:bytes) -> BatchRequest:
    ptr = 0
    while ptr < len(dat):
      datahash, datalen = binascii.hexlify(dat[ptr:ptr+0x20]).decode(), struct.unpack("<Q", dat[ptr+0x20:ptr+0x28])[0]
      self._h[datahash] = dat[ptr+0x28:ptr+0x28+datalen]
      ptr += 0x28+datalen
    self._q = safe_eval(ast.parse(self._h[datahash], mode="eval").body)
    return self

# ***** backend *****

@dataclass
class CloudSession:
  programs: dict[tuple[str, str], Any] = field(default_factory=dict)
  # TODO: the buffer should track this internally
  buffers: dict[int, tuple[Any, int, Optional[BufferSpec]]] = field(default_factory=dict)

class CloudHandler(BaseHTTPRequestHandler):
  protocol_version = 'HTTP/1.1'
  device: str
  sessions: defaultdict[str, CloudSession] = defaultdict(CloudSession)

  def setup(self):
    super().setup()
    print(f"connection established with {self.client_address}, socket: {self.connection.fileno()}")

  def _do(self, method):
    session = CloudHandler.sessions[unwrap(self.headers.get("Cookie")).split("session=")[1]]
    ret, status_code = b"", 200
    if self.path == "/batch" and method == "POST":
      # TODO: streaming deserialize?
      req = BatchRequest().deserialize(self.rfile.read(int(unwrap(self.headers.get('Content-Length')))))
      # the cmds are always last (currently in datahash)
      for c in req._q:
        if DEBUG >= 1: print(c)
        match c:
          case BufferAlloc():
            assert c.buffer_num not in session.buffers, f"buffer {c.buffer_num} already allocated"
            session.buffers[c.buffer_num] = (Device[CloudHandler.device].allocator.alloc(c.size, c.options), c.size, c.options)
          case BufferFree():
            buf,sz,buffer_options = session.buffers[c.buffer_num]
            Device[CloudHandler.device].allocator.free(buf,sz,buffer_options)
            del session.buffers[c.buffer_num]
          case CopyIn(): Device[CloudHandler.device].allocator._copyin(session.buffers[c.buffer_num][0], memoryview(bytearray(req._h[c.datahash])))
          case CopyOut():
            buf,sz,_ = session.buffers[c.buffer_num]
            Device[CloudHandler.device].allocator._copyout(memoryview(ret:=bytearray(sz)), buf)
          case ProgramAlloc():
            lib = Device[CloudHandler.device].compiler.compile_cached(req._h[c.datahash].decode())
            session.programs[(c.name, c.datahash)] = Device[CloudHandler.device].runtime(c.name, lib)
          case ProgramFree(): del session.programs[(c.name, c.datahash)]
          case ProgramExec():
            bufs = [session.buffers[x][0] for x in c.bufs]
            extra_args = {k:v for k,v in [("global_size", c.global_size), ("local_size", c.local_size)] if v is not None}
            r = session.programs[(c.name, c.datahash)](*bufs, vals=c.vals, wait=c.wait, **extra_args)
            if r is not None: ret = str(r).encode()
    elif self.path == "/renderer" and method == "GET":
      cls, args = Device[CloudHandler.device].renderer.__reduce__()
      ret = json.dumps((cls.__module__, cls.__name__, args)).encode()
    else: status_code = 404
    self.send_response(status_code)
    self.send_header('Content-Length', str(len(ret)))
    self.end_headers()
    return self.wfile.write(ret)

  def do_GET(self): return self._do("GET")
  def do_POST(self): return self._do("POST")

def cloud_server(port:int):
  multiprocessing.current_process().name = "MainProcess"
  CloudHandler.device = getenv("CLOUDDEV", "METAL") if Device.DEFAULT == "CLOUD" else Device.DEFAULT
  print(f"start cloud server on {port} with device {CloudHandler.device}")
  server = HTTPServer(('', port), CloudHandler)
  server.serve_forever()

# ***** frontend *****

class CloudAllocator(Allocator):
  def __init__(self, dev:CloudDevice):
    self.device = dev
    super().__init__()
  # TODO: ideally we shouldn't have to deal with images here
  def _alloc(self, size:int, options:BufferSpec) -> int:
    self.device.buffer_num += 1
    self.device.req.q(BufferAlloc(self.device.buffer_num, size, options))
    return self.device.buffer_num
  # TODO: options should not be here in any Allocator
  def _free(self, opaque:int, options): self.device.req.q(BufferFree(opaque))
  def _copyin(self, dest:int, src:memoryview): self.device.req.q(CopyIn(dest, self.device.req.h(bytes(src))))
  def _copyout(self, dest:memoryview, src:int):
    self.device.req.q(CopyOut(src))
    resp = self.device.batch_submit()
    assert len(resp) == len(dest), f"buffer length mismatch {len(resp)} != {len(dest)}"
    dest[:] = resp

class CloudProgram:
  def __init__(self, dev:CloudDevice, name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.datahash = self.dev.req.h(lib)
    self.dev.req.q(ProgramAlloc(self.name, self.datahash))
    super().__init__()
  def __del__(self): self.dev.req.q(ProgramFree(self.name, self.datahash))

  def __call__(self, *bufs, global_size=None, local_size=None, vals:tuple[int, ...]=(), wait=False):
    self.dev.req.q(ProgramExec(self.name, self.datahash, bufs, vals, global_size, local_size, wait))
    if wait: return float(self.dev.batch_submit())

class CloudDevice(Compiled):
  def __init__(self, device:str):
    if (host:=getenv("HOST", "")) != "": self.host = host
    else:
      p = multiprocessing.Process(target=cloud_server, args=(6667,))
      p.daemon = True
      p.start()
      self.host = "127.0.0.1:6667"

    # state for the connection
    self.session = binascii.hexlify(os.urandom(0x10)).decode()
    self.buffer_num = 0
    self.req: BatchRequest = BatchRequest()

    if DEBUG >= 1: print(f"cloud with host {self.host}")
    while 1:
      try:
        self.conn = http.client.HTTPConnection(self.host, timeout=60.0)
        clouddev = json.loads(self.send("GET", "renderer").decode())
        break
      except Exception as e:
        print(e)
        time.sleep(0.1)
    if DEBUG >= 1: print(f"remote has device {clouddev}")
    # TODO: how to we have BEAM be cached on the backend? this should just send a specification of the compute. rethink what goes in Renderer
    if not clouddev[0].startswith("tinygrad.renderer.") or not clouddev[1].endswith("Renderer"): raise RuntimeError(f"bad renderer {clouddev}")
    renderer_class = fromimport(clouddev[0], clouddev[1])  # TODO: is this secure?
    if not issubclass(renderer_class, Renderer): raise RuntimeError(f"renderer isn't a Renderer {clouddev}")
    super().__init__(device, CloudAllocator(self), renderer_class(*clouddev[2]), Compiler(), functools.partial(CloudProgram, self))

  def __del__(self):
    # TODO: this is never being called
    # TODO: should close the whole session
    with contextlib.suppress(ConnectionRefusedError, http.client.CannotSendRequest, http.client.RemoteDisconnected): self.batch_submit()

  def batch_submit(self):
    data = self.req.serialize()
    with Timing(f"*** send {len(self.req._q):-3d} requests {len(self.req._h):-3d} hashes with len {len(data)/1024:.2f} kB in ", enabled=DEBUG>=1):
      ret = self.send("POST", "batch", data)
    self.req = BatchRequest()
    return ret

  def send(self, method, path, data:Optional[bytes]=None) -> bytes:
    # TODO: retry logic
    self.conn.request(method, "/"+path, data, headers={"Cookie": f"session={self.session}"})
    response = self.conn.getresponse()
    assert response.status == 200, f"failed on {method} {path}"
    return response.read()

if __name__ == "__main__": cloud_server(getenv("PORT", 6667))
