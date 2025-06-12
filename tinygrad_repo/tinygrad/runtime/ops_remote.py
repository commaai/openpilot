# the REMOTE=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with REMOTE tinygrad is  frontend <-> middleware <-> RemoteDevice ///HTTP/// remote_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Callable, Iterator, Optional, Any, cast
from collections import defaultdict
from dataclasses import dataclass, field, replace
import multiprocessing, functools, itertools, asyncio, http, http.client, hashlib, time, os, binascii, struct, ast, contextlib, weakref
from tinygrad.renderer import Renderer, ProgramSpec
from tinygrad.dtype import DTYPES_DICT, dtypes
from tinygrad.uop.ops import UOp, Ops, Variable, sint
from tinygrad.helpers import getenv, DEBUG, fromimport, unwrap, Timing
from tinygrad.engine.jit import GraphRunner, MultiGraphRunner, ExecItem, graph_class
from tinygrad.engine.realize import CompiledRunner, BufferXfer
from tinygrad.device import Compiled, Buffer, Allocator, Compiler, Device, BufferSpec
from tinygrad.runtime.graph.cpu import CPUGraph

# ***** API *****

@dataclass(frozen=True)
class RemoteRequest: session: tuple[str, int]|None = field(default=None, kw_only=True)

@dataclass(frozen=True)
class SessionFree(RemoteRequest): pass

@dataclass(frozen=True)
class RemoteProperties:
  real_device: str
  renderer: tuple[str, str, tuple[Any, ...]]
  graph_supported: bool
  graph_supports_multi: bool
  transfer_supported: bool
  offset_supported: bool

@dataclass(frozen=True)
class GetProperties(RemoteRequest): pass

@dataclass(frozen=True)
class BufferAlloc(RemoteRequest): buffer_num: int; size: int; options: BufferSpec # noqa: E702

@dataclass(frozen=True)
class BufferOffset(RemoteRequest): buffer_num: int; size: int; offset: int; sbuffer_num: int # noqa: E702

@dataclass(frozen=True)
class BufferFree(RemoteRequest): buffer_num: int # noqa: E702

@dataclass(frozen=True)
class CopyIn(RemoteRequest): buffer_num: int; datahash: str # noqa: E702

@dataclass(frozen=True)
class CopyOut(RemoteRequest): buffer_num: int

@dataclass(frozen=True)
class Transfer(RemoteRequest): buffer_num: int; ssession: tuple[str, int]; sbuffer_num: int # noqa: E702

@dataclass(frozen=True)
class ProgramAlloc(RemoteRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramFree(RemoteRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramExec(RemoteRequest):
  name: str; datahash: str; bufs: tuple[int, ...]; vals: tuple[int, ...] # noqa: E702
  global_size: Optional[tuple[int, ...]]; local_size: Optional[tuple[int, ...]]; wait: bool # noqa: E702

@dataclass(frozen=True)
class GraphComputeItem:
  session: tuple[str, int]
  name: str
  datahash: str
  bufs: tuple[int, ...]
  vars: tuple[Variable, ...]
  fixedvars: dict[Variable, int]
  ins: tuple[int, ...]
  outs: tuple[int, ...]
  global_size: tuple[sint, ...]|None
  local_size: tuple[sint, ...]|None

@dataclass(frozen=True)
class GraphAlloc(RemoteRequest):
  graph_num: int
  jit_cache: tuple[GraphComputeItem|Transfer, ...]
  bufs: tuple[tuple[tuple[str, int], int], ...]
  var_vals: dict[Variable, int]

@dataclass(frozen=True)
class GraphFree(RemoteRequest):
  graph_num: int

@dataclass(frozen=True)
class GraphExec(RemoteRequest):
  graph_num: int
  bufs: tuple[tuple[tuple[str, int], int], ...]
  var_vals: dict[Variable, int]
  wait: bool

# for safe deserialization
eval_globals = {x.__name__:x for x in [SessionFree, RemoteProperties, GetProperties, BufferAlloc, BufferOffset, BufferFree, CopyIn, CopyOut, Transfer,
                                       ProgramAlloc, ProgramFree, ProgramExec, GraphComputeItem, GraphAlloc, GraphFree, GraphExec, BufferSpec, UOp,
                                       Ops, dtypes]}
attribute_whitelist: dict[Any, set[str]] = {dtypes: {*DTYPES_DICT.keys(), 'imagef', 'imageh'}, Ops: {x.name for x in Ops}}
eval_fxns = {ast.Constant: lambda x: x.value, ast.Tuple: lambda x: tuple(map(safe_eval, x.elts)), ast.List: lambda x: list(map(safe_eval, x.elts)),
  ast.Dict: lambda x: {safe_eval(k):safe_eval(v) for k,v in zip(x.keys, x.values)},
  ast.Call: lambda x: safe_eval(x.func)(*[safe_eval(arg) for arg in x.args], **{kwarg.arg: safe_eval(kwarg.value) for kwarg in x.keywords}),
  ast.Name: lambda x: eval_globals[x.id], ast.Attribute: lambda x: safe_getattr(safe_eval(x.value), x.attr)}
def safe_getattr(value, attr):
  assert attr in attribute_whitelist.get(value, set()), f'getattr({value}, {repr(attr)}) is not whitelisted'
  return getattr(value, attr)
def safe_eval(node): return eval_fxns[node.__class__](node)

class BatchRequest:
  def __init__(self):
    self._q: list[RemoteRequest] = []
    self._h: dict[str, bytes] = {}
  def h(self, d:bytes|memoryview) -> str:
    datahash = hashlib.sha256(d).hexdigest() # NOTE: this is very slow, should use blake3 on gpu instead
    if datahash not in self._h:
      self._h[datahash] = bytes.fromhex(datahash)+struct.pack("<Q", len(d))+bytes(d)
    return datahash
  def q(self, x:RemoteRequest): self._q.append(x)
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
class RemoteSession:
  programs: dict[tuple[str, str], Any] = field(default_factory=dict)
  graphs: dict[int, GraphRunner] = field(default_factory=dict)
  buffers: dict[int, Buffer] = field(default_factory=dict)

class RemoteHandler:
  def __init__(self, base_device: str):
    self.base_device = base_device
    self.sessions: defaultdict[tuple[str, int], RemoteSession] = defaultdict(RemoteSession)

  async def __call__(self, reader:asyncio.StreamReader, writer:asyncio.StreamWriter):
    while (req_hdr:=(await reader.readline()).decode().strip()):
      req_method, req_path, _ = req_hdr.split(' ')
      req_headers = {}
      while (hdr:=(await reader.readline()).decode().strip()):
        key, value = hdr.split(':', 1)
        req_headers[key.lower()] = value.strip()
      req_body = await reader.readexactly(int(req_headers.get("content-length", "0")))
      res_status, res_body = self.handle(req_method, req_path, req_body)
      writer.write(f"HTTP/1.1 {res_status.value} {res_status.phrase}\r\nContent-Length: {len(res_body)}\r\n\r\n".encode() + res_body)

  def handle(self, method:str, path:str, body:bytes) -> tuple[http.HTTPStatus, bytes]:
    status, ret = http.HTTPStatus.OK, b""
    if path == "/batch" and method == "POST":
      # TODO: streaming deserialize?
      req = BatchRequest().deserialize(body)
      # the cmds are always last (currently in datahash)
      for c in req._q:
        if DEBUG >= 1: print(c)
        session, dev = self.sessions[unwrap(c.session)], Device[f"{self.base_device}:{unwrap(c.session)[1]}"]
        match c:
          case SessionFree(): del self.sessions[unwrap(c.session)]
          case GetProperties():
            cls, args = dev.renderer.__reduce__()
            # CPUGraph re-renders kernel from uops specified in CompiledRunner, this is not supported
            graph_cls = gt if (gt:=graph_class(Device[self.base_device])) is not CPUGraph else None
            rp = RemoteProperties(
              real_device=dev.device, renderer=(cls.__module__, cls.__name__, args),
              graph_supported=graph_cls is not None, graph_supports_multi=graph_cls is not None and issubclass(graph_cls, MultiGraphRunner),
              transfer_supported=hasattr(dev.allocator, '_transfer'), offset_supported=hasattr(dev.allocator, '_offset'),
            )
            ret = repr(rp).encode()
          case BufferAlloc():
            assert c.buffer_num not in session.buffers, f"buffer {c.buffer_num} already allocated"
            session.buffers[c.buffer_num] = Buffer(dev.device, c.size, dtypes.uint8, options=c.options, preallocate=True)
          case BufferOffset():
            assert c.buffer_num not in session.buffers, f"buffer {c.buffer_num} already exists"
            session.buffers[c.buffer_num] = session.buffers[c.sbuffer_num].view(c.size, dtypes.uint8, c.offset).allocate()
          case BufferFree(): del session.buffers[c.buffer_num]
          case CopyIn(): session.buffers[c.buffer_num].copyin(memoryview(bytearray(req._h[c.datahash])))
          case CopyOut(): session.buffers[c.buffer_num].copyout(memoryview(ret:=bytearray(session.buffers[c.buffer_num].nbytes)))
          case Transfer():
            ssession, sdev = self.sessions[c.ssession], Device[f"{self.base_device}:{unwrap(c.ssession)[1]}"]
            dbuf, sbuf = session.buffers[c.buffer_num], ssession.buffers[c.sbuffer_num]
            assert dbuf.nbytes == sbuf.nbytes, f"{dbuf.nbytes} != {sbuf.nbytes}"
            assert hasattr(dev.allocator, '_transfer'), f"Device {dev.device} doesn't support transfers"
            dev.allocator._transfer(dbuf._buf, sbuf._buf, dbuf.nbytes, dest_dev=dev, src_dev=sdev)
          case ProgramAlloc():
            lib = dev.compiler.compile_cached(req._h[c.datahash].decode())
            session.programs[(c.name, c.datahash)] = dev.runtime(c.name, lib)
          case ProgramFree(): del session.programs[(c.name, c.datahash)]
          case ProgramExec():
            bufs = [session.buffers[x]._buf for x in c.bufs]
            extra_args = {k:v for k,v in [("global_size", c.global_size), ("local_size", c.local_size)] if v is not None}
            r = session.programs[(c.name, c.datahash)](*bufs, vals=c.vals, wait=c.wait, **extra_args)
            if r is not None: ret = str(r).encode()
          case GraphAlloc():
            graph_fn: Callable = unwrap(dev.graph)
            def _parse_ji(gi: GraphComputeItem|Transfer):
              match gi:
                case GraphComputeItem():
                  prg = self.sessions[gi.session].programs[(gi.name, gi.datahash)]
                  ps = ProgramSpec(gi.name, '', f"{self.base_device}:{gi.session[1]}", UOp(Ops.NOOP),
                                   vars=list(gi.vars), ins=list(gi.ins), outs=list(gi.outs),
                                   global_size=list(cast(tuple[int], gi.global_size)) if gi.global_size is not None else None,
                                   local_size=list(cast(tuple[int], gi.local_size)) if gi.local_size is not None else None)
                  return ExecItem(CompiledRunner(ps, precompiled=b'', prg=prg), [self.sessions[gi.session].buffers[buf] for buf in gi.bufs],
                                  fixedvars=gi.fixedvars)
                case Transfer():
                  dbuf, sbuf = self.sessions[unwrap(gi.session)].buffers[gi.buffer_num], self.sessions[gi.ssession].buffers[gi.sbuffer_num]
                  assert dbuf.nbytes == sbuf.nbytes, f"{dbuf.nbytes} != {sbuf.nbytes}"
                  return ExecItem(BufferXfer(dbuf.nbytes, dbuf.device, sbuf.device), [dbuf, sbuf])
            assert c.graph_num not in session.graphs, f"graph {c.graph_num} already allocated"
            session.graphs[c.graph_num] = graph_fn(list(map(_parse_ji, c.jit_cache)), [self.sessions[s].buffers[i] for s,i in c.bufs], c.var_vals)
          case GraphFree(): del session.graphs[c.graph_num]
          case GraphExec():
            r = session.graphs[c.graph_num]([self.sessions[s].buffers[i] for s,i in c.bufs], c.var_vals, wait=c.wait)
            if r is not None: ret = str(r).encode()
    else: status, ret = http.HTTPStatus.NOT_FOUND, b"Not Found"
    return status, ret

def remote_server(port:int):
  device = getenv("REMOTEDEV", next(Device.get_available_devices()) if Device.DEFAULT == "REMOTE" else Device.DEFAULT)
  async def _inner_async(port:int, device:str):
    print(f"start remote server on {port} with device {device}")
    await (await asyncio.start_server(RemoteHandler(device), host='', port=port)).serve_forever()
  asyncio.run(_inner_async(port, device))

# ***** frontend *****

class RemoteAllocator(Allocator['RemoteDevice']):
  def __init__(self, dev:RemoteDevice):
    if dev.properties.offset_supported: self._offset = self._dyn_offset
    super().__init__(dev)
  # TODO: ideally we shouldn't have to deal with images here
  def _alloc(self, size:int, options:BufferSpec) -> int:
    self.dev.q(BufferAlloc(buffer_num:=next(self.dev.buffer_num), size, options))
    return buffer_num
  # TODO: options should not be here in any Allocator
  def _free(self, opaque:int, options): self.dev.q(BufferFree(opaque))
  def _copyin(self, dest:int, src:memoryview): self.dev.q(CopyIn(dest, self.dev.conn.req.h(src)))
  def _copyout(self, dest:memoryview, src:int):
    resp = self.dev.q(CopyOut(src), wait=True)
    assert len(resp) == len(dest), f"buffer length mismatch {len(resp)} != {len(dest)}"
    dest[:] = resp
  def _transfer(self, dest, src, sz, src_dev, dest_dev):
    if dest_dev.properties.transfer_supported and src_dev.conn == dest_dev.conn:
      dest_dev.q(Transfer(dest, src_dev.session, src))
    else:
      src_dev.allocator._copyout(tmp:=memoryview(bytearray(sz)), src)
      dest_dev.allocator._copyin(dest, tmp)
  def _dyn_offset(self, opaque:int, size:int, offset:int) -> int:
    self.dev.q(BufferOffset(buffer_num:=next(self.dev.buffer_num), size, offset, opaque))
    return buffer_num

class RemoteProgram:
  def __init__(self, dev:RemoteDevice, name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.datahash = self.dev.conn.req.h(lib)
    self.dev.q(ProgramAlloc(self.name, self.datahash))
    super().__init__()
    weakref.finalize(self, self._fini, self.dev, self.name, self.datahash)

  @staticmethod
  def _fini(dev:RemoteDevice, name:str, datahash:str): dev.q(ProgramFree(name, datahash))

  def __call__(self, *bufs, global_size=None, local_size=None, vals:tuple[int, ...]=(), wait=False):
    ret = self.dev.q(ProgramExec(self.name, self.datahash, bufs, vals, global_size, local_size, wait), wait=wait)
    if wait: return float(ret)

@functools.cache
class RemoteConnection:
  def __init__(self, host:str):
    if DEBUG >= 1: print(f"remote with host {host}")
    while 1:
      try:
        self.conn = http.client.HTTPConnection(host, timeout=getenv("REMOTE_TIMEOUT", 300.0))
        self.conn.connect()
        break
      except Exception as e:
        print(e)
        time.sleep(0.1)
    self.req: BatchRequest = BatchRequest()

  def batch_submit(self):
    data = self.req.serialize()
    with Timing(f"*** send {len(self.req._q):-3d} requests {len(self.req._h):-3d} hashes with len {len(data)/1024:.2f} kB in ", enabled=DEBUG>=3):
      self.conn.request("POST", "/batch", data)
      response = self.conn.getresponse()
      assert response.status == 200, f"POST /batch failed: {response}"
      ret = response.read()
    self.req = BatchRequest()
    return ret

class RemoteDevice(Compiled):
  def __init__(self, device:str):
    self.conn: RemoteConnection = RemoteConnection(getenv("HOST", "") or RemoteDevice.local_server())

    # state for the connection
    self.session = (binascii.hexlify(os.urandom(0x10)).decode(), int(device.split(":")[1]) if ":" in device else 0)
    self.buffer_num: Iterator[int] = itertools.count(0)
    self.graph_num: Iterator[int] = itertools.count(0)

    self.properties: RemoteProperties = safe_eval(ast.parse(self.q(GetProperties(), wait=True), mode="eval").body)
    if DEBUG >= 1: print(f"remote has device {self.properties.real_device}")
    # TODO: how to we have BEAM be cached on the backend? this should just send a specification of the compute. rethink what goes in Renderer
    renderer = self.properties.renderer
    if not renderer[0].startswith("tinygrad.renderer.") or not renderer[1].endswith("Renderer"): raise RuntimeError(f"bad renderer {renderer}")
    renderer_class = fromimport(renderer[0], renderer[1])  # TODO: is this secure?
    if not issubclass(renderer_class, Renderer): raise RuntimeError(f"renderer isn't a Renderer {renderer}")
    renderer_instance = renderer_class(*renderer[2])
    renderer_instance.device = device
    graph_supported, graph_multi = self.properties.graph_supported, self.properties.graph_supports_multi
    graph = fromimport('tinygrad.runtime.graph.remote', f"Remote{'Multi' if graph_multi else ''}Graph") if graph_supported else None
    super().__init__(device, RemoteAllocator(self), renderer_instance, Compiler(), functools.partial(RemoteProgram, self), graph)

  def finalize(self):
    with contextlib.suppress(ConnectionError, http.client.HTTPException): self.q(SessionFree(), wait=True)

  def q(self, x:RemoteRequest, wait:bool=False):
    self.conn.req.q(replace(x, session=self.session))
    if wait: return self.conn.batch_submit()

  @functools.cache
  @staticmethod
  def local_server():
    multiprocessing.Process(target=remote_server, args=(6667,), name="MainProcess", daemon=True).start()
    return "127.0.0.1:6667"

if __name__ == "__main__": remote_server(getenv("PORT", 6667))
