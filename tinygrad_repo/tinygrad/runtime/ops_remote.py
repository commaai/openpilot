# the REMOTE=1 device is a process boundary between the frontend/runtime
# normally tinygrad is    frontend <-> middleware <-> runtime <-> hardware
# with REMOTE tinygrad is  frontend <-> middleware <-> RemoteDevice ///HTTP/// remote_server <-> runtime <-> hardware
# this client and server can be on the same machine, same network, or just same internet
# it should be a secure (example: no use of pickle) boundary. HTTP is used for RPC

from __future__ import annotations
from typing import Callable, Iterator, Any, cast
from collections import defaultdict
from dataclasses import dataclass, field, replace
import multiprocessing, threading, functools, itertools, asyncio, http, http.client, hashlib, time, os, binascii, struct, ast, contextlib, weakref
import traceback, builtins
from tinygrad.renderer import Renderer, ProgramSpec
from tinygrad.dtype import DTYPES_DICT, dtypes
from tinygrad.uop.ops import UOp, Ops, Variable, sint
from tinygrad.helpers import getenv, DEBUG, fromimport, unwrap, LazySeq, Timing
from tinygrad.engine.jit import GraphRunner, MultiGraphRunner, ExecItem, graph_class
from tinygrad.engine.realize import CompiledRunner, BufferXfer
from tinygrad.device import Compiled, Buffer, Allocator, Compiler, Device, BufferSpec
from tinygrad.runtime.support.ib import IBCtx, IBConn, SGE

# ***** API *****

@dataclass(frozen=True)
class SessionKey: host: str; idx: int; nonce: str # noqa: E702

@dataclass(frozen=True)
class RemoteRequest: session: SessionKey|None = field(default=None, kw_only=True)

@dataclass(frozen=True)
class SessionFree(RemoteRequest): pass

@dataclass(frozen=True)
class RemoteProperties:
  real_device: str
  renderer: tuple[str, str, tuple[Any, ...]]
  offset_supported: bool
  graph_supported: bool
  graph_supports_multi: bool
  ib_gid: bytes|None

@dataclass(frozen=True)
class RemoteException:
  exc: Exception
  trace: str = ""

@dataclass(frozen=True)
class GetProperties(RemoteRequest): pass

@dataclass(frozen=True)
class Event(RemoteRequest): event_session: SessionKey; event: int # noqa: E702

@dataclass(frozen=True)
class Wait(RemoteRequest): event: int

@dataclass(frozen=True)
class IBConnect(RemoteRequest): host: str; gid: bytes; qp_num: int # noqa: E702

@dataclass(frozen=True)
class BufferAlloc(RemoteRequest): buffer_num: int; size: int; options: BufferSpec # noqa: E702

@dataclass(frozen=True)
class BufferOffset(RemoteRequest): buffer_num: int; size: int; offset: int; sbuffer_num: int # noqa: E702

@dataclass(frozen=True)
class BufferIOVAS(RemoteRequest): buffer_nums: list[tuple[SessionKey, int]] # noqa: E702

@dataclass(frozen=True)
class BufferFree(RemoteRequest): buffer_num: int # noqa: E702

@dataclass(frozen=True)
class CopyIn(RemoteRequest): buffer_num: int; datahash: str # noqa: E702

@dataclass(frozen=True)
class CopyOut(RemoteRequest): buffer_num: int

@dataclass(frozen=True)
class Transfer(RemoteRequest): buffer_num: int; dsession: SessionKey; dbuffer_num: int # noqa: E702

@dataclass(frozen=True)
class BatchTransfer(RemoteRequest):
  sbuffer_nums: list[tuple[SessionKey, int]]
  dbuffer_nums: list[tuple[SessionKey, int]]

@dataclass(frozen=True)
class ProgramAlloc(RemoteRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramFree(RemoteRequest): name: str; datahash: str # noqa: E702

@dataclass(frozen=True)
class ProgramExec(RemoteRequest):
  name: str; datahash: str; bufs: tuple[int, ...]; vals: tuple[int, ...] # noqa: E702
  global_size: tuple[int, ...]|None; local_size: tuple[int, ...]|None; wait: bool # noqa: E702

@dataclass(frozen=True)
class GraphComputeItem:
  session: SessionKey
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
  bufs: tuple[tuple[SessionKey, int], ...]
  var_vals: dict[Variable, int]

@dataclass(frozen=True)
class GraphFree(RemoteRequest):
  graph_num: int

@dataclass(frozen=True)
class GraphExec(RemoteRequest):
  graph_num: int
  bufs: tuple[tuple[SessionKey, int], ...]
  var_vals: dict[Variable, int]
  wait: bool

# for safe deserialization
eval_excs = [v for k,v in builtins.__dict__.items() if isinstance(v, type) and issubclass(v, Exception) and not k.endswith("Warning")]
eval_globals = {x.__name__:x for x in [SessionKey, SessionFree, RemoteProperties, GetProperties, Event, Wait, BufferAlloc, BufferOffset, BufferIOVAS,
                                       BufferFree, CopyIn, CopyOut, Transfer, BatchTransfer, IBConnect, ProgramAlloc, ProgramFree, ProgramExec,
                                       GraphComputeItem, GraphAlloc, GraphFree, GraphExec, BufferSpec, UOp, Ops, dtypes, RemoteException] + eval_excs}
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
  events: defaultdict[int, asyncio.Event] = field(default_factory=functools.partial(defaultdict, asyncio.Event))

class RemoteHandler:
  def __init__(self, base_device: str):
    self.base_device = base_device
    self.sessions: defaultdict[SessionKey, RemoteSession] = defaultdict(RemoteSession)

    try: self.ib_ctx: IBCtx|None = IBCtx(getenv("IB_DEV", 0))
    except (IndexError, AttributeError): self.ib_ctx = None
    self.ib_lock = asyncio.Lock()
    self.ib_conns: dict[str, IBConn|None] = {}
    self.iova_cache: dict[tuple[SessionKey, int], tuple[int, int, int]] = {}

  async def __call__(self, reader:asyncio.StreamReader, writer:asyncio.StreamWriter):
    while (req_hdr:=(await reader.readline()).decode().strip()):
      req_method, req_path, _ = req_hdr.split(' ')
      req_headers = {}
      while (hdr:=(await reader.readline()).decode().strip()):
        key, value = hdr.split(':', 1)
        req_headers[key.lower()] = value.strip()
      req_body = await reader.readexactly(int(req_headers.get("content-length", "0")))
      try: res_status, res_body = await self.handle(req_method, req_path, req_body)
      except Exception as e:
        res_status, res_body = http.HTTPStatus.INTERNAL_SERVER_ERROR, repr(RemoteException(e, traceback.format_exc())).encode()
        print(f"{traceback.format_exc()}", flush=True)
      writer.write(f"HTTP/1.1 {res_status.value} {res_status.phrase}\r\nContent-Length: {len(res_body)}\r\n\r\n".encode() + res_body)

  async def ib_connect(self, ssession:SessionKey, dsession:SessionKey) -> IBConn|None:
    if self.ib_ctx is None: return None
    await self.ib_lock.acquire()
    conn = RemoteConnection(dsession.host)
    if dsession.host not in self.ib_conns:
      props = safe_eval(ast.parse(conn.q(GetProperties(session=dsession), wait=True), mode="eval").body)
      if props.ib_gid is not None:
        self.ib_conns[dsession.host] = ib_conn = IBConn(self.ib_ctx)
        ibxc_ret = conn.q(IBConnect(ssession.host, ib_conn.gid, ib_conn.qp_num, session=dsession), wait=True)
        ib_conn.connect(*struct.unpack('<16sQ', ibxc_ret))
      else:
        self.ib_conns[dsession.host] = None
    self.ib_lock.release()
    return self.ib_conns[dsession.host]

  async def get_iovas(self, bufs:list[tuple[SessionKey, int]]) -> list[tuple[int, int, int]]:
    await self.ib_lock.acquire()
    if (rbufs:=[buf for buf in bufs if buf not in self.iova_cache]):
      conn = RemoteConnection(rbufs[0][0].host)
      resp = await conn.aq(BufferIOVAS(rbufs, session=rbufs[0][0]), wait=True)
      self.iova_cache.update({rbuf: struct.unpack('<QQQ', resp[i*24:(i+1)*24]) for i,rbuf in enumerate(rbufs)})
    self.ib_lock.release()
    return [self.iova_cache[buf] for buf in bufs]

  async def handle(self, method:str, path:str, body:bytes) -> tuple[http.HTTPStatus, bytes]:
    status, ret = http.HTTPStatus.OK, b""
    if path == "/batch" and method == "POST":
      # TODO: streaming deserialize?
      req = BatchRequest().deserialize(body)
      # the cmds are always last (currently in datahash)
      for c in req._q:
        if DEBUG >= 1: print(c)
        session, dev = self.sessions[unwrap(c.session)], Device[f"{self.base_device}:{unwrap(c.session).idx}"]
        match c:
          case SessionFree(): del self.sessions[unwrap(c.session)]
          case GetProperties():
            cls, args = dev.renderer.__reduce__()
            graph_cls = graph_class(Device[self.base_device])
            rp = RemoteProperties(
              real_device=dev.device, renderer=(cls.__module__, cls.__name__, args), offset_supported=hasattr(dev.allocator, '_offset'),
              graph_supported=graph_cls is not None,
              graph_supports_multi=graph_cls is not None and issubclass(graph_cls, MultiGraphRunner) and hasattr(dev.allocator, '_transfer'),
              ib_gid=bytes(self.ib_ctx.gid_attr.raw) if self.ib_ctx is not None else None,
            )
            ret = repr(rp).encode()
          case Event():
            if c.session == c.event_session:
              session.events[c.event].set()
            else:
              for d in Device._opened_devices: Device[d].synchronize() # wait for device*s* to finish executing previous stuff
              # TODO: don't wait, just send
              await RemoteConnection(c.event_session.host).aq(Event(c.event_session, c.event, session=c.event_session), wait=True)
          case Wait():
            assert await session.events[c.event].wait()
            del session.events[c.event] # do not leak memory
          case IBConnect():
            self.ib_conns[c.host] = ibc = IBConn(unwrap(self.ib_ctx))
            ibc.connect(c.gid, c.qp_num)
            ret = struct.pack('<16sQ', ibc.gid, ibc.qp_num)
          case BufferAlloc():
            assert c.buffer_num not in session.buffers, f"buffer {c.buffer_num} already allocated"
            session.buffers[c.buffer_num] = Buffer(dev.device, c.size, dtypes.uint8, options=c.options, preallocate=True)
          case BufferIOVAS():
            rets = []
            for buffer_session,buffer_num in c.buffer_nums:
              iova, mr = unwrap(self.ib_ctx).reg(buf:=self.sessions[buffer_session].buffers[buffer_num])
              rets.append(struct.pack("<QQQ", iova, mr.contents.rkey, buf.nbytes))
            ret = b"".join(rets)
          case BufferOffset():
            assert c.buffer_num not in session.buffers, f"buffer {c.buffer_num} already exists"
            session.buffers[c.buffer_num] = session.buffers[c.sbuffer_num].view(c.size, dtypes.uint8, c.offset).allocate()
          case BufferFree(): del session.buffers[c.buffer_num]
          case CopyIn(): session.buffers[c.buffer_num].copyin(memoryview(bytearray(req._h[c.datahash])))
          case CopyOut(): session.buffers[c.buffer_num].copyout(memoryview(ret:=bytearray(session.buffers[c.buffer_num].nbytes)))
          case Transfer():
            if c.dsession.host == unwrap(c.session).host:
              dsession, ddev = self.sessions[c.dsession], Device[f"{self.base_device}:{unwrap(c.dsession).idx}"]
              dbuf, sbuf = dsession.buffers[c.dbuffer_num], session.buffers[c.buffer_num]
              if hasattr(ddev.allocator, '_transfer'):
                assert dbuf.nbytes == sbuf.nbytes, f"{dbuf.nbytes} != {sbuf.nbytes}"
                ddev.allocator._transfer(dbuf._buf, sbuf._buf, dbuf.nbytes, dest_dev=ddev, src_dev=dev)
              else:
                sbuf.copyout(data:=memoryview(bytearray(sbuf.nbytes)))
                dbuf.copyin(data)
            else:
              conn, ib_conn = RemoteConnection(c.dsession.host), await self.ib_connect(unwrap(c.session), c.dsession)
              sbuf = session.buffers[c.buffer_num]
              if ib_conn is not None:
                src_iova, src_mr = unwrap(self.ib_ctx).reg(sbuf)
                dst_iova, dst_key, dst_size = (await self.get_iovas([(c.dsession, c.dbuffer_num)]))[0]
                assert sbuf.nbytes == dst_size, f"{sbuf.nbytes} != {dst_size}"
                for d in Device._opened_devices: Device[d].synchronize()
                ib_conn.rdma_write([SGE(dst_iova, dst_key, src_iova, src_mr.contents.lkey, dst_size)])
              else:
                sbuf.copyout(data:=memoryview(bytearray(sbuf.nbytes)))
                await conn.aq(CopyIn(c.dbuffer_num, conn.req.h(data), session=c.dsession), wait=True)
          case BatchTransfer():
            conn, ib_conn = RemoteConnection(c.dbuffer_nums[0][0].host), await self.ib_connect(c.sbuffer_nums[0][0], c.dbuffer_nums[0][0])
            if ib_conn is not None:
              sbufs = [unwrap(self.ib_ctx).reg(self.sessions[s].buffers[bi]) for s,bi in c.sbuffer_nums]
              dbufs = await self.get_iovas(c.dbuffer_nums)
              for d in Device._opened_devices: Device[d].synchronize()
              ib_conn.rdma_write([SGE(di, dk, si, sm.contents.lkey, ds) for (di,dk,ds),(si,sm) in zip(dbufs, sbufs)])
            else:
              for (sbuf_session,sbuf_num),(dbuf_session,dbuf_num) in zip(c.sbuffer_nums, c.dbuffer_nums):
                sbuf = self.sessions[sbuf_session].buffers[sbuf_num]
                sbuf.copyout(data:=memoryview(bytearray(sbuf.nbytes)))
                await conn.aq(CopyIn(dbuf_num, conn.req.h(data), session=dbuf_session), wait=True)
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
                  ps = ProgramSpec(gi.name, '', f"{self.base_device}:{gi.session.idx}", UOp(Ops.NOOP),
                                   vars=list(gi.vars), ins=list(gi.ins), outs=list(gi.outs),
                                   global_size=list(cast(tuple[int], gi.global_size)) if gi.global_size is not None else None,
                                   local_size=list(cast(tuple[int], gi.local_size)) if gi.local_size is not None else None)
                  return ExecItem(CompiledRunner(ps, precompiled=b'', prg=prg), [self.sessions[gi.session].buffers[buf] for buf in gi.bufs],
                                  fixedvars=gi.fixedvars)
                case Transfer():
                  dbuf, sbuf = self.sessions[gi.dsession].buffers[gi.dbuffer_num], self.sessions[unwrap(gi.session)].buffers[gi.buffer_num]
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
  def _free(self, opaque:int, options):
    try: self.dev.q(BufferFree(opaque))
    except (TypeError, AttributeError): pass
  def _copyin(self, dest:int, src:memoryview): self.dev.q(CopyIn(dest, self.dev.conn.req.h(src)))
  def _copyout(self, dest:memoryview, src:int):
    resp = self.dev.q(CopyOut(src), wait=True)
    assert len(resp) == len(dest), f"buffer length mismatch {len(resp)} != {len(dest)}"
    dest[:] = resp
  def _transfer(self, dest, src, sz, src_dev, dest_dev):
    if dest_dev.conn != src_dev.conn:
      dest_dev.q(Event(src_dev.session, start_event:=next(src_dev.event_num)))
      src_dev.q(Wait(start_event))
    src_dev.q(Transfer(src, dest_dev.session, dest))
    if dest_dev.conn != src_dev.conn:
      src_dev.q(Event(dest_dev.session, end_event:=next(dest_dev.event_num)))
      dest_dev.q(Wait(end_event))
    if DEBUG >= 2: dest_dev.conn.batch_submit()
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
  q_lock = threading.Lock()
  all: dict[RemoteConnection, None] = {} # dict instead of set for deterministic ordering

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
    RemoteConnection.all[self] = None

  def q(self, x:RemoteRequest, wait:bool=False):
    with RemoteConnection.q_lock:
      self.req.q(x)
      if wait: return self.batch_submit(take_q=False)

  async def aq(self, x:RemoteRequest, wait:bool=False): return await asyncio.to_thread(self.q, x, wait=wait)

  def batch_submit(self, take_q:bool=True):
    if take_q: RemoteConnection.q_lock.acquire()
    conns = RemoteConnection.all.keys()
    datas = {conn: conn.req.serialize() for conn in conns}
    reqs, hashes, hash_datas = sum(len(c.req._q) for c in conns), sum(len(c.req._h) for c in conns), sum(len(data) for data in datas.values())
    with Timing(f"*** send {reqs:-3d} requests {hashes:-3d} hashes with len {hash_datas/1024:.2f} kB in ", enabled=DEBUG>=3):
      for conn,data in datas.items(): conn.conn.request("POST", "/batch", data)
      for conn in datas.keys():
        response = conn.conn.getresponse()
        resp = response.read()
        conn.req = BatchRequest() # no matter what response, reset conn
        if response.status == http.HTTPStatus.INTERNAL_SERVER_ERROR:
          exc_wrapper = safe_eval(ast.parse(resp.decode(), mode="eval").body)
          exc_wrapper.exc.add_note(exc_wrapper.trace)
          raise exc_wrapper.exc
        assert response.status == http.HTTPStatus.OK, f"POST /batch failed: {resp.decode()}"
        if conn == self: ret = resp
    if take_q: RemoteConnection.q_lock.release()
    return ret

def parse_hosts(hs:str) -> list[tuple[str, int]]|LazySeq[tuple[str, int]]:
  hosts = [(unwrap(h), int(c) if c is not None else c) for h,c in ((h.split("*", maxsplit=1)+[None,])[:2] for h in hs.split(","))]
  if len(hosts) == 1 and hosts[0][1] is None: return LazySeq(lambda idx: (hosts[0][0], idx))
  return [(h, i) for h,c in hosts for i in range(unwrap(c))]

class RemoteDevice(Compiled):
  devices = parse_hosts(getenv("HOST", ""))

  def __init__(self, device:str):
    host, idx = RemoteDevice.devices[int(device.split(":")[1]) if ":" in device else 0]

    # connection is shared between sessions on the same host
    self.session: SessionKey = SessionKey(host or RemoteDevice.local_server(), idx, binascii.hexlify(os.urandom(0x10)).decode())
    self.conn: RemoteConnection = RemoteConnection(self.session.host)

    # state for the session
    self.buffer_num: Iterator[int] = itertools.count(0)
    self.graph_num: Iterator[int] = itertools.count(0)
    self.event_num: Iterator[int] = itertools.count(0)

    self.properties: RemoteProperties = safe_eval(ast.parse(self.q(GetProperties(), wait=True), mode="eval").body)
    if DEBUG >= 1: print(f"remote has device {self.properties.real_device}")
    # TODO: how to we have BEAM be cached on the backend? this should just send a specification of the compute. rethink what goes in Renderer
    renderer = self.properties.renderer
    if not renderer[0].startswith("tinygrad.") or not renderer[1].endswith("Renderer"): raise RuntimeError(f"bad renderer {renderer}")
    renderer_class = fromimport(renderer[0], renderer[1])  # TODO: is this secure?
    if not issubclass(renderer_class, Renderer): raise RuntimeError(f"renderer isn't a Renderer {renderer}")
    renderer_instance = renderer_class(*renderer[2])
    renderer_instance.device = device
    graph = fromimport('tinygrad.runtime.graph.remote', "RemoteGraph") if self.properties.graph_supported else None
    super().__init__(device, RemoteAllocator(self), renderer_instance, Compiler(), functools.partial(RemoteProgram, self), graph, id(self.conn))

  def finalize(self):
    with contextlib.suppress(ConnectionError, http.client.HTTPException): self.q(SessionFree(), wait=True)

  def q(self, x:RemoteRequest, wait:bool=False): return self.conn.q(replace(x, session=self.session), wait=wait)

  @functools.cache
  @staticmethod
  def local_server():
    multiprocessing.Process(target=remote_server, args=(6667,), name="MainProcess", daemon=True).start()
    return "127.0.0.1:6667"

if __name__ == "__main__": remote_server(getenv("PORT", 6667))
