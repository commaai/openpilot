from __future__ import annotations
import os, functools, platform, time, re, contextlib, operator, hashlib, pickle, sqlite3, tempfile, pathlib, string, ctypes, sys, gzip
import urllib.request, subprocess, shutil, math, contextvars, types, copyreg, inspect, importlib
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List, ClassVar, Optional, Iterable, Any, TypeVar, Callable, Sequence, TypeGuard

T = TypeVar("T")
U = TypeVar("U")
# NOTE: it returns int 1 if x is empty regardless of the type of x
def prod(x:Iterable[T]) -> Union[T,int]: return functools.reduce(operator.mul, x, 1)

# NOTE: helpers is not allowed to import from anything else in tinygrad
OSX = platform.system() == "Darwin"
CI = os.getenv("CI", "") != ""

# fix colors on Windows, https://stackoverflow.com/questions/12492810/python-how-can-i-make-the-ansi-escape-codes-to-work-also-in-windows
if sys.platform == "win32": os.system("")

def dedup(x:Iterable[T]): return list(dict.fromkeys(x))   # retains list order
def argfix(*x):
  if x and x[0].__class__ in (tuple, list):
    if len(x) != 1: raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items:Union[Tuple[T, ...], List[T]]): return all(x == items[0] for x in items)
def all_int(t: Sequence[Any]) -> TypeGuard[Tuple[int, ...]]: return all(isinstance(s, int) for s in t)
def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line  # noqa: E501
def colorize_float(x: float): return colored(f"{x:7.2f}x", 'green' if x < 0.75 else 'red' if x > 1.15 else 'yellow')
def memsize_to_str(_bytes: int) -> str: return [f"{(_bytes / d):.2f} {pr}" for d,pr in [(1e9,"GB"),(1e6,"MB"),(1e3,"KB"),(1,"B")] if _bytes > d][0]
def ansistrip(s:str): return re.sub('\x1b\\[(K|.*?m)', '', s)
def ansilen(s:str): return len(ansistrip(s))
def make_tuple(x:Union[int, Sequence[int]], cnt:int) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else tuple(x)
def flatten(l:Iterable[Iterable[T]]): return [item for sublist in l for item in sublist]
def fully_flatten(l):
  if hasattr(l, "__len__") and hasattr(l, "__getitem__") and not isinstance(l, str):
    if hasattr(l, "shape") and l.shape == (): return [l[()]]
    flattened = []
    for li in l: flattened.extend(fully_flatten(li))
    return flattened
  return [l]
def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
def strip_parens(fst:str): return fst[1:-1] if fst[0] == '(' and fst[-1] == ')' and fst[1:-1].find('(') <= fst[1:-1].find(')') else fst
def ceildiv(num, amt): return int(ret) if isinstance((ret:=-(num//-amt)), float) else ret
def round_up(num:int, amt:int) -> int: return (num+amt-1)//amt * amt
def data64(data:Any) -> Tuple[Any, Any]: return (data >> 32, data & 0xFFFFFFFF) # Any is sint
def data64_le(data:Any) -> Tuple[Any, Any]: return (data & 0xFFFFFFFF, data >> 32) # Any is sint
def merge_dicts(ds:Iterable[Dict[T,U]]) -> Dict[T,U]:
  kvs = set([(k,v) for d in ds for k,v in d.items()])
  assert len(kvs) == len(set(kv[0] for kv in kvs)), f"cannot merge, {kvs} contains different values for the same key"
  return {k:v for d in ds for k,v in d.items()}
def partition(itr:Iterable[T], fxn:Callable[[T],bool]) -> Tuple[List[T], List[T]]:
  ret:Tuple[List[T], List[T]] = ([], [])
  for s in itr: (ret[0] if fxn(s) else ret[1]).append(s)
  return ret
def unwrap(x:Optional[T]) -> T:
  assert x is not None
  return x
def get_child(obj, key):
  for k in key.split('.'):
    if k.isnumeric(): obj = obj[int(k)]
    elif isinstance(obj, dict): obj = obj[k]
    else: obj = getattr(obj, k)
  return obj
def word_wrap(x, wrap=80): return x if len(x) <= wrap or '\n' in x[0:wrap] else (x[0:wrap] + "\n" + word_wrap(x[wrap:], wrap))

# for length N coefficients `p`, returns p[0] * x**(N-1) + p[1] * x**(N-2) + ... + p[-2] * x + p[-1]
def polyN(x:T, p:List[float]) -> T: return functools.reduce(lambda acc,c: acc*x+c, p, 0.0)  # type: ignore

@functools.lru_cache(maxsize=None)
def to_function_name(s:str): return ''.join([c if c in (string.ascii_letters+string.digits+'_') else f'{ord(c):02X}' for c in ansistrip(s)])
@functools.lru_cache(maxsize=None)
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))
def temp(x:str) -> str: return (pathlib.Path(tempfile.gettempdir()) / x).as_posix()

class Context(contextlib.ContextDecorator):
  stack: ClassVar[List[dict[str, int]]] = [{}]
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    Context.stack[-1] = {k:o.value for k,o in ContextVar._cache.items()} # Store current state.
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v # Update to new temporary state.
    Context.stack.append(self.kwargs) # Store the temporary state so we know what to undo later.
  def __exit__(self, *args):
    for k in Context.stack.pop(): ContextVar._cache[k].value = Context.stack[-1].get(k, ContextVar._cache[k].value)

class ContextVar:
  _cache: ClassVar[Dict[str, ContextVar]] = {}
  value: int
  key: str
  def __init__(self, key, default_value):
    assert key not in ContextVar._cache, f"attempt to recreate ContextVar {key}"
    ContextVar._cache[key] = self
    self.value, self.key = getenv(key, default_value), key
  def __bool__(self): return bool(self.value)
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

DEBUG, IMAGE, BEAM, NOOPT, JIT = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0), ContextVar("BEAM", 0), ContextVar("NOOPT", 0), ContextVar("JIT", 1)
WINO, CAPTURING, TRACEMETA = ContextVar("WINO", 0), ContextVar("CAPTURING", 1), ContextVar("TRACEMETA", 1)
PROFILE, PROFILEPATH = ContextVar("PROFILE", 0), ContextVar("PROFILEPATH", temp("tinygrad_profile.json"))
USE_TC, TC_OPT, AMX, TRANSCENDENTAL = ContextVar("TC", 1), ContextVar("TC_OPT", 0), ContextVar("AMX", 0), ContextVar("TRANSCENDENTAL", 1)
FUSE_ARANGE, FUSE_CONV_BW = ContextVar("FUSE_ARANGE", 0), ContextVar("FUSE_CONV_BW", 0)
SPLIT_REDUCEOP, NO_MEMORY_PLANNER, RING = ContextVar("SPLIT_REDUCEOP", 1), ContextVar("NO_MEMORY_PLANNER", 0), ContextVar("RING", 1)
PICKLE_BUFFERS = ContextVar("PICKLE_BUFFERS", 1)

@dataclass(frozen=True)
class Metadata:
  name: str
  caller: str
  backward: bool = False
  def __hash__(self): return hash(self.name)
  def __repr__(self): return str(self) + (f" - {self.caller}" if self.caller else "")
  def __str__(self): return self.name + (" bw" if self.backward else "")
_METADATA: contextvars.ContextVar[Optional[Metadata]] = contextvars.ContextVar("_METADATA", default=None)

# **************** global state Counters ****************

class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count = 0,0,0.0,0

# **************** timer and profiler ****************

class Timing(contextlib.ContextDecorator):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, *exc):
    self.et = time.perf_counter_ns() - self.st
    if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

def _format_fcn(fcn): return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"
class Profiling(contextlib.ContextDecorator):
  def __init__(self, enabled=True, sort='cumtime', frac=0.2, fn=None, ts=1):
    self.enabled, self.sort, self.frac, self.fn, self.time_scale = enabled, sort, frac, fn, 1e3/ts
  def __enter__(self):
    import cProfile
    self.pr = cProfile.Profile()
    if self.enabled: self.pr.enable()
  def __exit__(self, *exc):
    if self.enabled:
      self.pr.disable()
      if self.fn: self.pr.dump_stats(self.fn)
      import pstats
      stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
      for fcn in stats.fcn_list[0:int(len(stats.fcn_list)*self.frac)]:    # type: ignore[attr-defined]
        (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]    # type: ignore[attr-defined]
        scallers = sorted(callers.items(), key=lambda x: -x[1][2])
        print(f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms",
              colored(_format_fcn(fcn).ljust(50), "yellow"),
              colored(f"<- {(scallers[0][1][2]/tottime)*100:3.0f}% {_format_fcn(scallers[0][0])}", "BLACK") if scallers else '')

# *** universal database cache ***

cache_dir: str = os.path.join(getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache")), "tinygrad")
CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(cache_dir, "cache.db")))
CACHELEVEL = getenv("CACHELEVEL", 2)

VERSION = 17
_db_connection = None
def db_connection():
  global _db_connection
  if _db_connection is None:
    os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
    _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")
    # another connection has set it already or is in the process of setting it
    # that connection will lock the database
    with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
    if DEBUG >= 7: _db_connection.set_trace_callback(print)
  return _db_connection

def diskcache_clear():
  cur = db_connection().cursor()
  drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
  cur.executescript("\n".join([s[0] for s in drop_tables] + ["VACUUM;"]))

def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
  if CACHELEVEL == 0: return None
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  try:
    res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  except sqlite3.OperationalError:
    return None  # table doesn't exist
  if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
  return None

_db_tables = set()
def diskcache_put(table:str, key:Union[Dict, str, int], val:Any, prepickled=False):
  if CACHELEVEL == 0: return val
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  if table not in _db_tables:
    TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
    ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}_{VERSION}' ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
    _db_tables.add(table)
  cur.execute(f"REPLACE INTO '{table}_{VERSION}' ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key.keys()))}, ?)", tuple(key.values()) + (val if prepickled else pickle.dumps(val), ))  # noqa: E501
  conn.commit()
  cur.close()
  return val

def diskcache(func):
  def wrapper(*args, **kwargs) -> bytes:
    table, key = f"cache_{func.__name__}", hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
    if (ret:=diskcache_get(table, key)): return ret
    return diskcache_put(table, key, func(*args, **kwargs))
  return wrapper

# *** http support ***

def _ensure_downloads_dir() -> pathlib.Path:
  # if we are on a tinybox, use the raid array
  if pathlib.Path("/etc/tinybox-release").is_file():
    # try creating dir with sudo
    if not (downloads_dir := pathlib.Path("/raid/downloads")).exists():
      subprocess.run(["sudo", "mkdir", "-p", downloads_dir], check=True)
      subprocess.run(["sudo", "chown", "tiny:root", downloads_dir], check=True)
      subprocess.run(["sudo", "chmod", "775", downloads_dir], check=True)
    return downloads_dir
  return pathlib.Path(cache_dir) / "downloads"

def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None, subdir:Optional[str]=None, gunzip:bool=False,
          allow_caching=not getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  if url.startswith(("/", ".")): return pathlib.Path(url)
  if name is not None and (isinstance(name, pathlib.Path) or '/' in name): fp = pathlib.Path(name)
  else: fp = _ensure_downloads_dir() / (subdir or "") / ((name or hashlib.md5(url.encode('utf-8')).hexdigest()) + (".gunzip" if gunzip else ""))
  if not fp.is_file() or not allow_caching:
    (_dir := fp.parent).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=10) as r:
      assert r.status == 200, r.status
      length = int(r.headers.get('content-length', 0)) if not gunzip else None
      readfile = gzip.GzipFile(fileobj=r) if gunzip else r
      progress_bar = tqdm(total=length, unit='B', unit_scale=True, desc=f"{url}", disable=CI)
      with tempfile.NamedTemporaryFile(dir=_dir, delete=False) as f:
        while chunk := readfile.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        pathlib.Path(f.name).rename(fp)
      progress_bar.update(close=True)
      if length and (file_size:=os.stat(fp).st_size) < length: raise RuntimeError(f"fetch size incomplete, {file_size} < {length}")
  return fp

# *** Exec helpers

def cpu_time_execution(cb, enable):
  if enable: st = time.perf_counter()
  cb()
  if enable: return time.perf_counter()-st

def cpu_objdump(lib, objdump_tool='objdump'):
  with tempfile.NamedTemporaryFile(delete=True) as f:
    pathlib.Path(f.name).write_bytes(lib)
    print(subprocess.check_output([objdump_tool, '-d', f.name]).decode('utf-8'))

# *** ctypes helpers

# TODO: make this work with read only memoryviews (if possible)
def from_mv(mv:memoryview, to_type=ctypes.c_char):
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
def to_mv(ptr:int, sz:int) -> memoryview: return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")
def mv_address(mv:memoryview): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def to_char_p_p(options: List[bytes], to_type=ctypes.c_char):
  return (ctypes.POINTER(to_type) * len(options))(*[ctypes.cast(ctypes.create_string_buffer(o), ctypes.POINTER(to_type)) for o in options])
@functools.lru_cache(maxsize=None)
def init_c_struct_t(fields: Tuple[Tuple[str, ctypes._SimpleCData], ...]):
  class CStruct(ctypes.Structure):
    _pack_, _fields_ = 1, fields
  return CStruct
def init_c_var(ctypes_var, creat_cb): return (creat_cb(ctypes_var), ctypes_var)[1]
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

# *** tqdm

class tqdm:
  def __init__(self, iterable=None, desc:str='', disable:bool=False, unit:str='it', unit_scale=False, total:Optional[int]=None, rate:int=100):
    self.iterable, self.disable, self.unit, self.unit_scale, self.rate = iterable, disable, unit, unit_scale, rate
    self.st, self.i, self.n, self.skip, self.t = time.perf_counter(), -1, 0, 1, getattr(iterable, "__len__", lambda:0)() if total is None else total
    self.set_description(desc)
    self.update(0)
  def __iter__(self):
    for item in self.iterable:
      yield item
      self.update(1)
    self.update(close=True)
  def __enter__(self): return self
  def __exit__(self, *_): self.update(close=True)
  def set_description(self, desc:str): self.desc = f"{desc}: " if desc else ""
  def update(self, n:int=0, close:bool=False):
    self.n, self.i = self.n+n, self.i+1
    if self.disable or (not close and self.i % self.skip != 0): return
    prog, elapsed, ncols = self.n/self.t if self.t else 0, time.perf_counter()-self.st, shutil.get_terminal_size().columns
    if self.i/elapsed > self.rate and self.i: self.skip = max(int(self.i/elapsed)//self.rate,1)
    def HMS(t): return ':'.join(f'{x:02d}' if i else str(x) for i,x in enumerate([int(t)//3600,int(t)%3600//60,int(t)%60]) if i or x)
    def SI(x): return (f"{x/1000**int(g:=math.log(x,1000)):.{int(3-3*math.fmod(g,1))}f}"[:4].rstrip('.')+' kMGTPEZY'[int(g)].strip()) if x else '0.00'
    prog_text = f'{SI(self.n)}{f"/{SI(self.t)}" if self.t else self.unit}' if self.unit_scale else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
    est_text = f'<{HMS(elapsed/prog-elapsed) if self.n else "?"}' if self.t else ''
    it_text = (SI(self.n/elapsed) if self.unit_scale else f"{self.n/elapsed:5.2f}") if self.n else "?"
    suf = f'{prog_text} [{HMS(elapsed)}{est_text}, {it_text}{self.unit}/s]'
    sz = max(ncols-len(self.desc)-3-2-2-len(suf), 1)
    bar = '\r' + self.desc + (f'{100*prog:3.0f}%|{("█"*int(num:=sz*prog)+" ▏▎▍▌▋▊▉"[int(8*num)%8].strip()).ljust(sz," ")}| ' if self.t else '') + suf
    print(bar[:ncols+1], flush=True, end='\n'*close, file=sys.stderr)
  @classmethod
  def write(cls, s:str): print(f"\r\033[K{s}", flush=True, file=sys.stderr)

class trange(tqdm):
  def __init__(self, n:int, **kwargs): super().__init__(iterable=range(n), total=n, **kwargs)

# *** universal support for code object pickling

def _reconstruct_code(*args): return types.CodeType(*args)
def _serialize_code(code:types.CodeType):
  args = inspect.signature(types.CodeType).parameters  # NOTE: this works in Python 3.10 and up
  return _reconstruct_code, tuple(code.__getattribute__('co_'+x.replace('codestring', 'code').replace('constants', 'consts')) for x in args)
copyreg.pickle(types.CodeType, _serialize_code)

def _serialize_module(module:types.ModuleType): return importlib.import_module, (module.__name__,)
copyreg.pickle(types.ModuleType, _serialize_module)
