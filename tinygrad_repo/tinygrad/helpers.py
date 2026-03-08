from __future__ import annotations
import os, functools, platform, time, re, contextlib, operator, hashlib, pickle, sqlite3, tempfile, pathlib, string, ctypes, sys, gzip, getpass, gc
import subprocess, shutil, math, types, copyreg, inspect, importlib, decimal, itertools
from dataclasses import dataclass, field
from typing import ClassVar, Iterable, Any, TypeVar, Callable, Sequence, TypeGuard, Iterator, Generic, Generator, cast, overload

T = TypeVar("T")
U = TypeVar("U")
# NOTE: it returns int 1 if x is empty regardless of the type of x
def prod(x:Iterable[T]) -> T|int: return functools.reduce(operator.mul, x, 1)

# NOTE: helpers is not allowed to import from anything else in tinygrad
OSX, WIN = platform.system() == "Darwin", sys.platform == "win32"
CI = os.getenv("CI", "") != ""
ARCH_X86 = any(x in platform.processor() for x in ("Intel", "i386", "x86_64"))

# fix colors on Windows, https://stackoverflow.com/questions/12492810/python-how-can-i-make-the-ansi-escape-codes-to-work-also-in-windows
if WIN: os.system("")

def dedup(x:Iterable[T]): return list(dict.fromkeys(x))   # retains list order
def argfix(*x):
  if x and x[0].__class__ in (tuple, list):
    if len(x) != 1: raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return x
# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__))
def all_same(items:Sequence): return all(x == items[0] for x in items)  # works for empty input
def all_int(t: Sequence[Any]) -> TypeGuard[tuple[int, ...]]: return all(isinstance(s, int) for s in t)
def colored(st, color:str|None, background=False): # replace the termcolor library
  colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
  return f"\u001b[{10*background+60*(color.upper() == color)+30+colors.index(color.lower())}m{st}\u001b[0m" if color is not None else st
def colorize_float(x: float): return colored(f"{x:7.2f}x", 'green' if x < 0.75 else 'red' if x > 1.15 else 'yellow')
def time_to_str(t:float, w=8) -> str: return next((f"{t * d:{w}.2f}{pr}" for d,pr in [(1, "s "),(1e3, "ms")] if t > 10/d), f"{t * 1e6:{w}.2f}us")
def ansistrip(s:str): return re.sub('\x1b\\[(K|.*?m)', '', s)
def ansilen(s:str): return len(ansistrip(s))
def make_tuple(x:int|Sequence[int], cnt:int) -> tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else tuple(x)
def flatten(l:Iterable[Iterable[T]]): return [item for sublist in l for item in sublist]
def fully_flatten(l):
  if not (hasattr(l, "__len__") and hasattr(l, "__getitem__")) or isinstance(l, str): return [l]
  return [l[()]] if hasattr(l, "shape") and l.shape == () else [x for li in l for x in fully_flatten(li)]
def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
def _is_balanced(s:str) -> bool: return (d := 0, all((d := d + (c == '(') - (c == ')')) >= 0 for c in s))[1] and d == 0
def strip_parens(fst:str) -> str: return fst[1:-1] if fst[:1]=='(' and fst[-1:]==')' and _is_balanced(fst[1:-1]) else fst
def ceildiv(num, amt):
  # use (num + amt - 1) // amt when num is a UOp and non-negative to avoid C/Python division mismatch
  if hasattr(num, 'vmin') and num.vmin >= 0 and (amt > 0 if isinstance(amt, int) else amt.vmin > 0): return (num + amt - 1) // amt
  return int(ret) if isinstance((ret:=-(num//-amt)), float) else ret
def round_up(num:int, amt:int) -> int: return (num+amt-1)//amt * amt
def round_down(num:int, amt:int) -> int: return -round_up(-num, amt)
def next_power2(x): return 1 if x == 0 else 1 << (x - 1).bit_length()
# cstyle div and mod
def cdiv(x:int, y:int) -> int: return abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else 0
def cmod(x:int, y:int) -> int: return x-cdiv(x,y)*y
def lo32(x:Any) -> Any: return x & 0xFFFFFFFF # Any is sint
def hi32(x:Any) -> Any: return x >> 32 # Any is sint
def data64(data:Any) -> tuple[Any, Any]: return (data >> 32, data & 0xFFFFFFFF) # Any is sint
def data64_le(data:Any) -> tuple[Any, Any]: return (data & 0xFFFFFFFF, data >> 32) # Any is sint
def getbits(value: int, start: int, end: int): return (value >> start) & ((1 << (end - start + 1)) - 1)
def i2u(bits: int, value: int): return value if value >= 0 else (1<<bits)+value
def is_numpy_ndarray(x) -> bool: return str(type(x)) == "<class 'numpy.ndarray'>"
def merge_dicts(ds:Iterable[dict[T,U]]) -> dict[T,U]:
  kvs = set([(k,v) for d in ds for k,v in d.items()])
  if len(kvs) != len(set(kv[0] for kv in kvs)): raise RuntimeError(f"{kvs} contains different values for the same key")
  return {k:v for d in ds for k,v in d.items()}
def partition(itr:Iterable[T], fxn:Callable[[T],bool]) -> tuple[list[T], list[T]]:
  ret:tuple[list[T], list[T]] = ([], [])
  for s in itr: (ret[0] if fxn(s) else ret[1]).append(s)
  return ret
def unwrap(x:T|None) -> T:
  assert x is not None
  return x
def get_single_element(x:Sequence[T]) -> T:
  assert len(x) == 1, f"{x} must only have 1 element"
  return x[0]
def get_child(obj, key):
  for k in key.split('.'):
    if k.isnumeric(): obj = obj[int(k)]
    elif isinstance(obj, dict): obj = obj[k]
    else: obj = getattr(obj, k)
  return obj
def word_wrap(x, wrap=80):
  if len(ansistrip(x)) <= wrap: return x
  if len(lines:=x.splitlines()) > 1: return "\n".join(word_wrap(line, wrap) for line in lines)
  i = 0
  while len(ansistrip(x[:i])) < wrap and i < len(x): i += 1
  return x[:i] + "\n" + word_wrap(x[i:], wrap)
def pad_bytes(b:bytes, align:int) -> bytes: return b + b'\x00' * ((align - (len(b) % align)) % align)
def panic(e:Exception|None=None): raise e if e is not None else RuntimeError("PANIC!")

@functools.cache
def canonicalize_strides(shape:tuple[T, ...], strides:tuple[T, ...]) -> tuple[T, ...]:
  return tuple(cast(T, 0) if s == 1 else st for s, st in zip(shape, strides))

@functools.cache
def strides_for_shape(shape:tuple[T, ...]) -> tuple[T, ...]:
  if not shape: return ()
  strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))[::-1]
  return canonicalize_strides(shape, strides)

# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
def get_contraction(old_shape:tuple[T, ...], new_shape:tuple[T, ...]) -> list[list[int]]|None: # T is sint
  acc_old, acc_new = list(itertools.accumulate(old_shape, operator.mul)), list(itertools.accumulate(new_shape, operator.mul))
  try: split = [acc_old.index(acc)+1 if acc != 1 else 0 for acc in acc_new]
  except ValueError: return None
  return [list(range(st,ed)) for st,ed in zip([0]+split[:-1], split[:-1]+[len(old_shape)])]

def suppress_finalizing(func):
  def wrapper(*args, **kwargs):
    try: return func(*args, **kwargs)
    except (RuntimeError, AttributeError, TypeError, ImportError):
      if not getattr(sys, 'is_finalizing', lambda: True)(): raise # re-raise if not finalizing
  return wrapper

def select_first_inited(candidates:Sequence[Callable[...,T]|Sequence[Callable[...,T]|None]], err_msg:str, cache:dict|None=None):
  excs = []
  for typ in candidates:
    if cache is not None and typ in cache: return cache[typ]
    try:
      x = tuple([cast(Callable, t)() if t is not None else None for t in typ]) if isinstance(typ, Sequence) else cast(Callable, typ)()
      if cache is not None: cache[typ] = x
      return x
    except Exception as e: excs.append(e)
  raise ExceptionGroup(err_msg, excs)

def unwrap_class_type(cls_t): return cls_t.func if isinstance(cls_t, functools.partial) else cls_t

def pluralize(st:str, cnt:int): return f"{cnt} {st}"+('' if cnt == 1 else 's')

# for length N coefficients `p`, returns p[0] * x**(N-1) + p[1] * x**(N-2) + ... + p[-2] * x + p[-1]
def polyN(x:T, p:list[float]) -> T: return functools.reduce(lambda acc,c: acc*x+c, p, 0.0)  # type: ignore

@functools.cache
def to_function_name(s:str): return ''.join([c if c in (string.ascii_letters+string.digits+'_') else f'{ord(c):02X}' for c in ansistrip(s)])
@overload
def getenv(key:str) -> int: ...
@overload
def getenv(key:str, default:T) -> T: ...
@functools.cache
def getenv(key:str, default:Any=0): return type(default)(os.getenv(key, default))

def temp(x:str, append_user:bool=False) -> str:
  return (pathlib.Path(tempfile.gettempdir()) / (f"{x}.{getpass.getuser()}" if append_user else x)).as_posix()

def stderr_log(msg:str): print(msg, end='', file=sys.stderr, flush=True)

class Context(contextlib.ContextDecorator):
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    self.old_context:dict[str, Any] = {k: ContextVar._cache[k].value for k in self.kwargs}
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v
  def __exit__(self, *args):
    for k,v in self.old_context.items(): ContextVar._cache[k].value = v

class ContextVar(Generic[T]):
  _cache: ClassVar[dict[str, ContextVar]] = {}
  value: T
  key: str
  def __init__(self, key: str, default_value: T):
    if key in ContextVar._cache: raise RuntimeError(f"attempt to recreate ContextVar {key}")
    ContextVar._cache[key] = self
    self.value, self.key = getenv(key, default_value), key
  def __bool__(self): return bool(self.value)
  def __eq__(self, x): return self.value == x
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x
  def tolist(self, obj=None):
    assert isinstance(self.value, str)
    return [getattr(obj, x) if obj else x for x in self.value.split(',') if x]

DEBUG, IMAGE, BEAM, NOOPT = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0), ContextVar("BEAM", 0), ContextVar("NOOPT", 0)
JIT, JIT_BATCH_SIZE = ContextVar("JIT", 2 if OSX and ARCH_X86 else 1), ContextVar("JIT_BATCH_SIZE", 32)
WINO, CAPTURING, TRACEMETA = ContextVar("WINO", 0), ContextVar("CAPTURING", 1), ContextVar("TRACEMETA", 1)
USE_TC, TC_SELECT, TC_OPT, AMX = ContextVar("TC", 1), ContextVar("TC_SELECT", -1), ContextVar("TC_OPT", 0), ContextVar("AMX", 0)
TRANSCENDENTAL, NOLOCALS = ContextVar("TRANSCENDENTAL", 1), ContextVar("NOLOCALS", 0)
SPLIT_REDUCEOP, NO_MEMORY_PLANNER, LRU = ContextVar("SPLIT_REDUCEOP", 1), ContextVar("NO_MEMORY_PLANNER", 0), ContextVar("LRU", 1)
RING, ALL2ALL = ContextVar("RING", 1), ContextVar("ALL2ALL", 0)
CACHELEVEL, IGNORE_BEAM_CACHE, DEVECTORIZE = ContextVar("CACHELEVEL", 2), ContextVar("IGNORE_BEAM_CACHE", 0), ContextVar("DEVECTORIZE", 1)
VALIDATE_WITH_CPU, DISABLE_FAST_IDIV = ContextVar("VALIDATE_WITH_CPU", 0), ContextVar("DISABLE_FAST_IDIV", 0)
CORRECT_DIVMOD_FOLDING, FUSE_OPTIM = ContextVar("CORRECT_DIVMOD_FOLDING", 0), ContextVar("FUSE_OPTIM", 0)
ALLOW_DEVICE_USAGE, MAX_BUFFER_SIZE = ContextVar("ALLOW_DEVICE_USAGE", 1), ContextVar("MAX_BUFFER_SIZE", 0)
EMULATE, EMULATED_DTYPES = ContextVar("EMULATE", ""), ContextVar("EMULATED_DTYPES", "")
CPU_COUNT = ContextVar("CPU_COUNT", max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)))
# Compilers
CPU_LLVM, CPU_LVP, AMD_LLVM = ContextVar("CPU_LLVM", 0), ContextVar("CPU_LVP", 0), ContextVar("AMD_LLVM", 0)
NV_PTX, CUDA_PTX, NV_NAK, QCOM_IR3 = ContextVar("NV_PTX", 0), ContextVar("CUDA_PTX", 0), ContextVar("NV_NAK", 0), ContextVar("QCOM_IR3", 0)
NULL_IR3, NULL_NAK, NULL_ALLOW_COPYOUT = ContextVar("NULL_IR3", 0), ContextVar("NULL_NAK", 0), ContextVar("NULL_ALLOW_COPYOUT", 0)
AMD_CC, CPU_CC, NV_CC, CUDA_CC = ContextVar("AMD_CC", ""), ContextVar("CPU_CC", ""), ContextVar("NV_CC", ""), ContextVar("CUDA_CC", "")
QCOM_CC = ContextVar("QCOM_CC", "")
# VIZ implies PROFILE, but you can run PROFILE without VIZ
VIZ = ContextVar("VIZ", 0)
PROFILE = ContextVar("PROFILE", abs(VIZ.value))
SPEC = ContextVar("SPEC", 1)
# TODO: disable by default due to speed
CHECK_OOB = ContextVar("CHECK_OOB", 0)
PCONTIG = ContextVar("PCONTIG", 0)  # partial contiguous in rangeify
DEBUG_RANGEIFY = ContextVar("DEBUG_RANGEIFY", 0)
# set to 1, this uses tuplize in the linearizer sort order
TUPLE_ORDER = ContextVar("TUPLE_ORDER", 1)
# set to 0 to disable the compiler cache
CCACHE = ContextVar("CCACHE", 1)
# allow tf32 to be used on NVIDIA GPUs
ALLOW_TF32 = ContextVar("ALLOW_TF32", 0)
# set to 0 to disable the scheduler cache
SCACHE = ContextVar("SCACHE", 1)
# allow use of atomics for embedding backward
USE_ATOMICS = ContextVar("USE_ATOMICS", 0)
# allow use of assembly for gemm
ASM_GEMM = ContextVar("ASM_GEMM", 0)

@dataclass(frozen=True)
class Metadata:
  name: str
  caller: str
  backward: bool = False
  def __hash__(self): return hash(self.name)
  def __str__(self): return self.name + (" bw" if self.backward else "")

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

def perf_counter_us() -> decimal.Decimal: return decimal.Decimal(time.perf_counter_ns())/1000

@functools.cache
def lines(fn) -> list[str]:
  try:
    with open(fn, encoding="utf-8") as f: return f.readlines()
  except (FileNotFoundError, OSError): return []

def printable(loc:tuple[str, int]) -> str:
  try: return lines(loc[0])[loc[1]-1].strip()
  except IndexError: return "<missing>"

def get_stacktrace(frm, max_frames=30) -> tuple[tuple, ...]:
  ret:list[tuple] = []
  for i in range(max_frames):
    if (frm:=frm.f_back) is None: break
    ret.append(((fc:=frm.f_code).co_filename, frm.f_lineno, fc.co_name, printable((fc.co_filename, frm.f_lineno))))
  return tuple(ret)

@dataclass(frozen=True)
class TracingKey:
  display_name:str                       # display name of this trace event
  keys:tuple[Any, ...]=()                # optional keys to search for related traces
  ret:Any=None
  tb:tuple[tuple, ...]|None=field(default_factory=lambda: get_stacktrace(sys._getframe(1)) if VIZ else None)

class ProfileEvent: pass

@dataclass
class ProfileRangeEvent(ProfileEvent):
  device:str; name:str|TracingKey; st:decimal.Decimal; en:decimal.Decimal|None=None; is_copy:bool=False # noqa: E702

@dataclass(frozen=True)
class ProfilePointEvent(ProfileEvent):
  device:str; name:str; key:Any; arg:dict=field(default_factory=dict); ts:decimal.Decimal=field(default_factory=perf_counter_us) # noqa: E702

cpu_events:list[ProfileEvent] = []
@contextlib.contextmanager
def cpu_profile(name:str|TracingKey, device="TINY", is_copy=False, display=True) -> Generator[ProfileRangeEvent, None, None]:
  res = ProfileRangeEvent(device, name, perf_counter_us(), is_copy=is_copy)
  try: yield res
  finally:
    res.en = perf_counter_us()
    if PROFILE and display: cpu_events.append(res)

def profile_marker(name:str, color="gray") -> None:
  cpu_events.append(ProfilePointEvent("TINY", "marker", None, {"name":name, "color":color}))

if getenv("DEBUG_GC"):
  gc_start: decimal.Decimal = perf_counter_us()
  def my_gc_callback(phase, info):
    global gc_start
    if phase == 'start': gc_start = perf_counter_us()
    elif phase == "stop":
      cpu_events.append(ProfileRangeEvent("GC", f"collected: {info['collected']} (gen {info['generation']})", gc_start, perf_counter_us()))
  if PROFILE: gc.callbacks.append(my_gc_callback)

# *** universal database cache ***

cache_dir: str = os.path.join(getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache")), "tinygrad")
CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(cache_dir, "cache.db")))

VERSION = 22
_db_connection = None
def db_connection():
  global _db_connection
  if _db_connection is None:
    os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
    _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")
    # another connection has set it already or is in the process of setting it
    # that connection will lock the database
    with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
    if DEBUG >= 8: _db_connection.set_trace_callback(print)
  return _db_connection

def diskcache_clear():
  cur = db_connection().cursor()
  drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
  cur.executescript("\n".join([s[0] for s in drop_tables] + ["VACUUM;"]))

def diskcache_get(table:str, key:dict|str|int) -> Any:
  if CACHELEVEL < 1: return None
  if isinstance(key, (str,int)): key = {"key": key}
  cur = db_connection().cursor()
  try:
    res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  except sqlite3.OperationalError:
    return None  # table doesn't exist
  if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
  return None

_db_tables = set()
def diskcache_put(table:str, key:dict|str|int, val:Any, prepickled=False):
  if CACHELEVEL < 1: return val
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  if table not in _db_tables:
    TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
    ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}_{VERSION}' ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
    _db_tables.add(table)
  cur.execute(f"REPLACE INTO '{table}_{VERSION}' ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key))}, ?)",
              tuple(key.values()) + (val if prepickled else pickle.dumps(val),))
  conn.commit()
  cur.close()
  return val

def diskcache(func:Callable[..., T]):
  def wrapper(*args, **kwargs) -> T:
    table, key = f"cache_{func.__name__}", hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
    if (ret:=diskcache_get(table, key)) is not None: return ret
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

def fetch(url:str, name:pathlib.Path|str|None=None, subdir:str|None=None, gunzip:bool=False,
          allow_caching=not getenv("DISABLE_HTTP_CACHE"), headers:dict[str, str]={}) -> pathlib.Path:
  import urllib.request
  if url.startswith(("/", ".")): return pathlib.Path(url)
  if name is not None and (isinstance(name, pathlib.Path) or '/' in name): fp = pathlib.Path(name)
  else:
    hh = "_"+hashlib.md5(("\n".join(f"{k.strip()}:{v.strip()}" for k,v in sorted(headers.items()))).encode("utf-8")).hexdigest() if headers else ""
    fp = _ensure_downloads_dir() / (subdir or "") / ((name or hashlib.md5(url.encode('utf-8')).hexdigest()) + hh + (".gunzip" if gunzip else ""))
  if not fp.is_file() or not allow_caching:
    (_dir := fp.parent).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tinygrad 0.12.0", **headers}), timeout=10) as r:
      assert r.status in {200, 206}, r.status
      length = int(r.headers.get('content-length', 0)) if not gunzip else None
      readfile = gzip.GzipFile(fileobj=r) if gunzip else r
      progress_bar:tqdm = tqdm(total=length, unit='B', unit_scale=True, desc=f"{url}", disable=CI)
      with tempfile.NamedTemporaryFile(dir=_dir, delete=False) as f:
        while chunk := readfile.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        pathlib.Path(f.name).rename(fp)
      progress_bar.update(close=True)
      if length and (file_size:=os.stat(fp).st_size) < length: raise RuntimeError(f"fetch size incomplete, {file_size} < {length}")
  return fp

# *** Exec helpers

def system(cmd:str, **kwargs) -> str:
  st = time.perf_counter()
  ret = subprocess.check_output(cmd.split(), **kwargs).decode().strip()
  if DEBUG >= 1: print(f"system: '{cmd}' returned {len(ret)} bytes in {(time.perf_counter() - st)*1e3:.2f} ms")
  return ret

def cpu_objdump(lib, objdump_tool='objdump'):
  with tempfile.NamedTemporaryFile(delete=True) as f:
    pathlib.Path(f.name).write_bytes(lib)
    print(system(f"{objdump_tool} -d {f.name}"))

def capstone_flatdump(lib: bytes):
  try: import capstone
  except ImportError:
    print("Disassembler Error: Capstone not installed.")
    return
  match platform.machine():
    case 'x86_64' | 'AMD64': cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    case 'aarch64' | 'arm64': cs = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    case machine: raise NotImplementedError(f"Capstone disassembly isn't supported for {machine}")
  cs.skipdata = True
  for instr in cs.disasm(lib, 0):
    print(f"{instr.address:#08x}: {instr.mnemonic}\t{instr.op_str}")
  sys.stdout.flush()

def wait_cond(cb, *args, value=True, timeout_ms=10000, msg="") -> bool:
  start_time = int(time.perf_counter() * 1000)
  while int(time.perf_counter() * 1000) - start_time < timeout_ms:
    if (val:=cb(*args)) == value: return val
  raise TimeoutError(f"{msg}. Timed out after {timeout_ms} ms, condition not met: {val} != {value}")

# *** ctypes helpers

# TODO: make this work with read only memoryviews (if possible)
def from_mv(mv:memoryview, to_type:type[ctypes._SimpleCData]=ctypes.c_char) -> ctypes.Array:
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
def to_mv(ptr:int, sz:int) -> memoryview: return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")
def mv_address(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def to_char_p_p(options: list[bytes], to_type=ctypes.c_char):
  return (ctypes.POINTER(to_type) * len(options))(*[ctypes.cast(ctypes.create_string_buffer(o), ctypes.POINTER(to_type)) for o in options])
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

# *** tqdm

class tqdm(Generic[T]):
  def __init__(self, iterable:Iterable[T]|None=None, desc:str='', disable:bool=False,
               unit:str='it', unit_scale=False, total:int|None=None, rate:int=100):
    self.iterable, self.disable, self.unit, self.unit_scale, self.rate = iterable, disable, unit, unit_scale, rate
    self.st, self.i, self.n, self.skip, self.t = time.perf_counter(), -1, 0, 1, getattr(iterable, "__len__", lambda:0)() if total is None else total
    self.set_description(desc)
    self.update(0)
  def __iter__(self) -> Iterator[T]:
    assert self.iterable is not None, "need an iterable to iterate"
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
    if elapsed and self.i/elapsed > self.rate and self.i: self.skip = max(int(self.i/elapsed)//self.rate,1)
    def HMS(t): return ':'.join(f'{x:02d}' if i else str(x) for i,x in enumerate([int(t)//3600,int(t)%3600//60,int(t)%60]) if i or x)
    def SI(x):
      if not x: return '0.00'
      v = f"{x/1000**int(g:=round(math.log(x,1000),6)):.{int(3-3*math.fmod(g,1))}f}"[:4].rstrip('.')
      return (f"{x/1000**(int(g)+1):.3f}"[:4].rstrip('.')+' kMGTPEZY'[int(g)+1]) if v == "1000" else v+' kMGTPEZY'[int(g)].strip()
    prog_text = f'{SI(self.n)}{f"/{SI(self.t)}" if self.t else self.unit}' if self.unit_scale else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
    est_text = f'<{HMS(elapsed/prog-elapsed) if self.n else "?"}' if self.t else ''
    it_text = (SI(self.n/elapsed) if self.unit_scale else f"{self.n/elapsed:5.2f}") if self.n else "?"
    suf = f'{prog_text} [{HMS(elapsed)}{est_text}, {it_text}{self.unit}/s]'
    sz = max(ncols-len(self.desc)-3-2-2-len(suf), 1)
    bar = '\r' + self.desc + (f'{100*prog:3.0f}%|{("█"*int(num:=sz*prog)+" ▏▎▍▌▋▊▉"[int(8*num)%8].strip()).ljust(sz," ")}| ' if self.t else '') + suf
    print(bar[:ncols+1], flush=True, end='\n'*close, file=sys.stderr)
  @classmethod
  def write(cls, s:str): print(f"\r\033[K{s}", flush=True, file=sys.stderr)

def trange(n:int, **kwargs) -> tqdm[int]: return tqdm(range(n), total=n, **kwargs)

class disable_gc(contextlib.ContextDecorator):
  def __enter__(self):
    self._was_enabled = gc.isenabled()
    if self._was_enabled: gc.disable()
  def __exit__(self, *exc):
    if self._was_enabled: gc.enable()

# *** universal support for code object pickling

def _reconstruct_code(*args): return types.CodeType(*args)
def _serialize_code(code:types.CodeType):
  args = inspect.signature(types.CodeType).parameters  # NOTE: this works in Python 3.10 and up
  return _reconstruct_code, tuple(code.__getattribute__('co_'+x.replace('codestring', 'code').replace('constants', 'consts')) for x in args)
copyreg.pickle(types.CodeType, _serialize_code)

def _serialize_module(module:types.ModuleType): return importlib.import_module, (module.__name__,)
copyreg.pickle(types.ModuleType, _serialize_module)

class count:
  def __init__(self, start:int=0, step:int=1):
    self.n, self.step = start, step
  def __next__(self) -> int:
    cur = self.n
    self.n += self.step
    return cur
