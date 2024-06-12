from __future__ import annotations
import os, functools, platform, time, re, contextlib, operator, hashlib, pickle, sqlite3
import numpy as np
from typing import Dict, Tuple, Union, List, NamedTuple, Final, Iterator, ClassVar, Optional, Iterable, Any, TypeVar, TYPE_CHECKING
if TYPE_CHECKING:  # TODO: remove this and import TypeGuard from typing once minimum python supported version is 3.10
  from typing_extensions import TypeGuard

T = TypeVar("T")
# NOTE: it returns int 1 if x is empty regardless of the type of x
def prod(x:Iterable[T]) -> Union[T,int]: return functools.reduce(operator.__mul__, x, 1)

# NOTE: helpers is not allowed to import from anything else in tinygrad
OSX = platform.system() == "Darwin"
CI = os.getenv("CI", "") != ""

def dedup(x): return list(dict.fromkeys(x))   # retains list order
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items): return all(x == items[0] for x in items)
def all_int(t: Tuple[Any, ...]) -> TypeGuard[Tuple[int, ...]]: return all(isinstance(s, int) for s in t)
def colored(st, color, background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line
def ansistrip(s): return re.sub('\x1b\\[(K|.*?m)', '', s)
def ansilen(s): return len(ansistrip(s))
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Union[List, Iterator]): return [item for sublist in l for item in sublist]
def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
def strip_parens(fst): return fst[1:-1] if fst[0] == '(' and fst[-1] == ')' and fst[1:-1].find('(') <= fst[1:-1].find(')') else fst
def merge_dicts(ds:Iterable[Dict]) -> Dict:
  assert len(kvs:=set([(k,v) for d in ds for k,v in d.items()])) == len(set(kv[0] for kv in kvs)), f"cannot merge, {kvs} contains different values for the same key"
  return {k:v for d in ds for k,v in d.items()}
def partition(lst, fxn):
  a: list[Any] = []
  b: list[Any] = []
  for s in lst: (a if fxn(s) else b).append(s)
  return a,b

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

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
  def __new__(cls, key, default_value):
    if key in ContextVar._cache: return ContextVar._cache[key]
    instance = ContextVar._cache[key] = super().__new__(cls)
    instance.value = getenv(key, default_value)
    return instance
  def __bool__(self): return bool(self.value)
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

DEBUG, IMAGE, BEAM, NOOPT = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0), ContextVar("BEAM", 0), ContextVar("NOOPT", 0)
GRAPH, GRAPHPATH = getenv("GRAPH", 0), getenv("GRAPHPATH", "/tmp/net")

class Timing(contextlib.ContextDecorator):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.et = time.perf_counter_ns() - self.st
    if self.enabled: print(f"{self.prefix}{self.et*1e-6:.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

# **** tinygrad now supports dtypes! *****

class DType(NamedTuple):
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  sz: int = 1
  def __repr__(self): return f"dtypes.{INVERSE_DTYPES_DICT[self]}"

# dependent typing?
class ImageDType(DType):
  def __new__(cls, priority, itemsize, name, np, shape):
    return super().__new__(cls, priority, itemsize, name, np)
  def __init__(self, priority, itemsize, name, np, shape):
    self.shape: Tuple[int, ...] = shape  # arbitrary arg for the dtype, used in image for the shape
    super().__init__()
  def __repr__(self): return f"dtypes.{self.name}({self.shape})"
  # TODO: fix this to not need these
  def __hash__(self): return hash((super().__hash__(), self.shape))
  def __eq__(self, x): return super().__eq__(x) and self.shape == x.shape
  def __ne__(self, x): return super().__ne__(x) or self.shape != x.shape

class PtrDType(DType):
  def __new__(cls, dt:DType): return super().__new__(cls, dt.priority, dt.itemsize, dt.name, dt.np, dt.sz)
  def __repr__(self): return f"ptr.{super().__repr__()}"

class dtypes:
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType)-> bool: return x in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def is_float(x: DType) -> bool: return x in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes._half4, dtypes._float2, dtypes._float4)
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]
  @staticmethod
  def fields() -> Dict[str, DType]: return DTYPES_DICT
  bool: Final[DType] = DType(0, 1, "bool", np.bool_)
  float16: Final[DType] = DType(0, 2, "half", np.float16)
  half = float16
  float32: Final[DType] = DType(4, 4, "float", np.float32)
  float = float32
  float64: Final[DType] = DType(0, 8, "double", np.float64)
  double = float64
  int8: Final[DType] = DType(0, 1, "char", np.int8)
  int16: Final[DType] = DType(1, 2, "short", np.int16)
  int32: Final[DType] = DType(2, 4, "int", np.int32)
  int64: Final[DType] = DType(3, 8, "long", np.int64)
  uint8: Final[DType] = DType(0, 1, "unsigned char", np.uint8)
  uint16: Final[DType] = DType(1, 2, "unsigned short", np.uint16)
  uint32: Final[DType] = DType(2, 4, "unsigned int", np.uint32)
  uint64: Final[DType] = DType(3, 8, "unsigned long", np.uint64)

  # NOTE: bfloat16 isn't supported in numpy
  bfloat16: Final[DType] = DType(0, 2, "__bf16", None)

  # NOTE: these are internal dtypes, should probably check for that
  _int2: Final[DType] = DType(2, 4*2, "int2", None, 2)
  _half4: Final[DType] = DType(0, 2*4, "half4", None, 4)
  _float2: Final[DType] = DType(4, 4*2, "float2", None, 2)
  _float4: Final[DType] = DType(4, 4*4, "float4", None, 4)
  _arg_int32: Final[DType] = DType(2, 4, "_arg_int32", None)

  # NOTE: these are image dtypes
  @staticmethod
  def imageh(shp): return ImageDType(100, 2, "imageh", np.float16, shp)
  @staticmethod
  def imagef(shp): return ImageDType(100, 4, "imagef", np.float32, shp)

# HACK: staticmethods are not callable in 3.8 so we have to compare the class
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}
INVERSE_DTYPES_DICT = {v:k for k,v in DTYPES_DICT.items()}

class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  mem_cached: ClassVar[int] = 0 # NOTE: this is not reset
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count = 0,0,0.0,0

# *** universal database cache ***

CACHEDB = getenv("CACHEDB", "/tmp/tinygrad_cache")
CACHELEVEL = getenv("CACHELEVEL", 2)

VERSION = 6
_db_connection = None
def db_connection():
  global _db_connection
  if _db_connection is None:
    _db_connection = sqlite3.connect(CACHEDB)
    if DEBUG >= 5: _db_connection.set_trace_callback(print)
    if diskcache_get("meta", "version") != VERSION:
      print("cache is out of date, clearing it")
      os.unlink(CACHEDB)
      _db_connection = sqlite3.connect(CACHEDB)
      if DEBUG >= 5: _db_connection.set_trace_callback(print)
      diskcache_put("meta", "version", VERSION)
  return _db_connection

def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
  if isinstance(key, (str,int)): key = {"key": key}
  try:
    res = db_connection().cursor().execute(f"SELECT val FROM {table} WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  except sqlite3.OperationalError:
    return None  # table doesn't exist
  if (val:=res.fetchone()) is not None:
    return pickle.loads(val[0])
  return None

_db_tables = set()
def diskcache_put(table:str, key:Union[Dict, str, int], val:Any):
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  if table not in _db_tables:
    TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
    ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
    _db_tables.add(table)
  cur.execute(f"REPLACE INTO {table} ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key.keys()))}, ?)", tuple(key.values()) + (pickle.dumps(val), ))
  conn.commit()
  cur.close()
  return val

def diskcache(func):
  def wrapper(*args, **kwargs) -> bytes:
    table, key = f"cache_{func.__name__}", hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
    if (ret:=diskcache_get(table, key)): return ret
    return diskcache_put(table, key, func(*args, **kwargs))
  setattr(wrapper, "__wrapped__", func)
  return wrapper
