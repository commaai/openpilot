from __future__ import annotations
import ctypes, functools, os, pathlib, re, sys, sysconfig
from tinygrad.helpers import ceildiv, getenv, unwrap, DEBUG, OSX, WIN
from _ctypes import Array as _CArray, _SimpleCData, _Pointer
from typing import TYPE_CHECKING, get_type_hints, get_args, get_origin, overload, Annotated, Any, Generic, Iterable, ParamSpec, TypeVar

def _do_ioctl(__idir, __base, __nr, __struct, __fd, *args, __payload=None, **kwargs):
  assert not WIN, "ioctl not supported"
  import tinygrad.runtime.support.hcq as hcq, fcntl
  ioctl = __fd.ioctl if isinstance(__fd, hcq.FileIOInterface) else functools.partial(fcntl.ioctl, __fd)
  if (rc:=ioctl((__idir<<30)|(ctypes.sizeof(out:=(__payload or __struct(*args, **kwargs)))<<16)|(__base<<8)|__nr, out)):
    raise RuntimeError(f"ioctl returned {rc}")
  return out

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, typ): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, del_an(typ))
def _IOR(base, nr, typ): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, del_an(typ))
def _IOWR(base, nr, typ): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, del_an(typ))

def del_an(ty):
  if isinstance(ty, type) and issubclass(ty, Enum): return del_an(ty.__orig_bases__[0]) # type: ignore
  return ty.__metadata__[0] if get_origin(ty) is Annotated else (None if ty is type(None) else ty)

_pending_records = []

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
P = ParamSpec("P")

if TYPE_CHECKING:
  from ctypes import _CFunctionType
  from _ctypes import _CData
  class Array(Generic[T, U], _CData):
    @overload
    def __getitem__(self: Array[_SimpleCData[V], Any], key: int) -> V: ...
    @overload
    def __getitem__(self: Array[T, Any], key: int) -> T: ...
    def __getitem__(self, key) -> Any: ...
    @overload
    def __setitem__(self: Array[_SimpleCData[V], Any], key: int, val: V): ...
    @overload
    def __setitem__(self: Array[T, Any], key: int, val: T): ...
    @overload
    def __setitem__(self: Array[T, Any], key: slice, val: Iterable[T]): ...
    def __setitem__(self, key, val): ...
  class POINTER(Generic[T], _Pointer): ...
  class CFUNCTYPE(Generic[T, P], _CFunctionType): ...
  class Enum(_SimpleCData):
    @classmethod
    def get(cls, val:int, default="unknown") -> str: ...
    @classmethod
    def items(cls) -> Iterable[tuple[int,str]]: ...
    @classmethod
    def define(cls, name:str, val:int) -> int: ...
  CT = TypeVar("CT", bound=_CData)
  def pointer(obj: CT) -> POINTER[CT]: ...
else:
  class _Array:
    def __getitem__(self, key): return del_an(key[0]) * get_args(key[1])[0]
    def __call__(self, ty, l): return del_an(ty) * l
  Array = _Array()
  class POINTER:
    def __class_getitem__(cls, key): return ctypes.POINTER(del_an(key))
  class CFUNCTYPE:
    def __class_getitem__(cls, key): return ctypes.CFUNCTYPE(del_an(key[0]), *(del_an(a) for a in key[1]))
  class Enum:
    def __init_subclass__(cls): cls._val_to_name_ = {}

    @classmethod
    def get(cls, val, default="unknown"): return cls._val_to_name_.get(val, default)
    @classmethod
    def items(cls): return cls._val_to_name_.items()
    @classmethod
    def define(cls, name:str, val:int) -> int:
      cls._val_to_name_[val] = name
      return val
  def pointer(obj): return ctypes.pointer(obj)

def i2b(i:int, sz:int) -> bytes: return i.to_bytes(sz, sys.byteorder)
def b2i(b:bytes) -> int: return int.from_bytes(b, sys.byteorder)
def mv(st) -> memoryview: return memoryview(st).cast('B')

class Struct(ctypes.Structure):
  def __init__(self, *args, **kwargs):
    ctypes.Structure.__init__(self)
    self._objects_ = {}
    for f,v in [*zip((rf[0] for rf in self._real_fields_), args), *kwargs.items()]: setattr(self, f, v)

def record(cls) -> type[Struct]:
  struct = type(cls.__name__, (Struct,), {'_fields_': [('_mem_', ctypes.c_byte * cls.SIZE)]})
  _pending_records.append((cls, struct, unwrap(sys._getframe().f_back).f_globals))
  return struct

def init_records() -> None:
  for cls, struct, ns in _pending_records:
    setattr(struct, '_real_fields_', [])
    for nm, t in get_type_hints(cls, globalns=ns, include_extras=True).items():
      if t.__origin__ in (bool, bytes, str, int, float): setattr(struct, nm, Field(*(f:=t.__metadata__)))
      else: setattr(struct, nm, Field(*(f:=(del_an(t.__origin__), *t.__metadata__))))
      struct._real_fields_.append((nm,) + f) # type: ignore
  _pending_records.clear()

class Field(property):
  def __init__(self, typ, off:int, bit_width=None, bit_off=0):
    if bit_width is not None:
      sl, set_mask = slice(off,off+(sz:=ceildiv(bit_width+bit_off, 8))), ~((mask:=(1 << bit_width) - 1) << bit_off)
      # FIXME: signedness
      super().__init__(lambda self: (b2i(mv(self)[sl]) >> bit_off) & mask,
                       lambda self,v: mv(self).__setitem__(sl, i2b((b2i(mv(self)[sl]) & set_mask) | (v << bit_off), sz)))
    else:
      sl = slice(off, off + ctypes.sizeof(typ))
      def set_with_objs(f):
        def wrapper(self, v):
          if hasattr(v, '_objects') and hasattr(self, '_objects_'): self._objects_[off] = {'_self_': v, **(v._objects or {})}
          mv(self).__setitem__(sl, bytes(v if isinstance(v, typ) else f(v)))
        return wrapper
      if issubclass(typ, _CArray):
        getter = (lambda self: typ.from_buffer(mv(self)[sl]).value) if typ._type_ is ctypes.c_char else (lambda self: typ.from_buffer(mv(self)[sl]))
        super().__init__(getter, set_with_objs(lambda v: typ(*v)))
      else: super().__init__(lambda self: v.value if isinstance(v:=typ.from_buffer(mv(self)[sl]), _SimpleCData) else v, set_with_objs(typ))
    self.offset = off

@functools.cache
def init_c_struct_t(sz:int, fields: tuple[tuple, ...]):
  CStruct = type("CStruct", (Struct,), {'_fields_': [('_mem_', ctypes.c_byte * sz)], '_real_fields_': []})
  for nm,ty,*args in fields:
    setattr(CStruct, nm, Field(*(f:=(del_an(ty), *args))))
    CStruct._real_fields_.append((nm,) + f) # type: ignore
  return CStruct
def init_c_var(ty, creat_cb): return (creat_cb(v:=del_an(ty)()), v)[1]

class DLL(ctypes.CDLL):
  _loaded_: set[str] = set()

  @staticmethod
  def findlib(nm:str, paths:list[str], extra_paths=[]):
    if nm == 'libc' and OSX: return '/usr/lib/libc.dylib'
    if pathlib.Path(path:=getenv(nm.replace('-', '_').upper()+"_PATH", '')).is_file(): return path
    for p in paths:
      libpaths = {"posix": ["/usr/lib64", "/usr/lib", "/usr/local/lib"], "nt": os.environ['PATH'].split(os.pathsep),
                  "darwin": ["/opt/homebrew/lib", f"/System/Library/Frameworks/{p}.framework", f"/System/Library/PrivateFrameworks/{p}.framework"],
                  'linux': ['/lib', '/lib64', f"/lib/{sysconfig.get_config_var('MULTIARCH')}", "/usr/lib/wsl/lib/"]}
      if (pth:=pathlib.Path(p)).is_absolute():
        if pth.is_file(): return p
        else: continue
      for pre in (pathlib.Path(pre) for pre in ([path] if path else []) + libpaths.get(os.name, []) + libpaths.get(sys.platform, []) + extra_paths):
        if not pre.is_dir(): continue
        if WIN or OSX:
          for base in ([f"lib{p}.dylib", f"{p}.dylib", str(p)] if OSX else [f"{p}.dll"]):
            if (l:=pre / base).is_file() or (OSX and 'framework' in str(l) and l.is_symlink()): return str(l)
        else:
          for l in (l for l in pre.iterdir() if l.is_file() and re.fullmatch(f"lib{p}\\.so\\.?[0-9]*", l.name)):
            # filter out linker scripts
            with open(l, 'rb') as f:
              if f.read(4) == b'\x7FELF': return str(l)

  def __init__(self, nm:str, paths:str|list[str], extra_paths=[], emsg="", **kwargs):
    self.nm, self.emsg = nm, emsg
    if (path:= DLL.findlib(nm, paths if isinstance(paths, list) else [paths], extra_paths if isinstance(extra_paths, list) else [extra_paths])):
      if DEBUG >= 3: print(f"loading {nm} from {path}")
      try:
        super().__init__(path, **kwargs)
        self._loaded_.add(self.nm)
      except OSError as e:
        self.emsg = str(e)
        if DEBUG >= 3: print(f"loading {nm} failed: {e}")
    elif DEBUG >= 3: print(f"loading {nm} failed: not found on system")

  def bind(self, fn):
    restype, argtypes = del_an((hints:=get_type_hints(fn, include_extras=True)).pop('return', None)), tuple(del_an(h) for h in hints.values())
    cfunc = None
    def wrapper(*args):
      nonlocal cfunc
      if cfunc is None: (cfunc:=getattr(self, fn.__name__)).argtypes, cfunc.restype = argtypes, restype
      return cfunc(*args)
    return wrapper

  def __getattr__(self, nm):
    if self.nm not in self._loaded_:
      raise AttributeError(f"failed to load library {self.nm}: " + (self.emsg or f"try setting {self.nm.upper()+'_PATH'}?"))
    return super().__getattr__(nm)
