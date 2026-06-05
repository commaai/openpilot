from __future__ import annotations
import ctypes, functools, os, pathlib, re, sys, sysconfig
from tinygrad.helpers import ceildiv, getenv, DEBUG, OSX, WIN
from typing import TYPE_CHECKING, get_args, Generic, ParamSpec, TypeVar

def _do_ioctl(__idir, __base, __nr, __struct, __fd, *args, __payload=None, **kwargs):
  assert not WIN, "ioctl not supported"
  import tinygrad.runtime.support.hcq as hcq, fcntl
  ioctl = __fd.ioctl if isinstance(__fd, hcq.FileIOInterface) else functools.partial(fcntl.ioctl, __fd)
  if __struct is None: return ioctl((__base<<8)|__nr, __payload or (args[0] if args else 0))
  if (rc:=ioctl((__idir<<30)|(ctypes.sizeof(out:=(__payload or __struct(*args, **kwargs)))<<16)|(__base<<8)|__nr, out)):
    raise RuntimeError(f"ioctl returned {rc}")
  return out

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, typ): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, typ)
def _IOR(base, nr, typ): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, typ)
def _IOWR(base, nr, typ): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, typ)

T = TypeVar("T")
U = TypeVar("U")
P = ParamSpec("P")

# mypy can't understand eg. ctypes.POINTER(ctypes.c_int), and python < 3.14 cannot understand ctypes.POINTER[ctypes.c_int]
class POINTER(Generic[T], ctypes._Pointer):
  def __class_getitem__(cls, key): return ctypes.POINTER(key)
def pointer(x: T) -> POINTER[T]: return ctypes.pointer(x) # type: ignore

if TYPE_CHECKING: _CFuncPtr = ctypes._CFunctionType
else: _CFuncPtr = ctypes._CFuncPtr

class CFUNCTYPE(Generic[T, P], _CFuncPtr):
  _flags_ = 0
  def __class_getitem__(cls, key): return ctypes.CFUNCTYPE(key[0], *key[1])
class Array(Generic[T, U], ctypes.Array):
  _type_, _length_ = ctypes.c_byte, 0
  def __class_getitem__(cls, key): return key[0] * get_args(key[1])[0]
  def __new__(cls, ty, l): return ty * l

class Struct(ctypes.Structure):
  SIZE = 0

  def __init__(self, *args, **kwargs):
    ctypes.Structure.__init__(self)
    for f,v in [*zip((rf[0] for rf in self._real_fields_), args), *kwargs.items()]: setattr(self, f, v)

  @classmethod
  def register_fields(cls, fields):
    setattr(cls, "_real_fields_", fields)
    for i, (name, *args) in enumerate(fields): setattr(cls, name, Field(*args, name=name, idx=i))

def record(cls) -> type[Struct]:
  setattr(cls, "_fields_", [('_mem_', ctypes.c_byte * cls.SIZE)])
  return cls

class Field:
  def __init__(self, typ, off, bit_width=None, bit_off=0, *, name=None, idx=0):
    self.typ, self.off, self.bit_width, self.bit_off, self.name, self.idx = typ, off, bit_width, bit_off, name, idx

  def __set_name__(self, owner, name):
    entry = (name, self.typ, self.off) + ((self.bit_width, self.bit_off) if self.bit_width else ())
    if hasattr(owner, "_real_fields_"): owner._real_fields_.append(entry)
    else: setattr(owner, "_real_fields_", [entry])
    self.name, self.idx = name, len(owner._real_fields_) - 1

  # lazily resolve field descriptors
  def _resolve(self, cls):
    if self.bit_width: # handle bitfields ourselves
      sl, set_mask = slice(self.off, self.off+(sz:=ceildiv(self.bit_width+self.bit_off, 8))), ~((mask:=(1 << self.bit_width) - 1) << self.bit_off)
      def b2i(obj): return int.from_bytes(memoryview(obj).cast("B")[sl], sys.byteorder)
      def bset(obj, v): memoryview(obj).cast("B")[sl] = ((b2i(obj) & set_mask) | v << self.bit_off).to_bytes(sz, sys.byteorder)
      # FIXME: signedness
      cf = property(lambda obj: b2i(obj) >> self.bit_off & mask, bset)
    # pull the CField descriptor from a dummy class, zero length arrays are so ctypes manages references to child objects for us
    else: cf = type(self.name, (ctypes.Structure,), {"_layout_": "ms", "_pack_": 1, "_fields_": [(str(i), ctypes.c_byte*0) for i in range(self.idx)] +
                                                                                                [("_", ctypes.c_byte * self.off), ("v", self.typ)]}).v # type: ignore
    setattr(cls, self.name, cf)
    return cf

  def __get__(self, obj, objtype=None): return self._resolve(objtype).__get__(obj, objtype) if objtype else self
  def __set__(self, obj, value): self._resolve(obj.__class__).__set__(obj, value)

@functools.cache
def init_c_struct_t(sz:int, fields: tuple[tuple, ...]):
  (CStruct:=type("CStruct", (Struct,), {'_fields_': [('_mem_', ctypes.c_byte * sz)]})).register_fields(fields) # type: ignore
  return CStruct
def init_c_var(ty, creat_cb): return (creat_cb(v:=ty()), v)[1]

class DLL(ctypes.CDLL):
  _loaded_: set[str] = set()

  @staticmethod
  def findlib(nm:str, paths:list[str], extra_paths=[]):
    if nm == 'libc' and OSX: return '/usr/lib/libc.dylib'
    if pathlib.Path(path:=getenv(nm.replace('-', '_').upper()+"_PATH", '')).is_file(): return path
    for p in paths:
      libpaths = {"posix": [d for d in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep) if d] + ["/usr/lib64", "/usr/lib", "/usr/local/lib"],
                  "nt": os.environ['PATH'].split(os.pathsep),
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
    self.nm, self.emsg = nm, emsg or f"try setting {nm.upper()+'_PATH'}?"
    if (path:= DLL.findlib(nm, paths if isinstance(paths, list) else [paths], extra_paths if isinstance(extra_paths, list) else [extra_paths])):
      if DEBUG >= 3: print(f"loading {nm} from {path}")
      try:
        super().__init__(path, **kwargs)
        self._loaded_.add(self.nm)
      except OSError as e:
        self.emsg = str(e)
        if DEBUG >= 3: print(f"loading {nm} failed: {e}")
    elif DEBUG >= 3: print(f"loading {nm} failed: not found on system")

  def bind(self, restype, *argtypes):
    def wrap(fn):
      cfunc = None
      @functools.wraps(fn)
      def wrapper(*args):
        nonlocal cfunc
        if cfunc is None: (cfunc:=getattr(self, fn.__name__)).argtypes, cfunc.restype = argtypes, restype
        return cfunc(*args)
      return wrapper
    return wrap

  def __getattr__(self, nm):
    if self.nm not in self._loaded_: raise AttributeError(f"failed to load library {self.nm}: {self.emsg}")
    return super().__getattr__(nm)
