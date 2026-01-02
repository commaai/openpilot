import ctypes, functools, os, pathlib, re, sys, sysconfig
from typing import TYPE_CHECKING
from tinygrad.helpers import flatten, getenv, DEBUG, OSX, WIN
from _ctypes import _SimpleCData

def _do_ioctl(__idir, __base, __nr, __struct, __fd, *args, __payload=None, **kwargs):
  assert not WIN, "ioctl not supported"
  import tinygrad.runtime.support.hcq as hcq, fcntl
  ioctl = __fd.ioctl if isinstance(__fd, hcq.FileIOInterface) else functools.partial(fcntl.ioctl, __fd)
  if (rc:=ioctl((__idir<<30)|(ctypes.sizeof(out:=(__payload or __struct(*args, **kwargs)))<<16)|(__base<<8)|__nr, out)):
    raise RuntimeError(f"ioctl returned {rc}")
  return out

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, typ): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, typ)
def _IOR(base, nr, typ): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, typ)
def _IOWR(base, nr, typ): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, typ)

def CEnum(typ: type[ctypes._SimpleCData]):
  class _CEnum(typ): # type: ignore
    _val_to_name_: dict[int,str] = {}

    @classmethod
    def from_param(cls, val): return val if isinstance(val, cls) else cls(val)
    @classmethod
    def get(cls, val, default="unknown"): return cls._val_to_name_.get(val.value if isinstance(val, cls) else val, default)
    @classmethod
    def items(cls): return cls._val_to_name_.items()
    @classmethod
    def define(cls, name, val):
      cls._val_to_name_[val] = name
      return val

    def __eq__(self, other): return self.value == other
    def __repr__(self): return self.get(self) if self.value in self.__class__._val_to_name_ else str(self.value)
    def __hash__(self): return hash(self.value)

  return _CEnum

class DLL(ctypes.CDLL):
  @staticmethod
  def findlib(nm:str, paths:list[str], extra_paths=[]):
    if nm == 'libc' and OSX: return '/usr/lib/libc.dylib'
    if pathlib.Path(path:=getenv(nm.replace('-', '_').upper()+"_PATH", '')).is_file(): return path
    for p in paths:
      libpaths = {"posix": ["/usr/lib", "/usr/local/lib"], "nt": os.environ['PATH'].split(os.pathsep),
                  "darwin": ["/opt/homebrew/lib", f"/System/Library/Frameworks/{p}.framework"],
                  'linux': ['/lib', f"/lib/{sysconfig.get_config_var('MULTIARCH')}", "/usr/lib/wsl/lib/"]}
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
    self.nm, self.emsg, self.loaded = nm, emsg, False
    if (path:= DLL.findlib(nm, paths if isinstance(paths, list) else [paths], extra_paths if isinstance(extra_paths, list) else [extra_paths])):
      if DEBUG >= 3: print(f"loading {nm} from {path}")
      try:
        super().__init__(path, **kwargs)
        self.loaded = True
      except OSError as e:
        self.emsg = str(e)
        if DEBUG >= 3: print(f"loading {nm} failed: {e}")
    elif DEBUG >= 3: print(f"loading {nm} failed: not found on system")

  def __getattr__(self, nm):
    if not self.loaded: raise AttributeError(f"failed to load library {self.nm}: " + (self.emsg or f"try setting {self.nm.upper()+'_PATH'}?"))
    return super().__getattr__(nm)

# supports gcc (C11) __attribute__((packed))
if TYPE_CHECKING: Struct = ctypes.Structure
else:
  class MetaStruct(type(ctypes.Structure)):
    def __new__(mcs, name, bases, dct):
      fields = dct.pop("_fields_", None)
      cls = super().__new__(mcs, name, bases, dct)
      if dct.get("_packed_", False) and fields is not None: mcs._build(cls, fields)
      return cls

    def __setattr__(cls, k, v):
      # https://github.com/python/cpython/issues/90914
      if k == "_fields_": v = [(f[0], ctypes.c_uint8, f[2]) if len(f) == 3 and f[1] == ctypes.c_bool else f for f in v]
      # NB: _fields_ must be set after _packed_ because PyCStructType_setattro marks _fields_ as final.
      if k == "_fields_" and getattr(cls, "_packed_", False): type(cls)._build(cls, v)
      elif k == "_packed_" and hasattr(cls, "_fields_"): type(cls)._build(cls, cls._fields_)
      else: super().__setattr__(k, v)

    @staticmethod
    def _build(cls, fields):
      offset = 0
      for nm, ty, bf in [(f[0], f[1], f[2] if len(f) == 3 else 0) for f in fields]:
        if bf == 0: offset = (offset + 7) & ~7
        mask = (1 << (sz:=ctypes.sizeof(ty)*8 if bf == 0 else bf)) - 1
        def fget(self, mask, off, ty): return ((int.from_bytes(self._data, sys.byteorder)>>off)&mask if issubclass(ty, _SimpleCData) else
                                               ty.from_buffer(memoryview(self._data)[(st:=off//8):st+ctypes.sizeof(ty)]))
        def fset(self, val, mask, off):
          if val.__class__ is not int: val = int.from_bytes(val, sys.byteorder)
          self._data[:] = (((int.from_bytes(self._data, sys.byteorder) & ~(mask<<off))|((val&mask)<<off))
                                                              .to_bytes(len(self._data), sys.byteorder))
        setattr(cls, nm, property(functools.partial(fget, mask=mask, off=offset, ty=ty), functools.partial(fset, mask=mask, off=offset)))
        offset += sz

      def pget(ty, s): return getattr(ty, f'_packed_{s}_', getattr(ty, f'_{s}_', []))
      def get_aty(anm, fs=fields): return next(f[1] for f in fs if f[0] == anm)
      def get_fnms(ty): return [f[0] for f in pget(ty, 'fields') if f[0] not in pget(ty, 'anonymous')]

      if hasattr(cls, '_anonymous_'):
        for anm, aty in [(a, get_aty(a)) for a in cls._anonymous_]:
          for fnm in (get_fnms(aty) + flatten([get_fnms(get_aty(aanm, pget(aty, 'fields'))) for aanm in pget(aty, 'anonymous')])):
            setattr(cls, fnm, property(functools.partial(lambda self, anm, fnm: getattr(getattr(self, anm), fnm), anm=anm, fnm=fnm),
                                       functools.partial(lambda self, v, anm, fnm: setattr(getattr(self, anm), fnm, v), anm=anm, fnm=fnm)))
        setattr(cls, '_packed_anonymous_', cls._anonymous_)
        setattr(cls, '_anonymous_', [])
      type(ctypes.Structure).__setattr__(cls, '_fields_', [('_data', ctypes.c_ubyte * ((offset + 7) // 8))])
      type(ctypes.Structure).__setattr__(cls, '_packed_', True)
      setattr(cls, '_packed_fields_', fields)

  class Struct(ctypes.Structure, metaclass=MetaStruct):
    def __init__(self, *args, **kwargs):
      if hasattr(self, '_packed_fields_'):
        for f,v in zip(self._packed_fields_, args): setattr(self, f[0], v)
        for k,v in kwargs.items(): setattr(self, k, v)
      else: super().__init__(*args, **kwargs)
