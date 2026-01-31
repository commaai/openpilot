import ctypes, ctypes.util, functools, sys
from tinygrad.runtime.support.c import del_an
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING: id_ = ctypes.c_void_p
else:
  class id_(ctypes.c_void_p):
    _is_finalizing = sys.is_finalizing # FIXME: why is this needed

    retain: bool = False
    # This prevents ctypes from converting response to plain int, and dict.fromkeys() can use it to dedup
    def __hash__(self): return hash(self.value)
    def __eq__(self, other): return self.value == other.value
    def __del__(self):
      if self.retain and not self._is_finalizing(): self.release()
    def release(self): msg("release")(self)
    def retained(self):
      setattr(self, 'retain', True)
      return self

def returns_retained(f): return functools.wraps(f)(lambda *args, **kwargs: f(*args, **kwargs).retained())

lib = ctypes.CDLL(ctypes.util.find_library('objc'))
lib.sel_registerName.restype = id_
getsel = functools.cache(lib.sel_registerName)
lib.objc_getClass.restype = id_
dispatch_data_create = ctypes.CDLL("/usr/lib/libSystem.dylib").dispatch_data_create
dispatch_data_create.restype = id_
dispatch_data_create = returns_retained(dispatch_data_create)

def msg(sel:str, restype=id_, argtypes=[], retain=False, clsmeth=False):
  # Using attribute access returns a new reference so setting restype is safe
  (sender:=lib["objc_msgSend"]).restype, sender.argtypes = del_an(restype), [id_, id_]+[del_an(a) for a in argtypes] if argtypes else []
  def f(ptr, *args): return sender(ptr._objc_class_ if clsmeth else ptr, getsel(sel.encode()), *args)
  return returns_retained(f) if retain else f

if TYPE_CHECKING:
  import _ctypes
  class MetaSpec(_ctypes._PyCSimpleType):
    _objc_class_: id_
    def __getattr__(cls, nm:str) -> Any: ...
    def __setattr__(cls, nm:str, v:Any): ...
else:
  class MetaSpec(type(id_)):
    def __new__(mcs, name, bases, dct):
      cls = super().__new__(mcs, name, bases, {'_objc_class_': lib.objc_getClass(name.encode()), '_children_': set(), **dct})
      cls._methods_, cls._classmethods_ = dct.get('_methods_', []), dct.get('_classmethods_', [])
      return cls

    def __setattr__(cls, k, v):
      super().__setattr__(k, v)
      if k in ("_methods_", "_classmethods_"):
        for m in v: cls._addmeth(m, clsmeth=(v=="_classmethods_"))
        for c in cls._children_: c._inherit(cls)
      if k == "_bases_":
        for b in v:
          b._children_.add(cls)
          cls._inherit(b)

    def _inherit(cls, b):
      for _b in getattr(b, "_bases_", []): cls._inherit(_b)
      for m in getattr(b, "_methods_", []): cls._addmeth(m)
      for m in getattr(b, "_classmethods_", []): cls._addmeth(m, True)
      for c in cls._children_: c._inherit(cls)

    def _addmeth(cls, m, clsmeth=False):
      nm = m[0].strip(':').replace(':', '_')
      if clsmeth: setattr(cls, nm, classmethod(msg(m[0], cls if m[1] == 'instancetype' else m[1],
                                                   [cls if a == 'instancetype' else a for a in m[2]], *m[3:], clsmeth=True))) # type: ignore[misc]
      else: setattr(cls, nm, msg(m[0], cls if m[1] == 'instancetype' else m[1], [cls if a == 'instancetype' else a for a in m[2]], *m[3:]))

class Spec(id_, metaclass=MetaSpec):
  if TYPE_CHECKING:
    def __getattr__(self, nm:str) -> Any: ...
