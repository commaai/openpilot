import re
import sys
import unittest


def _to_safe_name(s):
  return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s)).strip("_")


class _ParameterizedMethod:
  """Expands into one test method per case when the owning class is created (via __set_name__)."""

  def __init__(self, func, cases):
    self.func = func
    self.cases = cases

  def __call__(self, *args, **kwargs):
    raise RuntimeError("parameterized.expand only works on methods of a class")

  def __set_name__(self, owner, name):
    func = self.func
    if not self.cases:
      def skip(self):
        raise unittest.SkipTest("no parameterized cases")
      skip.__name__ = skip.__qualname__ = name
      setattr(owner, name, skip)
      return

    delattr(owner, name)
    for i, case in enumerate(self.cases):
      case_args = case if isinstance(case, tuple) else (case,)
      suffix = "_".join(filter(None, (_to_safe_name(a) for a in case_args if isinstance(a, str))))
      case_name = f"{name}_{i}" + (f"_{suffix}" if suffix else "")

      def make_case(case_args):
        def fn(self):
          return func(self, *case_args)
        return fn

      case_fn = make_case(case_args)
      case_fn.__name__ = case_fn.__qualname__ = case_name
      case_fn.__doc__ = func.__doc__
      setattr(owner, case_name, case_fn)


class parameterized:
  @staticmethod
  def expand(cases):
    cases = list(cases)
    return lambda func: _ParameterizedMethod(func, cases)


def parameterized_class(attrs, input_list=None):
  if isinstance(attrs, list) and (not attrs or isinstance(attrs[0], dict)):
    params_list = attrs
  else:
    assert input_list is not None
    attr_names = (attrs,) if isinstance(attrs, str) else tuple(attrs)
    params_list = [dict(zip(attr_names, v if isinstance(v, (tuple, list)) else (v,), strict=False)) for v in input_list]

  def decorator(cls):
    globs = sys._getframe(1).f_globals
    # preserve a skip already applied to (or inherited by) the class, e.g. @slow
    skip = {"__unittest_skip__": getattr(cls, "__unittest_skip__", False), "__unittest_skip_why__": getattr(cls, "__unittest_skip_why__", "")}
    for i, params in enumerate(params_list):
      # append sanitized string param values so -k can filter by them
      suffix = "_".join(filter(None, (_to_safe_name(v) for v in params.values() if isinstance(v, str))))
      name = f"{cls.__name__}_{i}" + (f"_{suffix}" if suffix else "")
      new_cls = type(name, (cls,), {**params, **skip})
      new_cls.__module__ = cls.__module__
      globs[name] = new_cls
    # hide the un-parameterized base from unittest; generated subclasses explicitly un-skip
    cls.__unittest_skip__ = True
    cls.__unittest_skip_why__ = "parameterized_class base"
    return cls

  return decorator
