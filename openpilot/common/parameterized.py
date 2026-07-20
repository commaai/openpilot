import re
import sys
import inspect
import unittest


def _to_safe_name(s):
  return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s)).strip("_")


class parameterized:
  @staticmethod
  def expand(cases, names=None, ids=None):
    cases = list(cases)

    if not cases:
      return lambda func: unittest.skip("no parameterized cases")(func)

    def decorator(func):
      params = [p for p in inspect.signature(func).parameters if p != 'self']
      normalized = [c if isinstance(c, tuple) else (c,) for c in cases]
      # Infer arg count from first case so extra params (e.g. from @given) are left untouched
      expand_params = params[: len(normalized[0])]
      def wrapper(self):
        for case in normalized:
          label = ids(*case) if ids is not None else None
          with self.subTest(label, **dict(zip(expand_params, case, strict=False))):
            if names is None:
              func(self, *case)
            else:
              values = dict(zip(names, case, strict=True))
              values.update({name: self._fixture(name) for name in params if name not in values})
              func(self, **values)
      wrapper.__name__ = func.__name__
      wrapper.__doc__ = func.__doc__
      return wrapper

    return decorator


def parameterized_class(attrs, input_list=None):
  if isinstance(attrs, list) and (not attrs or isinstance(attrs[0], dict)):
    params_list = attrs
  else:
    assert input_list is not None
    attr_names = (attrs,) if isinstance(attrs, str) else tuple(attrs)
    params_list = [dict(zip(attr_names, v if isinstance(v, (tuple, list)) else (v,), strict=False)) for v in input_list]

  def decorator(cls):
    globs = sys._getframe(1).f_globals
    for i, params in enumerate(params_list):
      # Append sanitized values so unittest's -k can filter by them.
      suffix = "_".join(filter(None, (_to_safe_name(v) for v in params.values() if isinstance(v, str))))
      name = f"{cls.__name__}_{i}" + (f"_{suffix}" if suffix else "")
      new_cls = type(name, (cls,), dict(params))
      new_cls.__module__ = cls.__module__
      new_cls.__unittest_skip__ = False
      globs[name] = new_cls
    # Don't collect the un-parametrised base.
    cls.__unittest_skip__ = True
    cls.__unittest_skip_why__ = "parameterized base class"
    return cls

  return decorator
