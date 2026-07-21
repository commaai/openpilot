import re
import sys
import inspect
import unittest


def _to_safe_name(s):
  return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s)).strip("_")


class parameterized:
  @staticmethod
  def expand(cases, names=None, ids=None, serial=False):
    cases = list(cases)

    if not cases:
      return lambda func: unittest.skip("no parameterized cases")(func)

    if serial:
      def decorator(func):
        normalized = [case if isinstance(case, tuple) else (case,) for case in cases]

        def wrapper(self):
          for case in normalized:
            with self.subTest():
              func(self, *case)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

      return decorator

    return lambda func: _Expanded(func, cases, names, ids)


class _Expanded:
  """Descriptor that turns every parameter case into a real unittest method."""

  def __init__(self, func, cases, names, ids):
    self.func = func
    self.cases = [c if isinstance(c, tuple) else (c,) for c in cases]
    self.names = names
    self.ids = ids

  def __set_name__(self, owner, name):
    params = [p for p in inspect.signature(self.func).parameters if p != "self"]

    for index, case in enumerate(self.cases):
      label = self.ids(*case) if self.ids is not None else None
      method_name = f"{name}_{index}" + (f"_{_to_safe_name(label)}" if label is not None else "")

      def test_method(test_case, current_case=case):
        if self.names is None:
          self.func(test_case, *current_case)
        else:
          values = dict(zip(self.names, current_case, strict=True))
          values.update({param: test_case._fixture(param) for param in params if param not in values})
          self.func(test_case, **values)

      test_method.__name__ = method_name
      test_method.__doc__ = self.func.__doc__
      setattr(owner, method_name, test_method)

    # The descriptor itself is only a method factory, not a test.
    setattr(owner, name, None)


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
      # The template is marked skipped below, so generated classes need their
      # own skip state. Preserve explicit and environment-driven skips.
      if "__unittest_skip__" not in new_cls.__dict__:
        new_cls.__unittest_skip__ = getattr(cls, "__unittest_skip__", False)
        new_cls.__unittest_skip_why__ = getattr(cls, "__unittest_skip_why__", "")
      globs[name] = new_cls
    # Don't collect the un-parametrised base.
    cls.__unittest_skip__ = True
    cls.__unittest_skip_why__ = "parameterized base class"
    return cls

  return decorator
