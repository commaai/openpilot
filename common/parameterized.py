import sys
import pytest
import inspect


class parameterized:
  @staticmethod
  def expand(cases):
    cases = list(cases)

    if not cases:
      return lambda func: pytest.mark.skip("no parameterized cases")(func)

    def decorator(func):
      params = [p for p in inspect.signature(func).parameters if p != 'self']
      normalized = [c if isinstance(c, tuple) else (c,) for c in cases]
      # Infer arg count from first case so extra params (e.g. from @given) are left untouched
      expand_params = params[: len(normalized[0])]
      if len(expand_params) == 1:
        return pytest.mark.parametrize(expand_params[0], [c[0] for c in normalized])(func)
      return pytest.mark.parametrize(', '.join(expand_params), normalized)(func)

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
      name = f"{cls.__name__}_{i}"
      new_cls = type(name, (cls,), dict(params))
      new_cls.__module__ = cls.__module__
      new_cls.__test__ = True  # override inherited False so pytest collects this subclass
      globs[name] = new_cls
    # Don't collect the un-parametrised base, but return it so outer decorators
    # (e.g. @pytest.mark.skip) land on it and propagate to subclasses via MRO.
    cls.__test__ = False
    return cls

  return decorator
