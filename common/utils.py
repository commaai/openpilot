from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar


_RT = TypeVar("_RT")


class Freezable:
  _frozen: bool = False

  def freeze(self):
    if not self._frozen:
      self._frozen = True

  def __setattr__(self, *args, **kwargs):
    if self._frozen:
      raise Exception("cannot modify frozen object")
    super().__setattr__(*args, **kwargs)


def cache(user_function: Callable[..., _RT], /) -> Callable[..., _RT]:
  return lru_cache(maxsize=None)(user_function)
