import weakref
from collections.abc import Callable
from typing import Any


class CallbackManager:
  def __init__(self):
    self._callbacks: weakref.WeakSet[weakref.WeakMethod] = weakref.WeakSet()

  def add(self, callback_method: Callable[..., Any]) -> None:
    weak_cb = weakref.WeakMethod(callback_method)
    self._callbacks.add(weak_cb)

  def remove(self, callback_method: Callable[..., Any]) -> None:
    weak_cb = weakref.WeakMethod(callback_method)
    self._callbacks.discard(weak_cb)

  def clear(self) -> None:
    self._callbacks.clear()

  def __call__(self, *args, **kwargs) -> None:
    for weak_cb in list(self._callbacks):
      callback = weak_cb()
      if callback is not None:
        # The listener instance is still alive, call the method
        callback(*args, **kwargs)
      else:
        # The listener instance is gone. WeakSet handles automatic removal,
        pass
