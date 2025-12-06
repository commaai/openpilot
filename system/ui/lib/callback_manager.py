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


class SingleCallback:
  def __init__(self, callback: Callable[..., Any] | None = None):
    self._callback: weakref.WeakMethod | Callable | None = None
    if callback is not None:
      self.set(callback)

  def set(self, callback: Callable[..., Any] | None) -> None:
    """Set the callback. Supports bound methods, functions, and lambdas."""
    if callback is None:
      self._callback = None
    elif hasattr(callback, '__self__'):
      # Bound method - use WeakMethod to avoid keeping instance alive
      self._callback = weakref.WeakMethod(callback)
    elif callable(callback):
      # Lambdas and regular functions - store directly (strong reference)
      self._callback = callback
    else:
      raise TypeError(f"Expected callable, got {type(callback)}")

  def clear(self) -> None:
    self._callback = None

  def __call__(self, *args, **kwargs) -> Any:
    if self._callback is None:
      return None

    if isinstance(self._callback, weakref.WeakMethod):
      callback = self._callback()
      if callback is not None:
        return callback(*args, **kwargs)
      else:
        # Instance was garbage collected
        self._callback = None
        return None
    else:
      # Direct callable reference (lambda or function)
      return self._callback(*args, **kwargs)
