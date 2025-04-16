import threading
import time
from typing import Generic, TypeVar
from openpilot.system.ui.lib.application import gui_app

class Renderer:
  def render(self): ...

T = TypeVar("T", bound=Renderer)

class Wrapper(Generic[T]):
  def __init__(self, title: str, renderer_cls: type[T], *args):
    self._title = title
    self._renderer_class = renderer_cls
    self._renderer_args = args
    self._renderer: T | None = None
    self._stop_event = threading.Event()
    self._thread = threading.Thread(target=self._run, args=(self._stop_event,), daemon=True)
    self._thread.start()

  def _run(self, stop_event: threading.Event):
    gui_app.init_window(self._title)
    self._renderer = self._renderer_class(*self._renderer_args)
    try:
      for _ in gui_app.render():
        if stop_event.is_set():
          break
        self._renderer.render()
    finally:
      gui_app.close()

  def __enter__(self):
    return self

  def wait(self):
    """wait for renderer to be initialized"""
    while self._renderer is None and self._thread is not None and self._thread.is_alive():
      time.sleep(0.01)

  def close(self):
    if self._thread is not None and self._thread.is_alive():
      self._stop_event.set()
      self._thread.join(timeout=2.0)
      if self._thread.is_alive():
        print(f"WARNING: failed to join {self._title} thread")
    self._thread = None

  def __del__(self):
    self.close()

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()
