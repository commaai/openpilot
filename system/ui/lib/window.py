import threading
import time
import os
from typing import Generic, Protocol, TypeVar
from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import gui_app


class RendererProtocol(Protocol):
  def render(self): ...


R = TypeVar("R", bound=RendererProtocol)


class BaseWindow(Generic[R]):
  def __init__(self, title: str):
    self._title = title
    self._renderer: R | None = None
    self._stop_event = threading.Event()
    self._thread = threading.Thread(target=self._run)
    self._thread.start()

    # wait for the renderer to be initialized
    while self._renderer is None and self._thread.is_alive():
      time.sleep(0.01)

  def _create_renderer(self) -> R:
    raise NotImplementedError()

  def _run(self):
    if os.getenv("CI") is not None:
      return
    gui_app.init_window(self._title)
    self._renderer = self._create_renderer()
    try:
      for _ in gui_app.render():
        if self._stop_event.is_set():
          break
        self._renderer.render()
    finally:
      gui_app.close()

  def __enter__(self):
    return self

  def close(self):
    if self._thread.is_alive():
      self._stop_event.set()
      self._thread.join(timeout=2.0)
      if self._thread.is_alive():
        cloudlog.warning(f"Failed to join {self._title} thread")

  def __del__(self):
    self.close()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
