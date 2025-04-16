import threading
import time
from typing import Any
from openpilot.system.ui.lib.application import gui_app

class Wrapper:
  _renderer: Any | None = None

  def __init__(self, title: str, renderer_cls, *renderer_args):
    self._title = title
    self._renderer_cls = renderer_cls
    self._renderer_args = renderer_args
    self._stop_event = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

    # wait for renderer to be initialized
    while self._renderer is None and self._thread is not None and self._thread.is_alive():
      time.sleep(0.01)

  def _run(self):
    gui_app.init_window(self._title)
    self._renderer = renderer = self._renderer_cls(*self._renderer_args)
    try:
      for _ in gui_app.render():
        if self._stop_event.is_set():
          break
        renderer.render()
    finally:
      gui_app.close()

  def __enter__(self):
    return self

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
