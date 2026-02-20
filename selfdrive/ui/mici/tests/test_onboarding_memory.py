import gc
import time

import numpy as np
import pyray as rl
from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.selfdrive.ui.mici.layouts import onboarding


def test_training_guide_dm_dialog_does_not_leak_without_cyclic_gc():
  def assert_no_new_objects_reference_each_other(run):
    def desc(o):
      module = getattr(o, "__module__", getattr(type(o), "__module__", "?"))
      if isinstance(o, dict):
        keys = [str(k) for k in list(o.keys())[:12]]
        return f"{module}.dict(keys={keys})"
      name = getattr(o, "__name__", None)
      if isinstance(name, str):
        return f"{module}.{type(o).__name__}({name})"
      if hasattr(o, "__class__"):
        return f"{module}.{type(o).__name__}"
      return f"{module}.{type(o).__name__}"

    was_enabled = gc.isenabled()
    try:
      gc.collect()
      gc.disable()
      baseline = {id(o) for o in gc.get_objects()}
      run()
      gc.collect()
      new_objects = [o for o in gc.get_objects() if id(o) not in baseline]
      new_ids = {id(o) for o in new_objects}

      edges = []
      for src in new_objects:
        for ref in gc.get_referents(src):
          if id(ref) in new_ids:
            edges.append((src, ref))
            if len(edges) >= 40:
              break
        if len(edges) >= 40:
          break

      msg = "\n".join(f"{desc(src)} -> {desc(ref)}" for src, ref in edges)
      assert not edges, f"new objects reference each other ({len(edges)} shown):\n{msg}"
    finally:
      if was_enabled:
        gc.enable()

  try:
    onboarding.gui_app.init_window("onboarding-memory-test", fps=60)
    w, h = 1928, 1208
    vipc = VisionIpcServer("camerad")
    vipc.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 4, w, h)
    vipc.start_listener()
    yuv_size = w * h + (w // 2) * (h // 2) * 2
    yuv_data = np.random.randint(0, 256, yuv_size, dtype=np.uint8).tobytes()
    rect = rl.Rectangle(0, 0, onboarding.gui_app.width, onboarding.gui_app.height)
    def run():
      guide = onboarding.TrainingGuideDMTutorial(continue_callback=lambda: None)
      onboarding.gui_app.set_modal_overlay(guide)
      guide.show_event()
      for _ in range(10):
        for frame_idx in range(6):
          eof = int(time.monotonic_ns())
          vipc.send(VisionStreamType.VISION_STREAM_DRIVER, yuv_data, frame_idx % 4, eof, eof)
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)
          guide.render(rect)
          rl.end_drawing()
      onboarding.gui_app.set_modal_overlay(None)
      del guide

    assert_no_new_objects_reference_each_other(run)
  finally:
    onboarding.gui_app.set_modal_overlay(None)
    onboarding.gui_app.close()
