import gc
import time
import weakref

import numpy as np
import pyray as rl
from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.selfdrive.ui.mici.layouts import onboarding


def test_training_guide_dm_dialog_does_not_leak_without_cyclic_gc():
  """
  Integration regression test for callback-retained DM setup dialogs.

  The UI process disables automatic GC, so objects must be releasable by
  refcounting alone after modal close. If callbacks capture strong references
  to the guide, these objects stay alive until an explicit gc.collect().
  """
  was_enabled = gc.isenabled()
  gc.collect()
  gc.disable()

  try:
    onboarding.gui_app.init_window("onboarding-memory-test", fps=60)

    w, h = 1928, 1208
    vipc = VisionIpcServer("camerad")
    vipc.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 4, w, h)
    vipc.start_listener()

    yuv_size = w * h + (w // 2) * (h // 2) * 2
    yuv_data = np.random.randint(0, 256, yuv_size, dtype=np.uint8).tobytes()
    rect = rl.Rectangle(0, 0, onboarding.gui_app.width, onboarding.gui_app.height)

    refs = []
    for idx in range(10):
      guide = onboarding.TrainingGuideDMTutorial(continue_callback=lambda: None)
      onboarding.gui_app.set_modal_overlay(guide)
      guide.show_event()

      # Exercise real CameraView/VisionIPC path before closing modal.
      for frame_idx in range(6):
        eof = int(time.monotonic_ns())
        vipc.send(VisionStreamType.VISION_STREAM_DRIVER, yuv_data, frame_idx % 4, eof, eof)
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        guide.render(rect)
        rl.end_drawing()

      refs += [
        weakref.ref(guide),
        weakref.ref(guide._dialog),
        weakref.ref(guide._bad_face_page),
      ]

      # Mimic modal close path in UI.
      onboarding.gui_app.set_modal_overlay(None)
      del guide

      # With GC disabled, retained objects indicate a strong reference cycle.
      leaked = sum(1 for ref in refs if ref() is not None)
      assert leaked == 0, f"detected retained onboarding DM objects after iteration {idx}: {leaked}"

    # Collect once at end to keep test process clean.
    gc.collect()
    leaked = sum(1 for ref in refs if ref() is not None)
    assert leaked == 0, f"{leaked} onboarding DM objects remained alive without cyclic GC"
  finally:
    # Collect before closing the raylib window so __del__ cleanup paths that
    # call raylib APIs run while GL context is still valid.
    gc.collect()
    onboarding.gui_app.set_modal_overlay(None)
    onboarding.gui_app.close()
    if was_enabled:
      gc.enable()
