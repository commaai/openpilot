import gc
import weakref
from openpilot.selfdrive.ui.mici.layouts import onboarding
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import DriverCameraDialog

onboarding.gui_app.init_window("onboarding-memory-test", fps=60)

# ----- exercise the widget -----

# guide = onboarding.TrainingGuideDMTutorial(continue_callback=lambda: None)
guide = DriverCameraDialog()
onboarding.gui_app.set_modal_overlay(guide)

# guide.show_event()
# guide._show_bad_face_page()
# try:
#   guide.render()
# except Exception:
#   pass
# guide.hide_event()

# --- Option 4: weakref lifecycle check ---
guide_ref = weakref.ref(guide)

onboarding.gui_app.set_modal_overlay(None)
del guide

print("=== Option 4: weakref lifecycle ===")
leaked = guide_ref() is not None
print(f"  Widget alive after del: {leaked}")

# --- Option 5: weakref + gc.get_referrers diagnostic ---
if leaked:
  print("\n=== Option 5: what's keeping it alive? ===")
  obj = guide_ref()
  for r in gc.get_referrers(obj):
    if r is obj:
      continue
    if isinstance(r, dict):
      for attr, val in list(r.items()):
        if val is obj:
          owners = [o for o in gc.get_referrers(r) if o is not obj and not isinstance(o, list)]
          owner_names = [f"{type(o).__module__}.{type(o).__qualname__}" for o in owners[:3]]
          print(f"  dict['{attr}'] (owned by: {owner_names})")
    elif hasattr(r, '__self__') and r.__self__ is not obj:
      print(f"  bound method: {type(r.__self__).__qualname__}.{r.__name__}")
    elif hasattr(r, '__func__'):
      print(f"  method: {r.__name__}")
    else:
      print(f"  {type(r).__module__}.{type(r).__qualname__}")
  del obj

assert not leaked, "Circular reference: widget alive after del"
