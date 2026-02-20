import gc
import weakref
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.mici.layouts.onboarding import TrainingGuide, OnboardingWindow
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import DriverCameraDialog
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout

from selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
from selfdrive.ui.mici.layouts.settings.settings import SettingsLayout

gui_app.init_window("onboarding-memory-test")


def get_child_widgets(widget: Widget) -> list[Widget]:
  children = []
  for val in widget.__dict__.values():
    items = val if isinstance(val, (list, tuple)) else (val,)
    children.extend(w for w in items if isinstance(w, Widget))
  return children


for test_widget in (DriverCameraDialog, TrainingGuide, OnboardingWindow):
  widget = test_widget()
  widget_ref = weakref.ref(widget)

  all_widgets = get_child_widgets(widget) + [widget]

  all_refs = [weakref.ref(w) for w in all_widgets]

  del widget, all_widgets

  for ref in all_refs:
    if ref() is not None:
      print(f"\n===  Widget {type(ref()).__module__}.{type(ref()).__qualname__} alive after del: True")

      obj = ref()
      for r in gc.get_referrers(obj):
        if r is obj:
          continue

        if hasattr(r, '__self__') and r.__self__ is not obj:
          print(f"  bound method: {type(r.__self__).__qualname__}.{r.__name__}")
        elif hasattr(r, '__func__'):
          print(f"  method: {r.__name__}")
        else:
          print(f"  {type(r).__module__}.{type(r).__qualname__}")
      del obj

    # assert not leaked, "Circular reference: widget alive after del"

  # for child in child_widgets:
  #   run_test_on_widget(child)


gui_app.close()
