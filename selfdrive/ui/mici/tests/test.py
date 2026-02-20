import gc
import weakref
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.mici.layouts import onboarding
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import DriverCameraDialog
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout

from selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
from selfdrive.ui.mici.layouts.settings.settings import SettingsLayout

onboarding.gui_app.init_window("onboarding-memory-test", fps=60)


# def get_all_widgets(widget: Widget, seen: set[Widget] | None = None) -> set[Widget]:
#   if seen is None:
#     seen = set()
#
#   if widget in seen:
#     return seen
#
#   seen.add(widget)
#
#   for attr, val in widget.__dict__.items():
#     if isinstance(val, Widget):
#       get_all_widgets(val, seen)
#     elif isinstance(val, (list, tuple)):
#       for item in val:
#         if isinstance(item, Widget):
#           get_all_widgets(item, seen)
#
#   return seen


def get_all_widgets(widget: Widget) -> list[Widget]:
  seen = []

  for attr, val in widget.__dict__.items():
    if isinstance(val, Widget):
      seen.append(val)
      # seen.extend(get_all_widgets(val))
    elif isinstance(val, (list, tuple)):
      for item in val:
        if isinstance(item, Widget):
          seen.append(item)
          # seen.extend(get_all_widgets(item))

  return seen


# def run_test_on_widget(widget: Widget):
#   widget_ref = weakref.ref(widget)
#   del widget
#
#   leaked = widget_ref() is not None
#   print(f"  Widget {type(widget_ref()).__module__}.{type(widget_ref()).__qualname__} alive after del: {leaked}")
#
#   if leaked:
#     print("\n=== Option 5: what's keeping it alive? ===")
#     obj = widget_ref()
#     for r in gc.get_referrers(obj):
#       if r is obj:
#         continue
#       if isinstance(r, dict):
#         for attr, val in list(r.items()):
#           if val is obj:
#             owners = [o for o in gc.get_referrers(r) if o is not obj and not isinstance(o, list)]
#             owner_names = [f"{type(o).__module__}.{type(o).__qualname__}" for o in owners[:3]]
#             print(f"  dict['{attr}'] (owned by: {owner_names})")
#       elif hasattr(r, '__self__') and r.__self__ is not obj:
#         print(f"  bound method: {type(r.__self__).__qualname__}.{r.__name__}")
#       elif hasattr(r, '__func__'):
#         print(f"  method: {r.__name__}")
#       else:
#         print(f"  {type(r).__module__}.{type(r).__qualname__}")
#     del obj
#
#   assert not leaked, "Circular reference: widget alive after del"

# ----- exercise the widget -----

# widget = onboarding.TrainingGuide()
# widget = DriverCameraDialog()
# widget = SettingsLayout()
for test_widget in (
    onboarding.TrainingGuide,
  # onboarding.TrainingGuideDMTutorial(continue_callback=lambda: None),
):

  widget = test_widget()
  widget_ref = weakref.ref(widget)

  all_widgets = get_all_widgets(widget) + [widget]

  all_refs = [weakref.ref(w) for w in all_widgets]
  print('testing on', all_widgets)

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

