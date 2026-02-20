import pyray as rl
rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
import gc
import weakref
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

# mici modals
from openpilot.selfdrive.ui.mici.layouts.onboarding import TrainingGuide as MiciTrainingGuide, OnboardingWindow as MiciOnboardingWindow
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import DriverCameraDialog as MiciDriverCameraDialog
from openpilot.selfdrive.ui.mici.widgets.pairing_dialog import PairingDialog as MiciPairingDialog

# tici modals
from openpilot.selfdrive.ui.onroad.driver_camera_dialog import DriverCameraDialog as TiciDriverCameraDialog
from openpilot.selfdrive.ui.layouts.onboarding import OnboardingWindow as TiciOnboardingWindow
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog as TiciPairingDialog

# known small leaks not worth worrying about
KNOWN_LEAKS = {
  "openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog.DriverCameraView",
  "openpilot.selfdrive.ui.mici.layouts.onboarding.TermsPage",
  "openpilot.selfdrive.ui.mici.layouts.onboarding.TrainingGuide",
  "openpilot.selfdrive.ui.mici.layouts.onboarding.DeclinePage",
  "openpilot.selfdrive.ui.mici.layouts.onboarding.OnboardingWindow",
  "openpilot.selfdrive.ui.onroad.driver_state.DriverStateRenderer",
  "openpilot.selfdrive.ui.onroad.driver_camera_dialog.DriverCameraDialog",
  "openpilot.selfdrive.ui.layouts.onboarding.TermsPage",
  "openpilot.selfdrive.ui.layouts.onboarding.DeclinePage",
  "openpilot.selfdrive.ui.layouts.onboarding.OnboardingWindow",
}


def get_child_widgets(widget: Widget) -> list[Widget]:
  children = []
  for val in widget.__dict__.values():
    items = val if isinstance(val, (list, tuple)) else (val,)
    children.extend(w for w in items if isinstance(w, Widget))
  return children


def test_dialogs_do_not_leak():
  gui_app.init_window("ref-test")

  leaked_widgets = set()

  for test_widget in (
    MiciDriverCameraDialog, MiciTrainingGuide, MiciOnboardingWindow, MiciPairingDialog,
    TiciDriverCameraDialog, TiciOnboardingWindow, TiciPairingDialog,
  ):
    widget = test_widget()
    all_refs = [weakref.ref(w) for w in get_child_widgets(widget) + [widget]]

    del widget

    for ref in all_refs:
      if ref() is not None:
        obj = ref()
        name = f"{type(obj).__module__}.{type(obj).__qualname__}"
        leaked_widgets.add(name)

        print(f"\n===  Widget {name} alive after del")
        print("  Referrers:")
        for r in gc.get_referrers(obj):
          if r is obj:
            continue

          if hasattr(r, '__self__') and r.__self__ is not obj:
            print(f"    bound method: {type(r.__self__).__qualname__}.{r.__name__}")
          elif hasattr(r, '__func__'):
            print(f"    method: {r.__name__}")
          else:
            print(f"    {type(r).__module__}.{type(r).__qualname__}")
        del obj

  gui_app.close()

  unexpected = leaked_widgets - KNOWN_LEAKS
  assert not unexpected, f"New leaked widgets: {unexpected}"

  fixed = KNOWN_LEAKS - leaked_widgets
  assert not fixed, f"These leaks are fixed, remove from KNOWN_LEAKS: {fixed}"


if __name__ == "__main__":
  test_dialogs_do_not_leak()
