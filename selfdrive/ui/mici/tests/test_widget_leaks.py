import pyray as rl
rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
import gc
import weakref
import pytest
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

# mici dialogs
from openpilot.selfdrive.ui.mici.layouts.onboarding import TrainingGuide as MiciTrainingGuide, OnboardingWindow as MiciOnboardingWindow
from openpilot.selfdrive.ui.mici.onroad.driver_camera_dialog import DriverCameraDialog as MiciDriverCameraDialog
from openpilot.selfdrive.ui.mici.widgets.pairing_dialog import PairingDialog as MiciPairingDialog
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog, BigConfirmationDialogV2, BigInputDialog, BigMultiOptionDialog
from openpilot.selfdrive.ui.mici.layouts.settings.device import MiciFccModal

# tici dialogs
from openpilot.selfdrive.ui.onroad.driver_camera_dialog import DriverCameraDialog as TiciDriverCameraDialog
from openpilot.selfdrive.ui.layouts.onboarding import OnboardingWindow as TiciOnboardingWindow
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog as TiciPairingDialog
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog
from openpilot.system.ui.widgets.html_render import HtmlModal
from openpilot.system.ui.widgets.keyboard import Keyboard

# FIXME: known small leaks not worth worrying about at the moment
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
  "openpilot.system.ui.widgets.confirm_dialog.ConfirmDialog",
  "openpilot.system.ui.widgets.label.Label",
  "openpilot.system.ui.widgets.button.Button",
  "openpilot.selfdrive.ui.mici.widgets.dialog.BigDialog",
  "openpilot.system.ui.widgets.html_render.HtmlRenderer",
  "openpilot.system.ui.widgets.NavBar",
  "openpilot.system.ui.widgets.inputbox.InputBox",
  "openpilot.system.ui.widgets.scroller_tici.Scroller",
  "openpilot.system.ui.widgets.scroller.Scroller",
  "openpilot.system.ui.widgets.label.UnifiedLabel",
  "openpilot.selfdrive.ui.mici.widgets.dialog.BigMultiOptionDialog",
  "openpilot.system.ui.widgets.mici_keyboard.MiciKeyboard",
  "openpilot.selfdrive.ui.mici.widgets.dialog.BigConfirmationDialogV2",
  "openpilot.system.ui.widgets.keyboard.Keyboard",
  "openpilot.system.ui.widgets.slider.BigSlider",
  "openpilot.selfdrive.ui.mici.widgets.dialog.BigInputDialog",
  "openpilot.system.ui.widgets.option_dialog.MultiOptionDialog",
}


def get_child_widgets(widget: Widget) -> list[Widget]:
  children = []
  for val in widget.__dict__.values():
    items = val if isinstance(val, (list, tuple)) else (val,)
    children.extend(w for w in items if isinstance(w, Widget))
  return children


@pytest.mark.skip(reason="segfaults")
def test_dialogs_do_not_leak():
  gui_app.init_window("ref-test")

  leaked_widgets = set()

  for ctor in (
    # mici
    MiciDriverCameraDialog, MiciTrainingGuide, MiciOnboardingWindow, MiciPairingDialog,
    lambda: BigDialog("test", "test"),
    lambda: BigConfirmationDialogV2("test", "icons_mici/settings/network/new/trash.png"),
    lambda: BigInputDialog("test"),
    lambda: BigMultiOptionDialog(["a", "b"], "a"),
    lambda: MiciFccModal(text="test"),
    # tici
    TiciDriverCameraDialog, TiciOnboardingWindow, TiciPairingDialog, Keyboard,
    lambda: ConfirmDialog("test", "ok"),
    lambda: MultiOptionDialog("test", ["a", "b"]),
    lambda: HtmlModal(text="test"),
  ):
    widget = ctor()
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
