from openpilot.selfdrive.ui.layouts.settings.common import restart_needed_callback
from openpilot.selfdrive.ui.mici.onroad.cabin_camera_dialog import CabinCameraDialog
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigParamControl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller


class DriverMonitoringLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._always_on_dm = BigParamControl("always-on\nmonitoring", "AlwaysOnDM",
                                         description="Monitor the driver even when openpilot is not engaged.")
    self._always_on_dm.set_enabled(not ui_state.params.get_bool("AlwaysOnDMLock"))

    self._record_front = BigParamControl("record & upload cabin camera", "RecordFront",
                                         description_icon=gui_app.texture("icons_mici/settings/device/cameras.png", 64, 64),
                                         toggle_callback=restart_needed_callback,
                                         description="Upload cabin camera data to help improve driver monitoring.")
    self._record_front.set_enabled(False if ui_state.params.get_bool("RecordFrontLock") else (lambda: not ui_state.engaged))

    cabin_cam_btn = BigButton("cabin\ncamera preview", "", gui_app.texture("icons_mici/settings/device/cameras.png", 64, 64),
                              description="Preview the cabin camera to check driver monitoring visibility. The vehicle must be off.")
    cabin_cam_btn.set_click_callback(lambda: gui_app.push_widget(CabinCameraDialog()))
    cabin_cam_btn.set_enabled(lambda: ui_state.is_offroad())

    self._scroller.add_widgets([self._always_on_dm, self._record_front, cabin_cam_btn])
    ui_state.add_engaged_transition_callback(self._refresh_toggles)

  def show_event(self):
    super().show_event()
    self._refresh_toggles()

  def _refresh_toggles(self):
    self._always_on_dm.refresh()
    self._record_front.refresh()
