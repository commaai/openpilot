#!/usr/bin/env python3
import pyray as rl
from cereal import messaging
from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.sidebar import Sidebar, SIDEBAR_WIDTH
from openpilot.selfdrive.ui.layouts.home import HomeLayout
from openpilot.system.ui.onroad.augmented_road_view import AugmentedRoadView


class UI:
  def __init__(self, sm):
    self._sidbar = Sidebar()
    self._sidebar_visible = True
    self._is_onroad = True

    self._home_layout = HomeLayout()
    self._augmented_road_view = AugmentedRoadView(sm, VisionStreamType.VISION_STREAM_ROAD)

    self._sidebar_rect = rl.Rectangle(0, 0, SIDEBAR_WIDTH, gui_app.height)
    self._content_rect = rl.Rectangle(SIDEBAR_WIDTH, 0, gui_app.width - SIDEBAR_WIDTH, gui_app.height)

  def render(self, sm, rect):
    self._render_main_content(sm, rect)

    if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
      self._handle_mouse_click(rl.get_mouse_position())

  def _render_main_content(self, sm, rect):
    if self._sidebar_visible:
      self._sidbar.draw(sm, self._sidebar_rect)

    if self._is_onroad:
      self._augmented_road_view.render(self._content_rect)
    else:
      self._home_layout.render(self._content_rect, sm)

  def _handle_mouse_click(self, pos: rl.Vector2):
    if rl.check_collision_point_rec(pos, self._content_rect):
      self._sidebar_visible = not self._sidebar_visible
      if self._sidebar_visible:
        self._content_rect.x = SIDEBAR_WIDTH
        self._content_rect.width = gui_app.width - SIDEBAR_WIDTH
      else:
        self._content_rect.x = 0
        self._content_rect.width = gui_app.width

def main():
  gui_app.init_window("UI")
  # TODO: remove this after singlon ui_state has been implemented
  sm = messaging.SubMaster(
    [
      "modelV2",
      "controlsState",
      "liveCalibration",
      "radarState",
      "deviceState",
      "pandaStates",
      "carParams",
      "driverMonitoringState",
      "carState",
      "driverStateV2",
      "roadCameraState",
      "wideRoadCameraState",
      "managerState",
      "selfdriveState",
      "longitudinalPlan",
    ]
  )
  ui = UI(sm)
  for _ in gui_app.render():
    sm.update(0)
    ui.render(sm, rl.Rectangle(0, 0, gui_app.width, gui_app.height))


if __name__ == "__main__":
  main()
