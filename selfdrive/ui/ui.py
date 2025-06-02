#!/usr/bin/env python3
import pyray as rl
from msgq.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.sidebar import Sidebar, SIDEBAR_WIDTH
from openpilot.selfdrive.ui.layouts.home import HomeLayout
from openpilot.system.ui.lib.ui_state import ui_state
from openpilot.system.ui.onroad.augmented_road_view import AugmentedRoadView


class UI:
  def __init__(self):
    self._sidbar = Sidebar()
    self._sidebar_visible = True

    self._home_layout = HomeLayout()
    self._augmented_road_view = AugmentedRoadView()
    self._sidebar_rect = rl.Rectangle(0, 0, 0, 0)
    self._content_rect = rl.Rectangle(0, 0, 0, 0)

  def render(self, rect):
    self._update_layout_rects(rect)
    self._handle_input()
    self._render_main_content()

  def _update_layout_rects(self, rect):
    self._sidebar_rect = rl.Rectangle(
      rect.x,
      rect.y,
      SIDEBAR_WIDTH,
      rect.height
    )

    x_offset = SIDEBAR_WIDTH if self._sidebar_visible else 0
    self._content_rect = rl.Rectangle(
      rect.y + x_offset,
      rect.y,
      rect.width - x_offset,
      rect.height
    )

  def _render_main_content(self):
    if self._sidebar_visible:
      self._sidbar.render(self._sidebar_rect)

    if ui_state.started:
      self._augmented_road_view.render(self._content_rect)
    else:
      self._home_layout.render(self._content_rect)

  def _handle_input(self):
    mouse_pos = rl.get_mouse_position()
    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if rl.check_collision_point_rec(mouse_pos, self._content_rect):
        self._sidebar_visible = not self._sidebar_visible


def main():
  gui_app.init_window("UI")
  ui = UI()
  for _ in gui_app.render():
    ui_state.update()
    ui.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))


if __name__ == "__main__":
  main()
