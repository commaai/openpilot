#!/usr/bin/env python3
import argparse
import os
import pyautogui
import subprocess
import dearpygui.dearpygui as dpg
import threading
import numpy as np
from openpilot.common.basedir import BASEDIR
from openpilot.tools.jotpluggler.data import DataManager
from openpilot.tools.jotpluggler.views import DataTreeView
from openpilot.tools.jotpluggler.layout import PlotLayoutManager, SplitterNode, LeafNode

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"


class PlaybackManager:
  def __init__(self):
    self.is_playing = False
    self.current_time_s = 0.0
    self.duration_s = 0.0

  def set_route_duration(self, duration: float):
    self.duration_s = duration
    self.seek(min(self.current_time_s, duration))

  def toggle_play_pause(self):
    if not self.is_playing and self.current_time_s >= self.duration_s:
      self.seek(0.0)
    self.is_playing = not self.is_playing

  def seek(self, time_s: float):
    self.is_playing = False
    self.current_time_s = max(0.0, min(time_s, self.duration_s))

  def update_time(self, delta_t: float):
    if self.is_playing:
      self.current_time_s = min(self.current_time_s + delta_t, self.duration_s)
      if self.current_time_s >= self.duration_s:
        self.is_playing = False
    return self.current_time_s


def calculate_avg_char_width(font):
  sample_text = "abcdefghijklmnopqrstuvwxyz0123456789"
  if size := dpg.get_text_size(sample_text, font=font):
    return size[0] / len(sample_text)
  return None


def format_and_truncate(value, available_width: float, avg_char_width: float) -> str:
  s = f"{value:.5f}" if np.issubdtype(type(value), np.floating) else str(value)
  max_chars = int(available_width / avg_char_width) - 3
  if len(s) > max_chars:
    return s[: max(0, max_chars)] + "..."
  return s


class MainController:
  def __init__(self, scale: float = 1.0):
    self.ui_lock = threading.Lock()
    self.scale = scale
    self.data_manager = DataManager()
    self.playback_manager = PlaybackManager()
    self._create_global_themes()
    self.data_tree_view = DataTreeView(self.data_manager, self.ui_lock)
    self.plot_layout_manager = PlotLayoutManager(self.data_manager, self.playback_manager, scale=self.scale)
    self.data_manager.add_observer(self.on_data_loaded)
    self.avg_char_width = None
    self.visible_paths: set[str] = set()
    self.check_index = 0

  def _create_global_themes(self):
    with dpg.theme(tag="global_line_theme"):
      with dpg.theme_component(dpg.mvLineSeries):
        scaled_thickness = max(1.0, self.scale)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, scaled_thickness, category=dpg.mvThemeCat_Plots)

    with dpg.theme(tag="global_timeline_theme"):
      with dpg.theme_component(dpg.mvInfLineSeries):
        scaled_thickness = max(1.0, self.scale)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, scaled_thickness, category=dpg.mvThemeCat_Plots)
        dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 0, 0, 128), category=dpg.mvThemeCat_Plots)

  def on_data_loaded(self, data: dict):
    duration = data.get('duration', 0.0)
    self.playback_manager.set_route_duration(duration)

    if data.get('loading_complete'):
      num_paths = len(self.data_manager.get_all_paths())
      dpg.set_value("load_status", f"Loaded {num_paths} data paths")
      dpg.configure_item("load_button", enabled=True)
    elif data.get('segment_added'):
      segment_count = data.get('segment_count', 0)
      dpg.set_value("load_status", f"Loading... {segment_count} segments processed")

    dpg.configure_item("timeline_slider", max_value=duration)

  def setup_ui(self):
    dpg.set_viewport_resize_callback(callback=self.on_viewport_resize)

    with dpg.window(tag="Primary Window"):
      with dpg.group(horizontal=True):
        # Left panel - Data tree
        with dpg.child_window(label="Data Pool", width=300 * self.scale, tag="data_pool_window", border=True, resizable_x=True):
          with dpg.group(horizontal=True):
            dpg.add_input_text(tag="route_input", width=-75 * self.scale, hint="Enter route name...")
            dpg.add_button(label="Load", callback=self.load_route, tag="load_button", width=-1)
          dpg.add_text("Ready to load route", tag="load_status")
          dpg.add_separator()
          dpg.add_text("Available Data")
          dpg.add_separator()
          dpg.add_input_text(tag="search_input", width=-1, hint="Search fields...", callback=self.search_data)
          dpg.add_separator()
          with dpg.group(tag="data_tree_container", track_offset=True):
            pass

        # Right panel - Plots and timeline
        with dpg.group():
          with dpg.child_window(label="Plot Window", border=True, height=-(30 + 13 * self.scale), tag="main_plot_area"):
            self.plot_layout_manager.create_ui("main_plot_area")

          with dpg.child_window(label="Timeline", border=True):
            with dpg.table(header_row=False, borders_innerH=False, borders_innerV=False, borders_outerH=False, borders_outerV=False):
              dpg.add_table_column(width_fixed=True, init_width_or_weight=int(50 * self.scale))  # Play button
              dpg.add_table_column(width_stretch=True)  # Timeline slider
              dpg.add_table_column(width_fixed=True, init_width_or_weight=int(50 * self.scale))  # FPS counter
              with dpg.table_row():
                dpg.add_button(label="Play", tag="play_pause_button", callback=self.toggle_play_pause, width=int(50 * self.scale))
                dpg.add_slider_float(tag="timeline_slider", default_value=0.0, label="", width=-1, callback=self.timeline_drag)
                dpg.add_text("", tag="fps_counter")

    dpg.set_primary_window("Primary Window", True)

  def on_viewport_resize(self):
    self.plot_layout_manager.on_viewport_resize()

  def load_route(self):
    route_name = dpg.get_value("route_input").strip()
    if route_name:
      dpg.set_value("load_status", "Loading route...")
      dpg.configure_item("load_button", enabled=False)
      self.data_manager.load_route(route_name)

  def search_data(self):
    search_term = dpg.get_value("search_input")
    self.data_tree_view.search_data(search_term)

  def toggle_play_pause(self, sender):
    self.playback_manager.toggle_play_pause()
    label = "Pause" if self.playback_manager.is_playing else "Play"
    dpg.configure_item(sender, label=label)

  def timeline_drag(self, sender, app_data):
    self.playback_manager.seek(app_data)
    dpg.configure_item("play_pause_button", label="Play")

  def update_frame(self, font):
    with self.ui_lock:
      if self.avg_char_width is None:
        self.avg_char_width = calculate_avg_char_width(font)
      self.data_tree_view.update_frame()

      new_time = self.playback_manager.update_time(dpg.get_delta_time())
      if not dpg.is_item_active("timeline_slider"):
        dpg.set_value("timeline_slider", new_time)

      self._update_timeline_indicators(new_time)

      if self.avg_char_width:
        self._update_visible_set()
        self._update_data_values()

      dpg.set_value("fps_counter", f"{dpg.get_frame_rate():.1f} FPS")

  def _update_visible_set(self): # for some reason, dpg has no way to easily check for visibility, and checking is slow...
    all_paths = list(self.data_tree_view.created_leaf_paths)
    if not all_paths:
      self.visible_paths.clear()
      return
    chunk_size = min(50, len(all_paths)) # check up to 50 paths per frame
    end_index = min(self.check_index + chunk_size, len(all_paths))
    for i in range(self.check_index, end_index):
      path = all_paths[i]
      value_tag = f"value_{path}"
      if dpg.does_item_exist(value_tag) and dpg.is_item_visible(value_tag):
        self.visible_paths.add(path)
      else:
        self.visible_paths.discard(path)
    self.check_index = end_index if end_index < len(all_paths) else 0

  def _update_data_values(self):
    value_column_width = dpg.get_item_rect_size("data_pool_window")[0] // 2

    for path in self.visible_paths.copy(): # avoid modification during iteration
      value_tag = f"value_{path}"
      group_tag = f"group_{path}"

      if not dpg.does_item_exist(value_tag) or not dpg.does_item_exist(group_tag):
        self.visible_paths.discard(path)
        continue

      dpg.configure_item(group_tag, xoffset=value_column_width)
      value = self.data_manager.get_value_at(path, self.playback_manager.current_time_s)
      if value is not None:
        formatted_value = format_and_truncate(value, value_column_width, self.avg_char_width)
        dpg.set_value(value_tag, formatted_value)

  def _update_timeline_indicators(self, current_time_s: float):
    def update_node_recursive(node):
      if isinstance(node, LeafNode):
        if hasattr(node.panel, 'update_timeline_indicator'):
          node.panel.update_timeline_indicator(current_time_s)
      elif isinstance(node, SplitterNode):
        for child in node.children:
          update_node_recursive(child)

    if self.plot_layout_manager.root_node:
      update_node_recursive(self.plot_layout_manager.root_node)


def main(route_to_load=None):
  dpg.create_context()

  # TODO: find better way of calculating display scaling
  try:
    w, h = next(tuple(map(int, l.split()[0].split('x'))) for l in subprocess.check_output(['xrandr']).decode().split('\n') if '*' in l)  # actual resolution
    scale = pyautogui.size()[0] / w  # scaled resolution
  except Exception:
    scale = 1

  with dpg.font_registry():
    default_font = dpg.add_font(os.path.join(BASEDIR, "selfdrive/assets/fonts/Inter-Regular.ttf"), int(13 * scale))
  dpg.bind_font(default_font)

  viewport_width, viewport_height = int(1200 * scale), int(800 * scale)
  mouse_x, mouse_y = pyautogui.position()  # TODO: find better way of creating the window where the user is (default dpg behavior annoying on multiple displays)
  dpg.create_viewport(
    title='JotPluggler', width=viewport_width, height=viewport_height, x_pos=mouse_x - viewport_width // 2, y_pos=mouse_y - viewport_height // 2
  )
  dpg.setup_dearpygui()

  controller = MainController(scale=scale)
  controller.setup_ui()

  if route_to_load:
    dpg.set_value("route_input", route_to_load)
    controller.load_route()

  dpg.show_viewport()

  # Main loop
  while dpg.is_dearpygui_running():
    controller.update_frame(default_font)
    dpg.render_dearpygui_frame()

  dpg.destroy_context()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A tool for visualizing openpilot logs.")
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("route", nargs='?', default=None, help="Optional route name to load on startup.")
  args = parser.parse_args()
  route = DEMO_ROUTE if args.demo else args.route
  main(route_to_load=route)
