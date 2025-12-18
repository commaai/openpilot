#!/usr/bin/env python3
"""Standalone raylib video player"""
import os
import sys
import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.video_player import VideoPlayer


def main():
  if len(sys.argv) < 2:
    print("Usage: video_player.py <video_file>")
    print(f"Example: video_player.py ~/Downloads/clip-8864d9dac766cb5f|2022-12-03--20-22-31--17.mp4")
    sys.exit(1)

  video_path = os.path.expanduser(sys.argv[1])

  if not os.path.exists(video_path):
    print(f"Error: Video file not found: {video_path}")
    sys.exit(1)

  gui_app.init_window("Video Player")
  video_player = VideoPlayer(video_path)
  video_player.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))

  for should_render in gui_app.render():
    if should_render:
      video_player.render()


if __name__ == "__main__":
  main()

