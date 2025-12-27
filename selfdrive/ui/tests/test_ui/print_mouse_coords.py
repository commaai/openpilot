#!/usr/bin/env python3
"""
Simple script to print mouse coordinates on Ubuntu.
Run with: python print_mouse_coords.py
Press Ctrl+C to exit.
"""

from pynput import mouse

print("Mouse coordinate printer - Press Ctrl+C to exit")
print("Click to set the top left origin")

origin: tuple[int, int] | None = None
clicks: list[tuple[int, int]] = []


def on_click(x, y, button, pressed):
  global origin, clicks
  if pressed:  # Only on mouse down, not up
    if origin is None:
      origin = (x, y)
      print(f"Origin set to: {x},{y}")
    else:
      rel_x = x - origin[0]
      rel_y = y - origin[1]
      clicks.append((rel_x, rel_y))
      print(f"Clicks: {clicks}")


if __name__ == "__main__":
  try:
    # Start mouse listener
    with mouse.Listener(on_click=on_click) as listener:
      listener.join()
  except KeyboardInterrupt:
    print("\nExiting...")
