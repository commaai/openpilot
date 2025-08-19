# ui

The user interfaces here are built with [raylib](https://www.raylib.com/).

Quick start:
* set `SHOW_FPS=1` to show the FPS
* set `STRICT_MODE=1` to kill the app if it drops too much below 60fps
* set `SCALE=1.5` to scale the entire UI by 1.5x
* https://www.raylib.com/cheatsheet/cheatsheet.html
* https://electronstudio.github.io/raylib-python-cffi/README.html#quickstart

Style guide:
* All graphical elements should subclass [`Widget`](/system/ui/widgets/__init__.py).
  * Prefer a stateful widget over a function for easy migration from QT
* All internal class variables and functions should be prefixed with `_`
