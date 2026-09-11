# ui

The user interfaces here are built with [raylib](https://www.raylib.com/).

Quick start:
* set `BIG=1` to run the comma 3X UI (comma four UI runs by default)
* set `SHOW_FPS=1` to show the FPS
* set `STRICT_MODE=1` to kill the app if it drops too much below 60fps
* set `SCALE=1.5` to scale the entire UI by 1.5x
* set `BURN_IN=1` to get a burn-in heatmap version of the UI
* set `GRID=50` to show a 50-pixel alignment grid overlay
* set `MAGIC_DEBUG=1` to show every dropped frames (only on device)
* set `RECORD=1` to record the screen, output defaults to `output.mp4` but can be set with `RECORD_OUTPUT`
* set `PRIME_TYPE` to preview pairing/Prime status: `-1` unpaired, `0` paired without Prime, `2` Prime Lite, `1` Prime Full
* set `PAIRING_PROVIDER=github`, `google`, or `apple` to preview the paired account icon without an account lookup
* with `PRIME_TYPE=0`, set `PRIME_TRIAL_CLAIMED=0` to preview "claim prime trial", or `1` for "upgrade to prime"
* https://www.raylib.com/cheatsheet/cheatsheet.html
* https://electronstudio.github.io/raylib-python-cffi/README.html#quickstart

For example, preview a Google account paired without Prime on desktop:

```sh
PRIME_TYPE=0 PAIRING_PROVIDER=google python -m openpilot.selfdrive.ui.ui
```

Style guide:
* All graphical elements should subclass [`Widget`](/openpilot/system/ui/widgets/__init__.py).
  * Prefer a stateful widget over a function for easy migration from QT
* All internal class variables and functions should be prefixed with `_`
