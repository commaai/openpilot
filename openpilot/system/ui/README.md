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
* set `RECORD_DEVICE_FRAME=full` or `RECORD_DEVICE_FRAME=faded` with `RECORD=1` to include the comma four hardware on a white background (faded hides the top of the device). Device frames require a full FFmpeg build: set `RECORD_FFMPEG=/usr/bin/ffmpeg` on Linux (the bundled minimal FFmpeg does not include the compositing filters). Raw screen recording remains the default; device frames are not supported with `BIG=1`.
* frame an existing screenshot or recording with `python -m openpilot.system.ui.lib.device_frame input.png output.png --mode faded --ffmpeg /usr/bin/ffmpeg` (use `.mp4` output for video). Images preserve their aspect ratio within the screen aperture. Full frames are 1680×1120; faded frames crop the unused upper area to 1240×700. Video defaults to 30 fps; use `--fps` to match other input frame rates. The frame uses the same product render and screen placement as comma.ai.
* set `PRIME_TYPE` to preview pairing/Prime status: `-1` unpaired, `0` paired without Prime, `2` Prime Lite, `1` Prime Full
* set `PAIRING_PROVIDER` to preview the different states based on connect auth providers `google`, `github` or `apple`
* https://www.raylib.com/cheatsheet/cheatsheet.html
* https://electronstudio.github.io/raylib-python-cffi/README.html#quickstart

Style guide:
* All graphical elements should subclass [`Widget`](/openpilot/system/ui/widgets/__init__.py).
  * Prefer a stateful widget over a function for easy migration from QT
* All internal class variables and functions should be prefixed with `_`
