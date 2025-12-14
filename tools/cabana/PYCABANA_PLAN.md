# pycabana: Python Qt Rewrite Plan

## Overview

Rewrite `tools/cabana` from C++ Qt (~9k LOC) to Python using PySide2, maintaining 100% feature parity.

**Strategy: Incremental migration** - Use `shiboken2.wrapInstance()` to wrap existing C++ widgets, then replace them one-by-one with pure PySide2 implementations. This keeps the app working at every step.

**Guiding principle: when in doubt, match C++ cabana.**

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Qt Binding | **PySide2** | pip-installable, LGPL, official Qt binding |
| C++ Interop | **shiboken2 + Cython** | `wrapInstance()` wraps C++ QWidgets as Python objects |
| Charting | **vispy** (later) | GPU-accelerated, for when we rewrite ChartsWidget |
| DBC | **opendbc.can** | Already exists: `DBC`, `CANParser`, `get_raw_value` |
| Messaging | **cereal.messaging** | Already exists: `SubMaster`, `PubMaster` |
| Video | **msgq.visionipc** | Already exists: `VisionIpcClient` + OpenGL |
| Arrays | **NumPy** | Fast signal decoding, chart data buffers |
| Package | **root pyproject.toml** | Use existing openpilot packaging |

## Migration Strategy

### The Key Insight

PySide2 and C++ Qt are the same thing under the hood. A C++ `QMainWindow*` can be wrapped as a Python `QMainWindow` using `shiboken2.wrapInstance(ptr, QMainWindow)`. This means:

1. Python controls the app lifecycle (QApplication, event loop)
2. C++ widgets work inside Python's Qt world
3. We can replace widgets incrementally - swap a C++ widget for a PySide2 one, the rest keeps working

### Phase 0: Python Launches C++ Cabana (THIS PR)

**Goal:** `python -m openpilot.tools.cabana.pycabana` launches the full cabana UI, with Python in control.

**Files to create:**

```
tools/cabana/pycabana/
├── __init__.py          # Package init
├── __main__.py          # Entry point for `python -m`
├── main.py              # Main function, orchestrates everything
├── _cabana.cpp          # C helper: creates QApp, streams, MainWindow
└── bindings.pyx         # Cython bindings to _cabana.cpp
```

**_cabana.cpp** exposes simple C functions:

```cpp
extern "C" {
    // Initialize QApplication, parse args, return stream pointer (or 0 on failure)
    void* pycabana_init(int argc, char** argv, const char** dbc_out);

    // Create MainWindow with stream, return pointer (or 0 on failure)
    void* pycabana_create_main_window(void* stream, const char* dbc);

    // Run Qt event loop, return exit code
    int pycabana_exec();
}
```

**bindings.pyx** wraps these for Python:

```cython
def init(args: list[str]) -> tuple[int, str]:
    """Initialize app, return (stream_ptr, dbc_file)"""

def create_main_window(stream_ptr: int, dbc: str = "") -> int:
    """Create MainWindow, return pointer"""

def run() -> int:
    """Run event loop"""
```

**main.py** ties it together:

```python
import sys
from PySide2.QtWidgets import QMainWindow
import shiboken2
from .bindings import init, create_main_window, run

def main():
    stream_ptr, dbc = init(sys.argv)
    if not stream_ptr:
        return 1

    win_ptr = create_main_window(stream_ptr, dbc)
    if not win_ptr:
        return 1

    # Wrap C++ MainWindow as PySide2 QMainWindow
    window = shiboken2.wrapInstance(win_ptr, QMainWindow)
    window.show()

    return run()
```

**SConscript changes:**
- Build `_cabana.cpp` as shared library, linked against `cabana_lib`
- Build `bindings.pyx` via existing Cython tooling

**Success criteria for Phase 0:**
- [ ] `python -m openpilot.tools.cabana.pycabana --demo` opens cabana
- [ ] All existing functionality works (it's the same C++ code)
- [ ] Python has a reference to MainWindow as a PySide2 object

### Phase 1: First Python Widget

Pick the simplest widget to rewrite in pure PySide2. Good candidates:
- `HelpOverlay` - just draws text, minimal logic
- `StatusBar` components - simple labels

**Pattern for replacement:**
1. Write PySide2 version of widget
2. In Python, after getting MainWindow, find the C++ widget and replace it:
   ```python
   # Example: replace a dock widget's contents
   old_widget = main_window.findChild(QDockWidget, "messages_dock")
   new_widget = MyPySide2MessagesWidget()
   old_widget.setWidget(new_widget)
   ```

### Phase 2+: Widget-by-Widget Migration

Rough order (simplest to hardest):
1. **HelpOverlay** - overlay widget, standalone
2. **StatusBar** - labels and progress bar
3. **MessagesWidget** - QTreeView + model (medium complexity)
4. **DetailWidget** - container for tabs
5. **BinaryView** - custom painting
6. **SignalView** - QTreeView + custom delegate
7. **HistoryLog** - QTableView + model
8. **VideoWidget** - OpenGL, keep C++ longer
9. **ChartsWidget** - complex, replace with vispy last

### Final State

Eventually all widgets are pure PySide2. At that point:
- Remove C++ widget code
- Remove Cython bindings (no longer needed)
- Keep only Python code

## Architecture (Target State)

```
pycabana/
├── __init__.py
├── __main__.py
├── main.py                 # Entry point, MainWindow
├── streams/
│   ├── __init__.py
│   ├── abstract.py         # AbstractStream base class
│   ├── replay.py           # ReplayStream (wraps tools/replay)
│   ├── device.py           # DeviceStream (ZMQ/SubMaster)
│   └── panda.py            # PandaStream (USB via python-panda)
├── dbc/
│   ├── __init__.py
│   └── manager.py          # DBCManager (wraps opendbc.can.DBC)
├── chart/
│   ├── __init__.py
│   ├── widget.py           # ChartsWidget container
│   └── view.py             # ChartView (vispy-based)
├── widgets/
│   ├── __init__.py
│   ├── messages.py         # MessagesWidget
│   ├── signals.py          # SignalView
│   ├── binary.py           # BinaryView
│   ├── history.py          # HistoryLog
│   └── video.py            # VideoWidget
└── utils/
    ├── __init__.py
    ├── settings.py         # QSettings wrapper
    └── helpers.py          # Formatting, colors, etc.
```

## Build Integration

### SConscript additions

```python
# Build pycabana C++ helper
pycabana_env = cabana_env.Clone()
pycabana_env.Append(CXXFLAGS=['-fPIC'])

pycabana_helper = pycabana_env.SharedLibrary(
    'pycabana/_cabana',
    ['pycabana/_cabana.cpp', cabana_lib],
    LIBS=cabana_libs,
    FRAMEWORKS=base_frameworks
)

# Build Cython bindings
envCython.CythonModule(
    'pycabana/bindings',
    'pycabana/bindings.pyx',
    LIBS=['_cabana'] + cabana_libs
)
```

## Dependencies

Already in pyproject.toml:
- `pyside2` (includes `shiboken2`)

Already available from openpilot:
- `numpy`
- `opendbc`
- `cereal`
- `msgq`
- `panda`

Add later (for chart rewrite):
- `vispy>=0.14`

## Resolved Decisions

1. **Migration strategy** - Incremental via `shiboken2.wrapInstance()`, not full rewrite

2. **C++ interop** - Cython + simple `extern "C"` functions (not full Shiboken bindings)

3. **Arg parsing** - Keep in C++ for now, pass `sys.argv` through

4. **Error handling** - Return `nullptr`/`0` on failure, Python checks

5. **Signal colors** - Match C++ exactly when we rewrite:
   ```python
   h = (19 * lsb / 64.0) % 1.0
   s = 0.25 + 0.25 * (hash(name) & 0xff) / 255.0
   v = 0.75 + 0.25 * ((hash(name) >> 8) & 0xff) / 255.0
   ```

## Success Criteria

### Phase 0 (This PR)
- [ ] `python -m openpilot.tools.cabana.pycabana --demo` works
- [ ] `python -m openpilot.tools.cabana.pycabana <route>` works
- [ ] All C++ functionality preserved

### Final
- [ ] All widgets pure PySide2
- [ ] All C++ features working
- [ ] 10+ FPS with active charting
- [ ] 1-hour replay handles smoothly
- [ ] Live streams at full CAN bus rate
