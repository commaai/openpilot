# pycabana: Python Qt Rewrite Plan

## Overview
Rewrite `tools/cabana` from C++ Qt (~9k LOC) to Python using PySide6, maintaining 100% feature parity.

**Guiding principle: when in doubt, match C++ cabana.**

## Tech Stack
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Qt Binding | **PySide6** | pip-installable, LGPL, official Qt binding |
| Charting | **vispy** | GPU-accelerated, handles real-time data well |
| DBC | **opendbc.can** | Already exists: `DBC`, `CANParser`, `get_raw_value` |
| Messaging | **cereal.messaging** | Already exists: `SubMaster`, `PubMaster` |
| Video | **msgq.visionipc** | Already exists: `VisionIpcClient` + OpenGL |
| Arrays | **NumPy** | Fast signal decoding, chart data buffers |
| Package | **root pyproject.toml** | Use existing openpilot packaging, no separate pyproject |

## Architecture

```
pycabana/
├── __init__.py
├── main.py                 # Entry point, MainWindow
├── streams/
│   ├── __init__.py
│   ├── abstract.py         # AbstractStream base class
│   ├── replay.py           # ReplayStream (LogReader-based)
│   ├── device.py           # DeviceStream (ZMQ/SubMaster)
│   ├── panda.py            # PandaStream (USB via python-panda)
│   └── socketcan.py        # SocketCanStream (python-can)
├── dbc/
│   ├── __init__.py
│   ├── manager.py          # DBCManager (wraps opendbc.can.DBC)
│   └── models.py           # Extended Signal/Msg if needed
├── chart/
│   ├── __init__.py
│   ├── widget.py           # ChartsWidget container
│   ├── view.py             # ChartView (vispy-based)
│   ├── segment_tree.py     # SegmentTree for O(log n) min/max
│   └── sparkline.py        # Sparkline mini-charts
├── widgets/
│   ├── __init__.py
│   ├── messages.py         # MessagesWidget (message list)
│   ├── signals.py          # SignalView (signal tree)
│   ├── binary.py           # BinaryView (bit visualization)
│   ├── history.py          # HistoryLog (time-series table)
│   └── video.py            # VideoWidget + CameraWidget (OpenGL)
├── tools/
│   ├── __init__.py
│   ├── find_signal.py      # Signal discovery tool
│   ├── find_similar.py     # Find similar bits tool
│   └── route_info.py       # Route metadata display
├── utils/
│   ├── __init__.py
│   ├── settings.py         # Settings (QSettings-based)
│   ├── theme.py            # Dark/light theme support
│   └── helpers.py          # Formatting, icons, etc.
└── commands.py             # Undo/redo command classes
```

## Component Mapping (C++ → Python)

### Core Data Flow
```
C++: AbstractStream → events_ (MessageEventsMap) → ChartView
Python: AbstractStream → events (dict[MessageId, list[CanEvent]]) → ChartView (vispy)
```

### Key Classes

| C++ Class | Python Class | Notes |
|-----------|--------------|-------|
| `AbstractStream` | `streams.AbstractStream` | Base class, same interface |
| `ReplayStream` | `streams.ReplayStream` | Use `LogReader` from tools/lib |
| `DeviceStream` | `streams.DeviceStream` | Use `SubMaster(['can', 'sendcan'])` |
| `PandaStream` | `streams.PandaStream` | Use `panda` Python package |
| `DBCManager` | `dbc.DBCManager` | Wrap `opendbc.can.DBC` |
| `cabana::Signal` | Extend `opendbc.can.dbc.Signal` | Add color, UI state |
| `ChartView` | `chart.ChartView` | vispy.scene.SceneCanvas |
| `SegmentTree` | `chart.SegmentTree` | Pure Python + NumPy |
| `CameraWidget` | `widgets.CameraWidget` | QOpenGLWidget + visionipc |
| `BinaryView` | `widgets.BinaryView` | Custom QWidget paint |

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] `AbstractStream` base class with threading model
- [ ] `CanEvent` / `CanData` data structures (dataclasses + NumPy)
- [ ] `DBCManager` wrapping opendbc
- [ ] Basic `MainWindow` shell

### Phase 2: Replay Stream
- [ ] `ReplayStream` using `LogReader`
- [ ] Event buffering and time-range filtering
- [ ] Playback controls (play/pause/seek/speed)

### Phase 3: Message & Signal Views
- [ ] `MessagesWidget` - QTableView with model
- [ ] `SignalView` - QTreeView with model
- [ ] `BinaryView` - Custom paint, bit selection
- [ ] `HistoryLog` - Time-series table

### Phase 4: Charting (vispy)
- [ ] `ChartView` - vispy SceneCanvas
- [ ] Line/step/scatter series types
- [ ] `SegmentTree` for efficient Y-axis scaling
- [ ] Zoom/pan with rubber band selection
- [ ] `ChartsWidget` - Multi-chart container with tabs
- [ ] `Sparkline` - Mini inline charts

### Phase 5: Video Integration
- [ ] `CameraWidget` - QOpenGLWidget subclass
- [ ] VisionIPC client in separate QThread
- [ ] YUV→RGB shader (port from C++)
- [ ] Timeline sync with CAN data
- [ ] Thumbnail generation

### Phase 6: Live Streams
- [ ] `DeviceStream` - SubMaster integration
- [ ] `PandaStream` - USB via panda package
- [ ] `SocketCanStream` - python-can integration

### Phase 7: DBC Editing
- [ ] Signal add/edit/remove dialogs
- [ ] Message add/edit/remove
- [ ] Undo/redo stack (QUndoStack)
- [ ] DBC file save/load
- [ ] Mask editing

### Phase 8: Analysis Tools
- [ ] Find Signal tool
- [ ] Find Similar Bits tool
- [ ] Route Info dialog
- [ ] CSV export

### Phase 9: Polish
- [ ] Settings dialog
- [ ] Dark/light themes
- [ ] Keyboard shortcuts
- [ ] Window state persistence
- [ ] Performance optimization pass

## Performance Considerations

### Hot Paths
1. **Signal Decoding** - Use NumPy vectorized ops or keep `get_raw_value` from opendbc
2. **Chart Updates** - vispy handles GPU rendering; limit point density
3. **Event Storage** - Pre-allocate NumPy arrays, avoid Python list appends
4. **SegmentTree** - NumPy-based implementation for O(log n) queries

### Threading Model
```
Main Thread (Qt Event Loop)
├── UI rendering
├── User interaction
└── Timer-based updates (10 Hz default)

Stream Thread(s)
├── ReplayStream: LogReader iteration
├── DeviceStream: SubMaster.update()
├── PandaStream: USB polling
└── SocketCanStream: can.Bus.recv()

Video Thread
└── VisionIpcClient frame reception
```

### Memory Management
- Use `collections.deque(maxlen=N)` for bounded event history
- NumPy structured arrays for CanEvent storage
- Lazy loading for replay segments (already in LogReader)

## Dependencies

Add to root pyproject.toml if not already present:
- `PySide6>=6.5`
- `vispy>=0.14`
- `python-can>=4.0` (for SocketCAN)

Already available from openpilot ecosystem:
- `numpy`
- `opendbc` (DBC parsing)
- `cereal` (messaging)
- `msgq` (visionipc)
- `panda` (USB)

## Resolved Decisions

1. **Signal color assignment** - Match C++ exactly:
   ```python
   # From dbc.cc:136-142
   h = (19 * lsb / 64.0) % 1.0
   s = 0.25 + 0.25 * (hash(name) & 0xff) / 255.0
   v = 0.75 + 0.25 * ((hash(name) >> 8) & 0xff) / 255.0
   color = QColor.fromHsvF(h, s, v)
   ```

2. **Package setup** - Use root pyproject.toml, no separate package.

3. **When in doubt** - Match C++ cabana behavior exactly.

## Open Questions

1. **DBC editing persistence** - C++ tracks modified DBCs in memory. Same approach or auto-save?

2. **Chart library fallback** - If vispy causes issues, pyqtgraph is a solid backup.

3. **Test strategy** - Port existing C++ tests or write new pytest suite?

## File Count Estimate
- C++ original: ~71 files, ~9,095 lines
- Python target: ~30-40 files, ~4,000-5,000 lines (Python is more concise)

## Success Criteria
- [ ] All C++ features working
- [ ] UI refresh at 10+ FPS with active charting
- [ ] Handle 1-hour replay smoothly
- [ ] Live streams work at full CAN bus rates
- [ ] Video syncs correctly with CAN data
