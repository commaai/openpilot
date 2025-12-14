# pycabana - Pure PySide2 Rewrite of Cabana

A complete Python rewrite of cabana using PySide2 and existing openpilot Python tools.

## Approach

- **Pure PySide2** - No C++ interop, no shiboken2 wrapping
- **Leverage existing tools** - Use `LogReader`, `FrameReader`, `Route`, `opendbc.can.dbc`
- **1:1 translation** - Mechanical port of C++ widgets to Python equivalents
- **Piecemeal development** - Build and test widget by widget

## Package Structure

```
tools/cabana/pycabana/
├── __init__.py
├── __main__.py
├── main.py              # Entry point, QApplication setup
├── dbc/
│   ├── __init__.py
│   ├── dbc.py           # MessageId, Signal, Msg dataclasses
│   ├── dbcfile.py       # DBC file I/O (uses opendbc.can.dbc)
│   └── dbcmanager.py    # DBCManager singleton
├── streams/
│   ├── __init__.py
│   ├── abstract.py      # AbstractStream base, CanEvent, CanData
│   └── replay.py        # ReplayStream (uses LogReader, FrameReader)
├── widgets/
│   ├── __init__.py
│   ├── messages.py      # MessagesWidget, MessageListModel
│   ├── binary.py        # BinaryView, BinaryViewModel
│   ├── signal.py        # SignalView, SignalModel
│   ├── history.py       # HistoryLog
│   ├── detail.py        # DetailWidget, CenterWidget
│   ├── video.py         # VideoWidget, Slider
│   ├── camera.py        # CameraView
│   └── charts.py        # ChartsWidget, ChartView
├── dialogs/
│   └── __init__.py
└── utils/
    └── __init__.py
```

## File-by-File Translation

### Core (Tier 1)

| C++ File | Lines | Python Target | Notes |
|----------|-------|---------------|-------|
| `dbc/dbc.h` + `.cc` | 343 | `dbc/dbc.py` | MessageId, Signal, Msg dataclasses |
| `settings.h` + `.cc` | 204 | `settings.py` | QSettings wrapper singleton |
| `dbc/dbcmanager.h` + `.cc` | 247 | `dbc/dbcmanager.py` | Use opendbc.can.dbc for parsing |
| `dbc/dbcfile.h` + `.cc` | 318 | `dbc/dbcfile.py` | DBC file I/O |
| `streams/abstractstream.h` + `.cc` | 482 | `streams/abstract.py` | Base stream, CanEvent, CanData |
| `commands.h` + `.cc` | 196 | `commands.py` | QUndoCommand subclasses |

### Streams (Tier 2)

| C++ File | Lines | Python Target | Notes |
|----------|-------|---------------|-------|
| `streams/replaystream.h` + `.cc` | 233 | `streams/replay.py` | Use LogReader, FrameReader, Route |

### Widgets (Tier 3)

| C++ File | Lines | Python Target | Notes |
|----------|-------|---------------|-------|
| `messageswidget.h` + `.cc` | 585 | `widgets/messages.py` | QAbstractTableModel + QTreeView |
| `binaryview.h` + `.cc` | 611 | `widgets/binary.py` | Custom delegate for bit display |
| `signalview.h` + `.cc` | 869 | `widgets/signal.py` | Tree model with inline editing |
| `historylog.h` + `.cc` | 329 | `widgets/history.py` | Time-filtered table |
| `detailwidget.h` + `.cc` | 365 | `widgets/detail.py` | Container for binary/signal/history |
| `videowidget.h` + `.cc` | 509 | `widgets/video.py` | Timeline slider, playback controls |
| `cameraview.h` + `.cc` | 325 | `widgets/camera.py` | Use FrameReader + QLabel/QPixmap |

### Charts (Tier 4)

| C++ File | Lines | Python Target | Notes |
|----------|-------|---------------|-------|
| `chart/chartswidget.h` + `.cc` | 698 | `widgets/charts.py` | QtCharts container |
| `chart/chart.h` + `.cc` | 985 | `widgets/charts.py` | QtCharts QChartView |
| `chart/sparkline.h` + `.cc` | 126 | `widgets/signal.py` | Inline mini-charts |

### Main (Tier 5)

| C++ File | Lines | Python Target | Notes |
|----------|-------|---------------|-------|
| `mainwin.h` + `.cc` | 765 | `mainwindow.py` | QMainWindow with docks |
| `cabana.cc` | 246 | `main.py` | Entry point, arg parsing |

## Existing Python Tools to Use

| Tool | Location | Usage |
|------|----------|-------|
| LogReader | `tools/lib/logreader.py` | Read rlog/qlog files |
| FrameReader | `tools/lib/framereader.py` | Read video frames |
| Route | `tools/lib/route.py` | Route loading and segment management |
| DBC | `opendbc/can/dbc.py` | Parse DBC files |
| CANParser | `opendbc/can/parser.py` | Decode CAN signals |
| CommaApi | `tools/lib/api.py` | Fetch routes from comma API |

## Implementation Order

1. `dbc/dbc.py` - Core data structures
2. `streams/abstract.py` - Base stream class
3. `streams/replay.py` - Load routes with LogReader
4. `dbc/dbcmanager.py` - DBC management
5. `widgets/messages.py` - Message list
6. `widgets/binary.py` - Binary view
7. `widgets/signal.py` - Signal view
8. `main.py` + `mainwindow.py` - Basic window
9. `widgets/video.py` + `widgets/camera.py` - Video playback
10. `widgets/charts.py` - Signal charts

## Dependencies

```
PySide2
numpy
```

Plus existing openpilot packages in PYTHONPATH.
