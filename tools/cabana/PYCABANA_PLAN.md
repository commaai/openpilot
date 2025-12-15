# pycabana - Pure PySide6 Rewrite of Cabana

A complete Python rewrite of cabana using PySide6 and existing openpilot Python tools.

## Approach

- **Pure PySide6** - No C++ interop, no shiboken2 wrapping
- **Leverage existing tools** - Use `LogReader`, `FrameReader`, `Route`, `opendbc.can.dbc`
- **1:1 translation** - Mechanical port of C++ widgets to Python equivalents
- **Piecemeal development** - Build and test widget by widget

## Code Style

- **2-space indentation** (matches openpilot Python code)
- **Type hints** where helpful
- only use comments where absolutely necessary
- **Dataclasses** for data structures

---

## v0.1 Roadmap

### Goal
A minimal cabana that can load a route, display CAN messages, and show decoded signals from a DBC file. No live replay, no video, no charts.

### v0.1a: Route Log Viewer (~450 lines)

**What it does:**
- Load a real openpilot route via command line
- Display all CAN messages in a table (Address, Bus, Count, Freq, Last Data)
- Live-ish updates as data loads

**Files to create:**

```
pycabana/
├── dbc/
│   └── dbc.py           # MessageId, CanEvent, CanData dataclasses
├── streams/
│   ├── abstract.py      # AbstractStream base class with signals
│   └── replay.py        # ReplayStream using LogReader
├── widgets/
│   └── messages.py      # MessagesWidget + MessageListModel
├── main.py              # Entry point, arg parsing
└── mainwindow.py        # MainWindow with messages dock
```

**Data flow:**
```
LogReader → CanEvent objects → AbstractStream.events_ dict
                                      ↓
                              msgsReceived signal
                                      ↓
                              MessagesWidget updates table
```

### v0.1b: Add DBC Decoding (~250 more lines)

**What it adds:**
- Load DBC file (from fingerprint or --dbc flag)
- Show message names in the table
- Click a message → see its signals + last values in a detail panel

**Files to add/modify:**

```
pycabana/
├── dbc/
│   ├── dbc.py           # Add Signal, Msg classes (wrap opendbc types)
│   └── dbcmanager.py    # DBCManager singleton, loads DBC files
├── widgets/
│   ├── messages.py      # Add Name column, use DBC for display
│   └── signal.py        # SignalView - simple table of signal values
└── mainwindow.py        # Add detail panel, connect selection
```

**Data flow:**
```
User clicks message → msgSelectionChanged signal
                              ↓
                      DetailWidget shows SignalView
                              ↓
                      SignalView reads last CanData + DBC
                              ↓
                      Displays signal names + decoded values
```

### v0.1 Final Structure

```
tools/cabana/pycabana/
├── __init__.py
├── __main__.py
├── main.py              # Entry point, QApplication, arg parsing
├── mainwindow.py        # MainWindow with docks
├── dbc/
│   ├── __init__.py
│   ├── dbc.py           # MessageId, CanEvent, CanData, Signal, Msg
│   └── dbcmanager.py    # DBCManager singleton
├── streams/
│   ├── __init__.py
│   ├── abstract.py      # AbstractStream base class
│   └── replay.py        # ReplayStream (LogReader-based)
├── widgets/
│   ├── __init__.py
│   ├── messages.py      # MessagesWidget, MessageListModel
│   └── signal.py        # SignalView (simple signal table)
├── dialogs/
│   └── __init__.py
└── utils/
    └── __init__.py
```

---

## v0.1a Implementation Details

### Step 1: `dbc/dbc.py` - Core Data Structures

```python
@dataclass(frozen=True)
class MessageId:
  source: int = 0
  address: int = 0

  def __hash__(self): ...
  def __str__(self): return f"{self.source}:{self.address:X}"

@dataclass
class CanEvent:
  src: int
  address: int
  mono_time: int  # nanoseconds
  dat: bytes

@dataclass
class CanData:
  """Processed message data for display."""
  ts: float = 0.0          # last timestamp in seconds
  count: int = 0           # total message count
  freq: float = 0.0        # messages per second
  dat: bytes = b''         # last data bytes

  def update(self, event: CanEvent, start_ts: int): ...
```

### Step 2: `streams/abstract.py` - Base Stream

```python
class AbstractStream(QObject):
  msgsReceived = Signal(set, bool)  # (msg_ids, has_new_ids)
  seekedTo = Signal(float)
  streamStarted = Signal()

  def __init__(self):
    self.events: dict[MessageId, list[CanEvent]] = {}
    self.last_msgs: dict[MessageId, CanData] = {}
    self.start_ts: int = 0

  def start(self): ...
  def stop(self): ...
  def lastMessage(self, msg_id: MessageId) -> CanData | None: ...
  def allEvents(self) -> list[CanEvent]: ...

  def updateEvent(self, event: CanEvent):
    """Process a single CAN event, update last_msgs."""
    ...
```

### Step 3: `streams/replay.py` - Route Loading

```python
class ReplayStream(AbstractStream):
  def __init__(self):
    super().__init__()
    self.lr: LogReader | None = None
    self.route_name: str = ""

  def loadRoute(self, route: str, segment: int = 0) -> bool:
    """Load route using LogReader, populate events."""
    from openpilot.tools.lib.logreader import LogReader
    self.lr = LogReader(route)
    for msg in self.lr:
      if msg.which() == 'can':
        for c in msg.can:
          event = CanEvent(c.src, c.address, msg.logMonoTime, bytes(c.dat))
          self.updateEvent(event)
    self.msgsReceived.emit(set(self.last_msgs.keys()), True)
    return True
```

### Step 4: `widgets/messages.py` - Message Table

```python
class MessageListModel(QAbstractTableModel):
  """Model for the messages table."""
  COLUMNS = ['Address', 'Bus', 'Count', 'Freq', 'Data']

  def __init__(self, stream: AbstractStream):
    self.stream = stream
    self.msg_ids: list[MessageId] = []

  def rowCount(self, parent=None): return len(self.msg_ids)
  def columnCount(self, parent=None): return len(self.COLUMNS)
  def data(self, index, role): ...
  def headerData(self, section, orientation, role): ...

  def updateMessages(self, msg_ids: set[MessageId], has_new: bool):
    """Called when stream emits msgsReceived."""
    ...

class MessagesWidget(QWidget):
  msgSelectionChanged = Signal(object)  # MessageId or None

  def __init__(self, stream: AbstractStream):
    self.model = MessageListModel(stream)
    self.view = QTableView()
    self.view.setModel(self.model)
    self.filter_input = QLineEdit()  # for filtering
    ...
```

### Step 5: `mainwindow.py` - Main Window

```python
class MainWindow(QMainWindow):
  def __init__(self, stream: AbstractStream):
    self.stream = stream
    self.setWindowTitle("pycabana")

    # Messages dock (left)
    self.messages_widget = MessagesWidget(stream)
    self.messages_dock = QDockWidget("Messages")
    self.messages_dock.setWidget(self.messages_widget)
    self.addDockWidget(Qt.LeftDockWidgetArea, self.messages_dock)

    # Central widget (placeholder for now)
    self.setCentralWidget(QLabel("Select a message"))

    # Status bar
    self.status_label = QLabel()
    self.statusBar().addWidget(self.status_label)

    # Connect signals
    self.stream.msgsReceived.connect(self._onMsgsReceived)

  def _onMsgsReceived(self, msg_ids, has_new):
    count = sum(self.stream.last_msgs[m].count for m in self.stream.last_msgs)
    self.status_label.setText(f"{len(self.stream.last_msgs)} messages, {count} events")
```

### Step 6: `main.py` - Entry Point

```python
def main():
  app = QApplication(sys.argv)
  app.setApplicationName("pycabana")

  parser = argparse.ArgumentParser()
  parser.add_argument('route', nargs='?', help='Route to load')
  parser.add_argument('--demo', action='store_true')
  args = parser.parse_args()

  route = args.route or (DEMO_ROUTE if args.demo else None)

  stream = ReplayStream()
  if route:
    if not stream.loadRoute(route):
      print(f"Failed to load route: {route}")
      return 1

  window = MainWindow(stream)
  window.show()
  return app.exec_()
```

---

## v0.1b Implementation Details

### Step 7: `dbc/dbc.py` - Add Signal/Msg (extend)

```python
@dataclass
class Signal:
  """Wraps opendbc.can.dbc.Signal with cabana-specific additions."""
  name: str
  start_bit: int
  size: int
  is_signed: bool
  factor: float
  offset: float
  is_little_endian: bool
  # ... other fields

  def getValue(self, data: bytes) -> float | None:
    """Decode signal value from CAN data."""
    ...

@dataclass
class Msg:
  """Wraps opendbc.can.dbc.Msg."""
  address: int
  name: str
  size: int
  signals: dict[str, Signal]
```

### Step 8: `dbc/dbcmanager.py` - DBC Management

```python
class DBCManager(QObject):
  """Singleton that manages loaded DBC files."""
  _instance: 'DBCManager | None' = None

  msgUpdated = Signal(object)  # MessageId
  dbcLoaded = Signal(str)      # dbc name

  def __init__(self):
    self.dbc: DBC | None = None
    self.msgs: dict[int, Msg] = {}  # address -> Msg

  @classmethod
  def instance(cls) -> 'DBCManager':
    if cls._instance is None:
      cls._instance = DBCManager()
    return cls._instance

  def load(self, dbc_name: str) -> bool:
    """Load DBC file using opendbc.can.dbc.DBC."""
    from opendbc.can.dbc import DBC
    self.dbc = DBC(dbc_name)
    # Convert to our Msg/Signal types
    ...
    self.dbcLoaded.emit(dbc_name)
    return True

  def msg(self, msg_id: MessageId) -> Msg | None:
    """Get message definition by ID."""
    return self.msgs.get(msg_id.address)

# Global accessor
def dbc() -> DBCManager:
  return DBCManager.instance()
```

### Step 9: `widgets/signal.py` - Signal View

```python
class SignalView(QTableWidget):
  """Simple table showing signal names and values."""

  def __init__(self):
    self.setColumnCount(2)
    self.setHorizontalHeaderLabels(['Signal', 'Value'])

  def setMessage(self, msg_id: MessageId, can_data: CanData):
    """Update to show signals for the given message."""
    msg = dbc().msg(msg_id)
    if not msg:
      self.setRowCount(0)
      return

    self.setRowCount(len(msg.signals))
    for i, (name, sig) in enumerate(msg.signals.items()):
      value = sig.getValue(can_data.dat)
      self.setItem(i, 0, QTableWidgetItem(name))
      self.setItem(i, 1, QTableWidgetItem(f"{value:.2f}" if value else "N/A"))
```

### Step 10: `mainwindow.py` - Add Detail Panel

```python
class MainWindow(QMainWindow):
  def __init__(self, stream: AbstractStream, dbc_file: str = ""):
    ...
    # Detail panel (center)
    self.signal_view = SignalView()
    self.setCentralWidget(self.signal_view)

    # Connect message selection
    self.messages_widget.msgSelectionChanged.connect(self._onMsgSelected)

    # Load DBC if provided
    if dbc_file:
      dbc().load(dbc_file)

  def _onMsgSelected(self, msg_id: MessageId | None):
    if msg_id and msg_id in self.stream.last_msgs:
      self.signal_view.setMessage(msg_id, self.stream.last_msgs[msg_id])
```

---

## Future Versions (v0.2+)

| Version | Features |
|---------|----------|
| v0.2 | BinaryView (byte/bit visualization with colors) |
| v0.3 | Live replay with seeking (timeline slider) |
| v0.4 | VideoWidget + CameraView (video playback) |
| v0.5 | ChartsWidget (signal plotting) |
| v0.6 | HistoryLog (message history table) |
| v0.7 | Settings, undo/redo, DBC editing |
| v1.0 | Feature parity with C++ cabana |

---

## File-by-File C++ to Python Translation Reference

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

---

## Existing Python Tools to Use

| Tool | Location | Usage |
|------|----------|-------|
| LogReader | `tools/lib/logreader.py` | Read rlog/qlog files |
| FrameReader | `tools/lib/framereader.py` | Read video frames |
| Route | `tools/lib/route.py` | Route loading and segment management |
| DBC | `opendbc/can/dbc.py` | Parse DBC files |
| CANParser | `opendbc/can/parser.py` | Decode CAN signals |
| CommaApi | `tools/lib/api.py` | Fetch routes from comma API |

---

## Dependencies

```
PySide6
numpy
```

Plus existing openpilot packages in PYTHONPATH.

---

## Running

```bash
# After implementation:
python -m openpilot.tools.cabana.pycabana --demo
python -m openpilot.tools.cabana.pycabana "a]2a0ccea32023010|2023-07-27--13-01-19"
python -m openpilot.tools.cabana.pycabana "a]2a0ccea32023010|2023-07-27--13-01-19" --dbc toyota_rav4_2017
```
