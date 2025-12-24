"""ChartsWidget - displays signal values over time with advanced charting features."""

from typing import Optional
from bisect import bisect_left
from dataclasses import dataclass
import math

from PySide6.QtCore import Qt, QRectF, QPointF, QRect, QPoint, QSize, Signal, QTimer
from PySide6.QtGui import (
  QPainter,
  QPen,
  QColor,
  QFont,
  QPainterPath,
  QBrush,
  QPalette,
  QFontMetrics,
  QPixmap,
  QMouseEvent,
)
from PySide6.QtWidgets import (
  QWidget,
  QVBoxLayout,
  QHBoxLayout,
  QScrollArea,
  QFrame,
  QLabel,
  QPushButton,
  QComboBox,
  QDialog,
  QGridLayout,
  QListWidget,
  QListWidgetItem,
  QDialogButtonBox,
  QMenu,
  QToolButton,
  QSizePolicy,
)

from openpilot.tools.cabana.pycabana.dbc.dbc import MessageId
from openpilot.tools.cabana.pycabana.dbc.dbcmanager import dbc_manager


# Constants
CHART_MIN_WIDTH = 300
CHART_HEIGHT = 200
CHART_SPACING = 4
AXIS_X_TOP_MARGIN = 4
MIN_ZOOM_SECONDS = 0.01
EPSILON = 0.000001

# Chart colors
CHART_COLORS = [
  QColor(102, 86, 169),   # purple
  QColor(76, 175, 80),    # green
  QColor(255, 152, 0),    # orange
  QColor(33, 150, 243),   # blue
  QColor(233, 30, 99),    # pink
  QColor(0, 188, 212),    # cyan
  QColor(255, 193, 7),    # amber
  QColor(121, 85, 72),    # brown
]


class SeriesType:
  """Chart series type enumeration."""
  LINE = 0
  STEP_LINE = 1
  SCATTER = 2


@dataclass
class SignalData:
  """Container for signal data and metadata."""
  msg_id: MessageId
  signal_name: str
  color: QColor
  values: list[tuple[float, float]]  # (time, value) pairs
  step_values: list[tuple[float, float]]  # for step line rendering
  min_val: float = 0.0
  max_val: float = 0.0
  track_point: Optional[tuple[float, float]] = None


class TipLabel(QLabel):
  """Tooltip label for displaying signal values at hover point."""

  def __init__(self, parent: Optional[QWidget] = None):
    super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
    self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    self.setForegroundRole(QPalette.ColorRole.ToolTipText)
    self.setBackgroundRole(QPalette.ColorRole.ToolTipBase)

    font = QFont()
    font.setPointSizeF(8.5)
    self.setFont(font)
    self.setMargin(3)
    self.setTextFormat(Qt.TextFormat.RichText)

  def showText(self, pt: QPoint, text: str, w: QWidget, rect: QRect):
    """Display tooltip at given point within rect."""
    self.setText(text)
    if text:
      self.resize(self.sizeHint() + QSize(2, 2))
      tip_pos = QPoint(pt.x() + 8, rect.top() + 2)
      if tip_pos.x() + self.width() >= rect.right():
        tip_pos.setX(pt.x() - self.width() - 8)

      if rect.contains(QRect(tip_pos, self.size())):
        self.move(w.mapToGlobal(tip_pos))
        self.setVisible(True)
        return

    self.setVisible(False)


class Sparkline:
  """Mini-chart for inline display of signal trends."""

  def __init__(self):
    self.pixmap: Optional[QPixmap] = None
    self.min_val: float = 0.0
    self.max_val: float = 0.0
    self.freq: float = 0.0
    self._points: list[tuple[float, float]] = []

  def update(self, values: list[tuple[float, float]], color: QColor, size: QSize, range_sec: int):
    """Update sparkline with new data."""
    if not values or size.isEmpty():
      self.pixmap = None
      return

    self._points = values
    self.min_val = min(v[1] for v in values)
    self.max_val = max(v[1] for v in values)

    if values:
      time_range = values[-1][0] - values[0][0]
      self.freq = len(values) / max(time_range, 1.0)

    self._render(color, size, range_sec)

  def _render(self, color: QColor, size: QSize, range_sec: int):
    """Render sparkline to pixmap."""
    if not self._points:
      return

    # Adjust for flat lines
    is_flat = self.min_val == self.max_val
    min_val = self.min_val - 1.0 if is_flat else self.min_val
    max_val = self.max_val + 1.0 if is_flat else self.max_val

    # Calculate scaling
    xscale = (size.width() - 1) / range_sec
    yscale = (size.height() - 3) / (max_val - min_val) if max_val != min_val else 1.0

    # Transform points
    render_points = []
    for t, v in self._points:
      x = t * xscale
      y = 1.0 + (max_val - v) * yscale
      render_points.append(QPointF(x, y))

    # Render to pixmap
    self.pixmap = QPixmap(size)
    self.pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(self.pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, len(render_points) <= 500)
    painter.setPen(QPen(color, 1))

    if len(render_points) > 1:
      path = QPainterPath()
      path.moveTo(render_points[0])
      for pt in render_points[1:]:
        path.lineTo(pt)
      painter.drawPath(path)

    # Draw points
    painter.setPen(QPen(color, 3))
    if len(render_points) > 0:
      painter.drawPoint(render_points[-1])

    painter.end()

  def isEmpty(self) -> bool:
    """Check if sparkline has no data."""
    return self.pixmap is None


class SignalSelector(QDialog):
  """Dialog for selecting signals to chart."""

  class ListItem(QListWidgetItem):
    """Custom list item storing signal metadata."""

    def __init__(self, msg_id: MessageId, sig_name: str, sig_color: QColor, parent: QListWidget):
      super().__init__(parent)
      self.msg_id = msg_id
      self.sig_name = sig_name
      self.sig_color = sig_color

  def __init__(self, title: str, dbc, parent: Optional[QWidget] = None):
    super().__init__(parent)
    self.setWindowTitle(title)
    self.dbc = dbc
    self._setup_ui()

  def _setup_ui(self):
    """Setup dialog UI."""
    layout = QGridLayout(self)

    # Left column - available signals
    layout.addWidget(QLabel("Available Signals"), 0, 0)

    self.msgs_combo = QComboBox()
    self.msgs_combo.setEditable(True)
    self.msgs_combo.lineEdit().setPlaceholderText("Select a message...")
    self.msgs_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    self.msgs_combo.currentIndexChanged.connect(self._update_available_list)
    layout.addWidget(self.msgs_combo, 1, 0)

    self.available_list = QListWidget()
    self.available_list.itemDoubleClicked.connect(self._add_signal)
    layout.addWidget(self.available_list, 2, 0)

    # Middle column - buttons
    btn_layout = QVBoxLayout()
    btn_layout.addStretch()

    self.add_btn = QPushButton("→")
    self.add_btn.setEnabled(False)
    self.add_btn.clicked.connect(lambda: self._add_signal(self.available_list.currentItem()))
    btn_layout.addWidget(self.add_btn)

    self.remove_btn = QPushButton("←")
    self.remove_btn.setEnabled(False)
    self.remove_btn.clicked.connect(lambda: self._remove_signal(self.selected_list.currentItem()))
    btn_layout.addWidget(self.remove_btn)

    btn_layout.addStretch()
    layout.addLayout(btn_layout, 0, 1, 3, 1)

    # Right column - selected signals
    layout.addWidget(QLabel("Selected Signals"), 0, 2)

    self.selected_list = QListWidget()
    self.selected_list.itemDoubleClicked.connect(self._remove_signal)
    layout.addWidget(self.selected_list, 1, 2, 2, 1)

    # Button box
    button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    button_box.accepted.connect(self.accept)
    button_box.rejected.connect(self.reject)
    layout.addWidget(button_box, 3, 2)

    # Connect signals
    self.available_list.currentRowChanged.connect(lambda row: self.add_btn.setEnabled(row != -1))
    self.selected_list.currentRowChanged.connect(lambda row: self.remove_btn.setEnabled(row != -1))

  def populate(self, messages: list[MessageId]):
    """Populate message combo with available messages."""
    self.msgs_combo.clear()
    for msg_id in sorted(messages, key=lambda m: (m.source, m.address)):
      msg = self.dbc.msg(msg_id)
      if msg and msg.sigs:
        name = f"{msg.name} (0x{msg_id.address:X})"
        self.msgs_combo.addItem(name, msg_id)

  def _update_available_list(self, index: int):
    """Update available signals list based on selected message."""
    self.available_list.clear()
    if index < 0:
      return

    msg_id = self.msgs_combo.itemData(index)
    if not msg_id:
      return

    msg = self.dbc.msg(msg_id)
    if not msg:
      return

    selected = self.selected_items()
    for sig_name, sig_info in msg.sigs.items():
      is_selected = any(item.msg_id == msg_id and item.sig_name == sig_name for item in selected)
      if not is_selected:
        self._add_item_to_list(self.available_list, msg_id, sig_name, sig_info.get('color', QColor(100, 100, 100)), False)

  def _add_signal(self, item: Optional[QListWidgetItem]):
    """Add signal from available to selected."""
    if not item:
      return

    list_item = item
    if isinstance(list_item, self.ListItem):
      self._add_item_to_list(self.selected_list, list_item.msg_id, list_item.sig_name, list_item.sig_color, True)
      row = self.available_list.row(item)
      self.available_list.takeItem(row)

  def _remove_signal(self, item: Optional[QListWidgetItem]):
    """Remove signal from selected."""
    if not item:
      return

    list_item = item
    if isinstance(list_item, self.ListItem):
      if list_item.msg_id == self.msgs_combo.currentData():
        msg = self.dbc.msg(list_item.msg_id)
        if msg:
          self._add_item_to_list(self.available_list, list_item.msg_id, list_item.sig_name, list_item.sig_color, False)

      row = self.selected_list.row(item)
      self.selected_list.takeItem(row)

  def _add_item_to_list(self, parent: QListWidget, msg_id: MessageId, sig_name: str, color: QColor, show_msg_name: bool):
    """Add item to list widget."""
    text = f"<span style='color:{color.name()};'>■ </span> {sig_name}"
    if show_msg_name:
      msg = self.dbc.msg(msg_id)
      msg_name = msg.name if msg else "Unknown"
      text += f" <font color='gray'>{msg_name} 0x{msg_id.address:X}</font>"

    label = QLabel(text)
    label.setContentsMargins(5, 0, 5, 0)
    item = self.ListItem(msg_id, sig_name, color, parent)
    item.setSizeHint(label.sizeHint())
    parent.setItemWidget(item, label)

  def add_selected(self, msg_id: MessageId, sig_name: str, color: QColor):
    """Pre-populate selected list with signal."""
    self._add_item_to_list(self.selected_list, msg_id, sig_name, color, True)

  def selected_items(self) -> list[ListItem]:
    """Get list of selected signals."""
    items = []
    for i in range(self.selected_list.count()):
      item = self.selected_list.item(i)
      if isinstance(item, self.ListItem):
        items.append(item)
    return items


class ChartView(QFrame):
  """Single chart view displaying one or more signals."""

  axisYLabelWidthChanged = Signal(int)
  removeRequested = Signal(object)

  def __init__(self, x_range: tuple[float, float], dbc, parent: Optional[QWidget] = None):
    super().__init__(parent)
    self.dbc = dbc
    self.signals: list[SignalData] = []
    self.axis_x_range = x_range
    self.axis_y_range = (0.0, 1.0)
    self.current_sec = 0.0
    self.series_type = SeriesType.LINE
    self.tooltip_x = -1.0
    self.y_label_width = 0
    self.align_to = 0
    self.is_scrubbing = False
    self.tip_label = TipLabel(self)

    self.setFrameShape(QFrame.Shape.StyledPanel)
    self.setMinimumHeight(CHART_HEIGHT)
    self.setMinimumWidth(CHART_MIN_WIDTH)
    self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
    self.setMouseTracking(True)

    self._setup_ui()

  def _setup_ui(self):
    """Setup chart UI."""
    layout = QVBoxLayout(self)
    layout.setContentsMargins(2, 2, 2, 2)

    # Header with controls
    header = QHBoxLayout()
    header.setContentsMargins(4, 4, 4, 2)

    self.title_label = QLabel()
    self.title_label.setTextFormat(Qt.TextFormat.RichText)
    header.addWidget(self.title_label)
    header.addStretch()

    # Menu button
    self.menu_btn = QToolButton()
    self.menu_btn.setText("☰")
    self.menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QMenu()

    # Series type submenu
    series_menu = menu.addMenu("Series Type")
    self.line_action = series_menu.addAction("Line", lambda: self.set_series_type(SeriesType.LINE))
    self.line_action.setCheckable(True)
    self.line_action.setChecked(True)
    self.step_action = series_menu.addAction("Step Line", lambda: self.set_series_type(SeriesType.STEP_LINE))
    self.step_action.setCheckable(True)
    self.scatter_action = series_menu.addAction("Scatter", lambda: self.set_series_type(SeriesType.SCATTER))
    self.scatter_action.setCheckable(True)

    menu.addSeparator()
    menu.addAction("Manage Signals", self._manage_signals)
    menu.addSeparator()

    self.close_action = menu.addAction("Remove Chart", lambda: self.removeRequested.emit(self))
    self.menu_btn.setMenu(menu)
    header.addWidget(self.menu_btn)

    # Close button
    self.close_btn = QPushButton("×")
    self.close_btn.setFixedSize(20, 20)
    self.close_btn.clicked.connect(lambda: self.removeRequested.emit(self))
    header.addWidget(self.close_btn)

    layout.addLayout(header)

    # Chart canvas
    self.canvas = ChartCanvas(self)
    layout.addWidget(self.canvas, 1)

  def add_signal(self, msg_id: MessageId, sig_name: str, color: QColor):
    """Add a signal to this chart."""
    if self.has_signal(msg_id, sig_name):
      return

    signal_data = SignalData(
      msg_id=msg_id,
      signal_name=sig_name,
      color=color,
      values=[],
      step_values=[],
    )
    self.signals.append(signal_data)
    self._update_title()

  def has_signal(self, msg_id: MessageId, sig_name: str) -> bool:
    """Check if signal is in this chart."""
    return any(s.msg_id == msg_id and s.signal_name == sig_name for s in self.signals)

  def remove_signal(self, msg_id: MessageId, sig_name: str):
    """Remove a signal from this chart."""
    self.signals = [s for s in self.signals if not (s.msg_id == msg_id and s.signal_name == sig_name)]
    if not self.signals:
      self.removeRequested.emit(self)
    else:
      self._update_title()
      self._update_axis_y()

  def update_series(self, events: dict[MessageId, list]):
    """Update signal data from events."""
    for signal in self.signals:
      if signal.msg_id not in events:
        continue

      msg_events = events[signal.msg_id]
      msg = self.dbc.msg(signal.msg_id)
      if not msg or signal.signal_name not in msg.sigs:
        continue

      sig_info = msg.sigs[signal.signal_name]

      # Clear old data
      signal.values = []
      signal.step_values = []

      # Extract signal values from events
      from openpilot.tools.cabana.pycabana.dbc.dbc import decode_signal
      for event in msg_events:
        try:
          value = decode_signal(sig_info, event.dat)
          signal.values.append((event.ts, value))

          # For step line, add horizontal segments
          if signal.step_values:
            signal.step_values.append((event.ts, signal.step_values[-1][1]))
          signal.step_values.append((event.ts, value))
        except Exception:
          pass

    self._update_axis_y()
    self.canvas.update()

  def update_plot(self, cur_sec: float, min_sec: float, max_sec: float):
    """Update plot with new time range."""
    self.current_sec = cur_sec
    if min_sec != self.axis_x_range[0] or max_sec != self.axis_x_range[1]:
      self.axis_x_range = (min_sec, max_sec)
      self._update_axis_y()
    self.canvas.update()

  def show_tip(self, sec: float):
    """Show tooltip at given time."""
    self.tooltip_x = self._map_to_position_x(sec)

    text_list = [f"<b>{sec:.3f}s</b>"]

    for signal in self.signals:
      if not signal.values:
        continue

      # Find value at this time (binary search)
      idx = bisect_left([v[0] for v in signal.values], sec)
      if idx > 0:
        idx -= 1

      if idx < len(signal.values) and signal.values[idx][0] >= self.axis_x_range[0]:
        t, v = signal.values[idx]
        signal.track_point = (t, v)
        value_str = f"{v:.2f}"
      else:
        signal.track_point = None
        value_str = "--"

      min_str = f"{signal.min_val:.2f}" if signal.min_val != float('inf') else "--"
      max_str = f"{signal.max_val:.2f}" if signal.max_val != float('-inf') else "--"

      text_list.append(
        f"<span style='color:{signal.color.name()};'>■ </span>"
        + f"{signal.signal_name}: <b>{value_str}</b> ({min_str}, {max_str})"
      )

    text = "<p style='white-space:pre'>" + "<br/>".join(text_list) + "</p>"
    pt = QPoint(int(self.tooltip_x), self.canvas.geometry().top())
    visible_rect = self.canvas.geometry()
    self.tip_label.showText(pt, text, self, visible_rect)
    self.canvas.update()

  def hide_tip(self):
    """Hide tooltip."""
    self.tooltip_x = -1.0
    for signal in self.signals:
      signal.track_point = None
    self.tip_label.hide()
    self.canvas.update()

  def set_series_type(self, series_type: int):
    """Change series rendering type."""
    self.series_type = series_type
    self.line_action.setChecked(series_type == SeriesType.LINE)
    self.step_action.setChecked(series_type == SeriesType.STEP_LINE)
    self.scatter_action.setChecked(series_type == SeriesType.SCATTER)
    self.canvas.update()

  def update_plot_area(self, left_pos: int, force: bool = False):
    """Update plot area alignment."""
    if self.align_to != left_pos or force:
      self.align_to = left_pos
      self._update_axis_y()
      self.canvas.update()

  def _manage_signals(self):
    """Show signal management dialog."""

    dlg = SignalSelector("Manage Chart", self.dbc, self)

    # Populate with available messages (would need events from parent)
    # For now, just show current signals
    for signal in self.signals:
      dlg.add_selected(signal.msg_id, signal.signal_name, signal.color)

    if dlg.exec() == QDialog.DialogCode.Accepted:
      selected = dlg.selected_items()

      # Add new signals
      for item in selected:
        if not self.has_signal(item.msg_id, item.sig_name):
          self.add_signal(item.msg_id, item.sig_name, item.sig_color)

      # Remove unselected signals
      to_remove = []
      for signal in self.signals:
        if not any(item.msg_id == signal.msg_id and item.sig_name == signal.signal_name for item in selected):
          to_remove.append((signal.msg_id, signal.signal_name))

      for msg_id, sig_name in to_remove:
        self.remove_signal(msg_id, sig_name)

  def _update_title(self):
    """Update chart title with signal names."""
    if not self.signals:
      self.title_label.setText("")
      return

    parts = []
    for signal in self.signals:
      msg = self.dbc.msg(signal.msg_id)
      msg_name = msg.name if msg else "Unknown"
      parts.append(
        f"<span style='color:{signal.color.name()};'><b>{signal.signal_name}</b></span> "
        + f"<font color='gray'>{msg_name}</font>"
      )

    self.title_label.setText(" | ".join(parts))

  def _update_axis_y(self):
    """Update Y-axis range based on visible data."""
    if not self.signals:
      return

    min_val = float('inf')
    max_val = float('-inf')

    for signal in self.signals:
      signal.min_val = float('inf')
      signal.max_val = float('-inf')

      # Find values in current time range
      for t, v in signal.values:
        if self.axis_x_range[0] <= t <= self.axis_x_range[1]:
          signal.min_val = min(signal.min_val, v)
          signal.max_val = max(signal.max_val, v)

      if signal.min_val != float('inf'):
        min_val = min(min_val, signal.min_val)
      if signal.max_val != float('-inf'):
        max_val = max(max_val, signal.max_val)

    if min_val == float('inf'):
      min_val = 0.0
    if max_val == float('-inf'):
      max_val = 0.0

    # Add padding
    delta = abs(max_val - min_val) * 0.05 if abs(max_val - min_val) >= 1e-3 else 1.0
    min_y, max_y, tick_count = self._get_nice_axis_numbers(min_val - delta, max_val + delta, 3)
    self.axis_y_range = (min_y, max_y)

    # Calculate label width
    font_metrics = QFontMetrics(QFont("monospace", 8))
    max_label_width = 0
    for i in range(tick_count):
      value = min_y + (i * (max_y - min_y) / (tick_count - 1))
      label = f"{value:.2f}"
      max_label_width = max(max_label_width, font_metrics.horizontalAdvance(label))

    new_width = max_label_width + 15
    if self.y_label_width != new_width:
      self.y_label_width = new_width
      self.axisYLabelWidthChanged.emit(new_width)

  def _get_nice_axis_numbers(self, min_val: float, max_val: float, tick_count: int) -> tuple[float, float, int]:
    """Calculate nice round numbers for axis labels."""
    def nice_number(x: float, ceiling: bool) -> float:
      exp = math.floor(math.log10(x)) if x > 0 else 0
      z = 10 ** exp
      q = x / z if z != 0 else x

      if ceiling:
        if q <= 1.0:
          q = 1
        elif q <= 2.0:
          q = 2
        elif q <= 5.0:
          q = 5
        else:
          q = 10
      else:
        if q < 1.5:
          q = 1
        elif q < 3.0:
          q = 2
        elif q < 7.0:
          q = 5
        else:
          q = 10

      return q * z

    range_val = nice_number(max_val - min_val, True)
    step = nice_number(range_val / (tick_count - 1), False)
    min_val = math.floor(min_val / step) if step != 0 else min_val
    max_val = math.ceil(max_val / step) if step != 0 else max_val
    tick_count = int(max_val - min_val) + 1 if step != 0 else tick_count

    return (min_val * step, max_val * step, tick_count)

  def _map_to_position_x(self, sec: float) -> float:
    """Map time value to x pixel position."""
    if self.axis_x_range[1] == self.axis_x_range[0]:
      return 0.0

    chart_rect = self.canvas.chart_rect()
    ratio = (sec - self.axis_x_range[0]) / (self.axis_x_range[1] - self.axis_x_range[0])
    return chart_rect.left() + ratio * chart_rect.width()

  def _map_from_position_x(self, x: float) -> float:
    """Map x pixel position to time value."""
    chart_rect = self.canvas.chart_rect()
    if chart_rect.width() == 0:
      return self.axis_x_range[0]

    ratio = (x - chart_rect.left()) / chart_rect.width()
    return self.axis_x_range[0] + ratio * (self.axis_x_range[1] - self.axis_x_range[0])

  def mouseMoveEvent(self, event: QMouseEvent):
    """Handle mouse move for tooltip."""
    chart_rect = self.canvas.chart_rect()
    if chart_rect.contains(event.pos()):
      sec = self._map_from_position_x(event.pos().x())
      self.show_tip(sec)
    elif self.tip_label.isVisible():
      self.hide_tip()

    super().mouseMoveEvent(event)

  def leaveEvent(self, event):
    """Hide tooltip when mouse leaves."""
    self.hide_tip()
    super().leaveEvent(event)


class ChartCanvas(QWidget):
  """Canvas widget for rendering chart contents."""

  def __init__(self, chart_view: ChartView):
    super().__init__()
    self.chart_view = chart_view
    self.setMinimumHeight(100)

  def chart_rect(self) -> QRectF:
    """Get the chart plotting area."""
    margin_left = max(self.chart_view.align_to, 50)
    margin_right = 30
    margin_top = 20
    margin_bottom = 30

    return QRectF(
      margin_left,
      margin_top,
      self.width() - margin_left - margin_right,
      self.height() - margin_top - margin_bottom
    )

  def paintEvent(self, event):
    """Paint the chart."""
    painter = QPainter(self)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    rect = self.rect()
    chart_rect = self.chart_rect()

    # Background
    painter.fillRect(rect, self.palette().color(QPalette.ColorRole.Base))

    # Draw grid
    self._draw_grid(painter, chart_rect)

    # Draw axes
    self._draw_axes(painter, chart_rect)

    # Draw data series
    self._draw_series(painter, chart_rect)

    # Draw timeline cursor
    self._draw_timeline(painter, chart_rect)

    # Draw track points
    self._draw_track_points(painter, chart_rect)

  def _draw_grid(self, painter: QPainter, rect: QRectF):
    """Draw grid lines."""
    painter.setPen(QPen(QColor(60, 60, 60), 1))

    # Horizontal grid lines
    for i in range(5):
      y = rect.top() + rect.height() * i / 4
      painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))

  def _draw_axes(self, painter: QPainter, rect: QRectF):
    """Draw axis labels."""
    painter.setPen(self.palette().color(QPalette.ColorRole.Text))
    font = QFont("monospace", 8)
    painter.setFont(font)
    font_metrics = QFontMetrics(font)

    # Y-axis labels
    min_y, max_y = self.chart_view.axis_y_range
    for i in range(5):
      value = max_y - (max_y - min_y) * i / 4
      y = rect.top() + rect.height() * i / 4
      label = f"{value:.2f}"
      label_width = self.chart_view.y_label_width - 10
      painter.drawText(
        int(rect.left() - label_width - 5),
        int(y - 6),
        label_width,
        12,
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        label
      )

    # X-axis labels
    min_x, max_x = self.chart_view.axis_x_range
    for i in range(5):
      value = min_x + (max_x - min_x) * i / 4
      x = rect.left() + rect.width() * i / 4
      label = f"{value:.1f}"
      label_width = font_metrics.horizontalAdvance(label)
      painter.drawText(
        int(x - label_width / 2),
        int(rect.bottom() + 5),
        label_width,
        15,
        Qt.AlignmentFlag.AlignCenter,
        label
      )

  def _draw_series(self, painter: QPainter, rect: QRectF):
    """Draw signal data series."""
    min_x, max_x = self.chart_view.axis_x_range
    min_y, max_y = self.chart_view.axis_y_range

    if max_x == min_x or max_y == min_y:
      return

    for signal in self.chart_view.signals:
      values = signal.step_values if self.chart_view.series_type == SeriesType.STEP_LINE else signal.values

      if not values:
        continue

      # Filter to visible range
      visible_values = [(t, v) for t, v in values if min_x <= t <= max_x]

      if not visible_values:
        continue

      # Map to screen coordinates
      points = []
      for t, v in visible_values:
        x = rect.left() + (t - min_x) / (max_x - min_x) * rect.width()
        y = rect.bottom() - (v - min_y) / (max_y - min_y) * rect.height()
        points.append(QPointF(x, y))

      # Draw based on series type
      if self.chart_view.series_type == SeriesType.SCATTER:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(signal.color))
        for pt in points:
          painter.drawEllipse(pt, 4, 4)
      else:
        painter.setPen(QPen(signal.color, 2))
        if len(points) > 1:
          path = QPainterPath()
          path.moveTo(points[0])
          for pt in points[1:]:
            path.lineTo(pt)
          painter.drawPath(path)

  def _draw_timeline(self, painter: QPainter, rect: QRectF):
    """Draw current time cursor."""
    min_x, max_x = self.chart_view.axis_x_range
    if max_x == min_x:
      return

    cur_sec = self.chart_view.current_sec
    x = rect.left() + (cur_sec - min_x) / (max_x - min_x) * rect.width()

    if rect.left() <= x <= rect.right():
      painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
      painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))

      # Draw time label
      time_str = f"{cur_sec:.2f}"
      font_metrics = QFontMetrics(painter.font())
      time_str_width = font_metrics.horizontalAdvance(time_str) + 8
      time_rect = QRect(int(x - time_str_width / 2), int(rect.bottom() + 5), time_str_width, 20)

      painter.fillRect(time_rect, QColor(100, 100, 100))
      painter.setPen(QColor(255, 255, 255))
      painter.drawText(time_rect, Qt.AlignmentFlag.AlignCenter, time_str)

  def _draw_track_points(self, painter: QPainter, rect: QRectF):
    """Draw track points at tooltip location."""
    min_x, max_x = self.chart_view.axis_x_range
    min_y, max_y = self.chart_view.axis_y_range

    if max_x == min_x or max_y == min_y:
      return

    painter.setPen(Qt.PenStyle.NoPen)

    track_line_x = -1.0
    for signal in self.chart_view.signals:
      if signal.track_point:
        t, v = signal.track_point
        x = rect.left() + (t - min_x) / (max_x - min_x) * rect.width()
        y = rect.bottom() - (v - min_y) / (max_y - min_y) * rect.height()

        painter.setBrush(QBrush(signal.color.darker(125)))
        painter.drawEllipse(QPointF(x, y), 5, 5)
        track_line_x = max(track_line_x, x)

    if track_line_x > 0:
      painter.setPen(QPen(QColor(100, 100, 100), 1, Qt.PenStyle.DashLine))
      painter.drawLine(int(track_line_x), int(rect.top()), int(track_line_x), int(rect.bottom()))


class ChartsWidget(QFrame):
  """Container for multiple chart views."""

  def __init__(self, dbc=None, parent: Optional[QWidget] = None):
    super().__init__(parent)
    self.dbc = dbc if dbc is not None else dbc_manager()
    self.charts: list[ChartView] = []
    self.display_range = (0.0, 30.0)
    self.max_chart_range = 30
    self.current_sec = 0.0
    self._align_timer = QTimer()
    self._align_timer.setSingleShot(True)
    self._align_timer.timeout.connect(self._align_charts)

    self.setFrameShape(QFrame.Shape.StyledPanel)
    self._setup_ui()

  def _setup_ui(self):
    """Setup main widget UI."""
    layout = QVBoxLayout(self)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(4)

    # Toolbar
    toolbar = QHBoxLayout()

    self.new_chart_btn = QPushButton("New Chart")
    self.new_chart_btn.clicked.connect(self._new_chart)
    toolbar.addWidget(self.new_chart_btn)

    self.title_label = QLabel("Charts: 0")
    toolbar.addWidget(self.title_label)
    toolbar.addStretch()

    self.remove_all_btn = QPushButton("Remove All")
    self.remove_all_btn.clicked.connect(self.remove_all)
    self.remove_all_btn.setEnabled(False)
    toolbar.addWidget(self.remove_all_btn)

    layout.addLayout(toolbar)

    # Scroll area for charts
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.Shape.NoFrame)

    self.charts_container = QWidget()
    self.charts_layout = QVBoxLayout(self.charts_container)
    self.charts_layout.setContentsMargins(0, 0, 0, 0)
    self.charts_layout.setSpacing(CHART_SPACING)
    self.charts_layout.addStretch()

    scroll.setWidget(self.charts_container)
    layout.addWidget(scroll, 1)

  def create_chart(self) -> ChartView:
    """Create a new chart view."""
    chart = ChartView(self.display_range, self.dbc, self)
    chart.removeRequested.connect(self._remove_chart)
    chart.axisYLabelWidthChanged.connect(lambda: self._align_timer.start(10))

    self.charts.append(chart)
    self.charts_layout.insertWidget(len(self.charts) - 1, chart)

    self._update_toolbar()
    return chart

  def show_chart(self, msg_id: MessageId, sig_name: str, color: QColor, merge: bool = False):
    """Show a chart for the given signal."""
    # Check if signal already exists
    for chart in self.charts:
      if chart.has_signal(msg_id, sig_name):
        return

    # Create or reuse chart
    if merge and self.charts:
      chart = self.charts[0]
    else:
      chart = self.create_chart()

    chart.add_signal(msg_id, sig_name, color)

  def _new_chart(self):
    """Show dialog to create new chart."""
    dlg = SignalSelector("New Chart", self.dbc, self)

    # Get available messages from events (would need to be passed from parent)
    # For now, just open empty dialog

    if dlg.exec() == QDialog.DialogCode.Accepted:
      selected = dlg.selected_items()
      if selected:
        chart = self.create_chart()
        for item in selected:
          chart.add_signal(item.msg_id, item.sig_name, item.sig_color)

  def _remove_chart(self, chart: ChartView):
    """Remove a chart."""
    if chart in self.charts:
      self.charts.remove(chart)
      self.charts_layout.removeWidget(chart)
      chart.deleteLater()
      self._update_toolbar()
      self._align_charts()

  def remove_all(self):
    """Remove all charts."""
    for chart in list(self.charts):
      self.charts_layout.removeWidget(chart)
      chart.deleteLater()
    self.charts.clear()
    self._update_toolbar()

  def update_events(self, events: dict[MessageId, list]):
    """Update all charts with new events."""
    for chart in self.charts:
      chart.update_series(events)

  def update_state(self, current_sec: float, min_sec: float, max_sec: float):
    """Update all charts with new time range."""
    self.current_sec = current_sec
    self.display_range = (min_sec, max_sec)

    for chart in self.charts:
      chart.update_plot(current_sec, min_sec, max_sec)

  def _update_toolbar(self):
    """Update toolbar state."""
    self.title_label.setText(f"Charts: {len(self.charts)}")
    self.remove_all_btn.setEnabled(len(self.charts) > 0)

  def _align_charts(self):
    """Align all charts' Y-axis labels."""
    if not self.charts:
      return

    max_width = max(chart.y_label_width for chart in self.charts)
    max_width = max((max_width // 10) * 10 + 10, 50)

    for chart in self.charts:
      chart.update_plot_area(max_width)

  def setEvents(self, events: list):
    """Set CAN events for all charts.

    This converts the flat event list to a per-message dict for update_events.
    """
    if not events:
      return

    # Group events by message ID
    events_by_msg: dict[MessageId, list] = {}
    for event in events:
      msg_id = event.msg_id
      if msg_id not in events_by_msg:
        events_by_msg[msg_id] = []
      events_by_msg[msg_id].append(event)

    self._events = events_by_msg
    self.update_events(events_by_msg)

  def setCurrentTime(self, time_sec: float):
    """Set the current playback time and update chart display range."""
    # Use a window around the current time
    half_range = self.max_chart_range / 2
    min_sec = max(0, time_sec - half_range)
    max_sec = time_sec + half_range

    self.update_state(time_sec, min_sec, max_sec)
