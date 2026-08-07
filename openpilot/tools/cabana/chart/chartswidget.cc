#include "tools/cabana/chart/chartswidget.h"
#include "tools/cabana/dbc/dbcqt.h"

#include <algorithm>
#include <future>

#include <QApplication>
#include <QMenu>
#include <QMouseEvent>
#include <QScrollBar>
#include <QToolBar>

#include "tools/cabana/chart/chart.h"

const int MAX_COLUMN_COUNT = 4;
const int CHART_SPACING = 4;

ChartsWidget::ChartsWidget(QWidget *parent) : QFrame(parent) {
  align_timer = new QTimer(this);
  auto_scroll_timer = new QTimer(this);
  setFrameStyle(QFrame::StyledPanel | QFrame::Plain);
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);

  // toolbar
  toolbar = new QToolBar(tr("Charts"), this);
  int icon_size = style()->pixelMetric(QStyle::PM_SmallIconSize);
  toolbar->setIconSize({icon_size, icon_size});

  auto new_plot_btn = new ToolButton("file-plus", tr("New Chart"));
  auto new_tab_btn = new ToolButton("window-stack", tr("New Tab"));
  toolbar->addWidget(new_plot_btn);
  toolbar->addWidget(new_tab_btn);
  toolbar->addWidget(title_label = new QLabel());
  title_label->setContentsMargins(0, 0, style()->pixelMetric(QStyle::PM_LayoutHorizontalSpacing), 0);

  auto chart_type_action = toolbar->addAction("");
  QMenu *chart_type_menu = new QMenu(this);
  auto types = std::array{tr("Line"), tr("Step"), tr("Scatter")};
  for (int i = 0; i < types.size(); ++i) {
    QString type_text = types[i];
    chart_type_menu->addAction(type_text, this, [=]() {
      settings.chart_series_type = i;
      chart_type_action->setText("Type: " + type_text);
      settingChanged();
    });
  }
  chart_type_action->setText("Type: " + types[settings.chart_series_type]);
  chart_type_action->setMenu(chart_type_menu);
  qobject_cast<QToolButton *>(toolbar->widgetForAction(chart_type_action))->setPopupMode(QToolButton::InstantPopup);

  QMenu *menu = new QMenu(this);
  for (int i = 0; i < MAX_COLUMN_COUNT; ++i) {
    menu->addAction(tr("%1").arg(i + 1), [=]() { setColumnCount(i + 1); });
  }
  columns_action = toolbar->addAction("");
  columns_action->setMenu(menu);
  qobject_cast<QToolButton*>(toolbar->widgetForAction(columns_action))->setPopupMode(QToolButton::InstantPopup);

  QWidget *spacer = new QWidget(this);
  spacer->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);
  toolbar->addWidget(spacer);

  range_lb_action = toolbar->addWidget(range_lb = new QLabel(this));
  range_slider = new LogSlider(1000, Qt::Horizontal, this);
  range_slider->setFixedWidth(150 * qApp->devicePixelRatio());
  range_slider->setToolTip(tr("Set the chart range"));
  range_slider->setRange(1, settings.max_cached_minutes * 60);
  range_slider->setSingleStep(1);
  range_slider->setPageStep(60);  // 1 min
  range_slider_action = toolbar->addWidget(range_slider);

  // zoom controls
  undo_zoom_action = toolbar->addAction(utils::icon("arrow-counterclockwise"), tr("Undo Zoom"), [this]() { zoom_undo_stack.undo(); });
  redo_zoom_action = toolbar->addAction(utils::icon("arrow-clockwise"), tr("Redo Zoom"), [this]() { zoom_undo_stack.redo(); });
  undo_zoom_action->setEnabled(false);
  redo_zoom_action->setEnabled(false);
  zoom_undo_stack.setCallbacks({.index_changed = [this]() {
    undo_zoom_action->setEnabled(zoom_undo_stack.canUndo());
    redo_zoom_action->setEnabled(zoom_undo_stack.canRedo());
  }});
  reset_zoom_action = toolbar->addWidget(reset_zoom_btn = new ToolButton("zoom-out", tr("Reset Zoom")));
  reset_zoom_btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

  toolbar->addWidget(remove_all_btn = new ToolButton("x-square", tr("Remove all charts")));
  toolbar->addWidget(dock_btn = new ToolButton(""));
  main_layout->addWidget(toolbar);

  // tabbar
  tabbar = new TabBar(this);
  tabbar->setAutoHide(true);
  tabbar->setExpanding(false);
  tabbar->setDrawBase(true);
  tabbar->setUsesScrollButtons(true);
  main_layout->addWidget(tabbar);

  // charts
  charts_container = new ChartsContainer(this);
  charts_container->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  charts_scroll = new QScrollArea(this);
  charts_scroll->viewport()->setBackgroundRole(QPalette::Base);
  charts_scroll->setFrameStyle(QFrame::NoFrame);
  charts_scroll->setWidgetResizable(true);
  charts_scroll->setWidget(charts_container);
  charts_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(charts_scroll);

  // chart drag preview
  drag_preview = new QLabel(this);
  drag_preview->setAttribute(Qt::WA_TransparentForMouseEvents);
  drag_preview->hide();

  // init settings
  current_theme = settings.theme;
  column_count = std::clamp(settings.chart_column_count, 1, MAX_COLUMN_COUNT);
  max_chart_range = std::clamp(settings.chart_range, 1, settings.max_cached_minutes * 60);
  display_range = std::make_pair(can->minSeconds(), can->minSeconds() + max_chart_range);
  range_slider->setValue(max_chart_range);
  updateToolBar();

  align_timer->setSingleShot(true);
  QObject::connect(align_timer, &QTimer::timeout, this, &ChartsWidget::alignCharts);
  QObject::connect(auto_scroll_timer, &QTimer::timeout, this, &ChartsWidget::doAutoScroll);
  QObject::connect(dbcNotifier(), &QtDBCNotifier::DBCFileChanged, this, &ChartsWidget::removeAll);
  QObject::connect(can, &AbstractStream::eventsMerged, this, &ChartsWidget::eventsMerged);
  QObject::connect(can, &AbstractStream::msgsReceived, this, &ChartsWidget::updateState);
  QObject::connect(can, &AbstractStream::seeking, this, &ChartsWidget::updateState);
  QObject::connect(can, &AbstractStream::timeRangeChanged, this, &ChartsWidget::timeRangeChanged);
  QObject::connect(range_slider, &QSlider::valueChanged, this, &ChartsWidget::setMaxChartRange);
  QObject::connect(new_plot_btn, &QToolButton::clicked, this, &ChartsWidget::newChart);
  QObject::connect(remove_all_btn, &QToolButton::clicked, this, &ChartsWidget::removeAll);
  QObject::connect(reset_zoom_btn, &QToolButton::clicked, this, &ChartsWidget::zoomReset);
  QObject::connect(&settings, &Settings::changed, this, &ChartsWidget::settingChanged);
  QObject::connect(new_tab_btn, &QToolButton::clicked, this, &ChartsWidget::newTab);
  QObject::connect(this, &ChartsWidget::seriesChanged, this, &ChartsWidget::updateTabBar);
  QObject::connect(tabbar, &QTabBar::tabCloseRequested, this, &ChartsWidget::removeTab);
  QObject::connect(tabbar, &QTabBar::currentChanged, [this](int index) {
    if (index != -1) updateLayout(true);
  });
  QObject::connect(dock_btn, &QToolButton::clicked, this, &ChartsWidget::toggleChartsDocking);

  setIsDocked(true);
  newTab();
  qApp->installEventFilter(this);
  setWhatsThis(tr(R"(
    <b>Chart View</b><br />
    <b>Click</b>: Click to seek to a corresponding time.<br />
    <b>Drag</b>: Zoom into the chart.<br />
    <b>Shift + Drag</b>: Scrub through the chart to view values.<br />
    <b>Right Mouse</b>: Open the context menu.<br />
  )"));
}

void ChartsWidget::newTab() {
  static int tab_unique_id = 0;
  int idx = tabbar->addTab("");
  tabbar->setTabData(idx, tab_unique_id++);
  tabbar->setCurrentIndex(idx);
  updateTabBar();
}

void ChartsWidget::removeTab(int index) {
  int id = tabbar->tabData(index).toInt();
  for (auto &c : tab_charts[id]) {
    removeChart(c);
  }
  tab_charts.erase(id);
  tabbar->removeTab(index);
  updateTabBar();
}

void ChartsWidget::updateTabBar() {
  for (int i = 0; i < tabbar->count(); ++i) {
    const auto &charts_in_tab = tab_charts[tabbar->tabData(i).toInt()];
    tabbar->setTabText(i, QString("Tab %1 (%2)").arg(i + 1).arg((int)charts_in_tab.size()));
  }
}

void ChartsWidget::eventsMerged(const MessageEventsMap &new_events) {
  std::vector<std::future<void>> futures;
  for (auto c : charts) {
    futures.push_back(std::async(std::launch::async, &ChartView::updateSeries, c, nullptr, &new_events));
  }
  for (auto &f : futures) f.get();
}

void ChartsWidget::timeRangeChanged(const std::optional<std::pair<double, double>> &time_range) {
  updateToolBar();
  updateState();
}

void ChartsWidget::zoomReset() {
  can->setTimeRange(std::nullopt);
  zoom_undo_stack.clear();
}

QRect ChartsWidget::chartVisibleRect(ChartView *chart) {
  const QRect visible_rect(-charts_container->pos(), charts_scroll->viewport()->size());
  return chart->rect().intersected(QRect(chart->mapFrom(charts_container, visible_rect.topLeft()), visible_rect.size()));
}

void ChartsWidget::showValueTip(double sec) {
  emit showTip(sec);
  if (sec < 0 && !value_tip_visible_) return;

  value_tip_visible_ = sec >= 0;
  for (auto c : currentCharts()) {
    value_tip_visible_ ? c->showTip(sec) : c->hideTip();
  }
}

void ChartsWidget::updateState() {
  if (charts.empty()) return;

  const auto &time_range = can->timeRange();
  const double cur_sec = can->currentSec();
  if (!time_range.has_value()) {
    double pos = (cur_sec - display_range.first) / std::max<float>(1.0, max_chart_range);
    if (pos < 0 || pos > 0.8) {
      display_range.first = std::max(can->minSeconds(), cur_sec - max_chart_range * 0.1);
    }
    double max_sec = std::min(display_range.first + max_chart_range, can->maxSeconds());
    display_range.first = std::max(can->minSeconds(), max_sec - max_chart_range);
    display_range.second = display_range.first + max_chart_range;
  }

  const auto &range = time_range ? *time_range : display_range;
  for (auto c : charts) {
    c->updatePlot(cur_sec, range.first, range.second);
  }
}

void ChartsWidget::setMaxChartRange(int value) {
  max_chart_range = settings.chart_range = range_slider->value();
  updateToolBar();
  updateState();
}

void ChartsWidget::setIsDocked(bool docked) {
  is_docked = docked;
  dock_btn->setIcon(is_docked ? "arrow-up-right-square" : "arrow-down-left-square");
  dock_btn->setToolTip(is_docked ? tr("Float the charts window") : tr("Dock the charts window"));
}

void ChartsWidget::updateToolBar() {
  title_label->setText(tr("Charts: %1").arg(charts.size()));
  columns_action->setText(tr("Columns: %1").arg(column_count));
  range_lb->setText(utils::formatSeconds(max_chart_range));

  bool is_zoomed = can->timeRange().has_value();
  range_lb_action->setVisible(!is_zoomed);
  range_slider_action->setVisible(!is_zoomed);
  undo_zoom_action->setVisible(is_zoomed);
  redo_zoom_action->setVisible(is_zoomed);
  reset_zoom_action->setVisible(is_zoomed);
  reset_zoom_btn->setText(is_zoomed ? tr("%1-%2").arg(can->timeRange()->first, 0, 'f', 2).arg(can->timeRange()->second, 0, 'f', 2) : "");
  remove_all_btn->setEnabled(!charts.empty());
}

void ChartsWidget::settingChanged() {
  if (std::exchange(current_theme, settings.theme) != current_theme) {
    undo_zoom_action->setIcon(utils::icon("arrow-counterclockwise"));
    redo_zoom_action->setIcon(utils::icon("arrow-clockwise"));
  }
  if (range_slider->maximum() != settings.max_cached_minutes * 60) {
    range_slider->setRange(1, settings.max_cached_minutes * 60);
  }
  for (auto c : charts) {
    c->setFixedHeight(settings.chart_height);
    c->setSeriesType((SeriesType)settings.chart_series_type);
    c->resetChartCache();
  }
}

ChartView *ChartsWidget::findChart(const MessageId &id, const cabana::Signal *sig) {
  for (auto c : charts)
    if (c->hasSignal(id, sig)) return c;
  return nullptr;
}

ChartView *ChartsWidget::createChart(int pos) {
  auto chart = new ChartView(can->timeRange().value_or(display_range), this);
  chart->setFixedHeight(settings.chart_height);
  chart->setMinimumWidth(CHART_MIN_WIDTH);
  chart->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
  QObject::connect(chart, &ChartView::axisYLabelWidthChanged, align_timer, qOverload<>(&QTimer::start));
  pos = std::clamp(pos, 0, (int)charts.size());
  charts.insert(charts.begin() + pos, chart);
  currentCharts().insert(currentCharts().begin() + pos, chart);
  updateLayout(true);
  updateToolBar();
  return chart;
}

void ChartsWidget::showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge) {
  ChartView *chart = findChart(id, sig);
  if (show && !chart) {
    chart = merge && currentCharts().size() > 0 ? currentCharts().front() : createChart();
    chart->addSignal(id, sig);
    updateState();
  } else if (!show && chart) {
    chart->removeIf([&](auto &s) { return s.msg_id == id && s.sig == sig; });
  }
}

void ChartsWidget::splitChart(ChartView *src_chart) {
  if (src_chart->sigs.size() > 1) {
    int pos = std::find(charts.begin(), charts.end(), src_chart) - charts.begin() + 1;
    for (auto it = src_chart->sigs.begin() + 1; it != src_chart->sigs.end(); /**/) {
      auto c = createChart(pos);
      // Restore to the original color
      it->color = toQColor(it->sig->color);
      c->sigs.emplace_back(std::move(*it));
      c->updateAxisY();
      c->updateTitle();
      it = src_chart->sigs.erase(it);
    }
    src_chart->updateAxisY();
    src_chart->updateTitle();
    updateState();
    QTimer::singleShot(0, src_chart, &ChartView::resetChartCache);
  }
}

QStringList ChartsWidget::serializeChartIds() const {
  QStringList chart_ids;
  for (auto c : charts) {
    QStringList ids;
    for (const auto& s : c->sigs)
      ids += QString("%1|%2").arg(QString::fromStdString(s.msg_id.toString()), QString::fromStdString(s.sig->name));
    chart_ids += ids.join(',');
  }
  std::reverse(chart_ids.begin(), chart_ids.end());
  return chart_ids;
}

void ChartsWidget::restoreChartsFromIds(const QStringList& chart_ids) {
  for (const auto& chart_id : chart_ids) {
    int index = 0;
    for (const auto& part : chart_id.split(',')) {
      const auto sig_parts = part.split('|');
      if (sig_parts.size() != 2) continue;
      MessageId msg_id = MessageId::fromString(sig_parts[0].toStdString());
      if (auto* msg = dbc()->msg(msg_id))
        if (auto* sig = msg->sig(sig_parts[1].toStdString()))
          showChart(msg_id, sig, true, index++ > 0);
    }
  }
}

void ChartsWidget::setColumnCount(int n) {
  n = std::clamp(n, 1, MAX_COLUMN_COUNT);
  if (column_count != n) {
    column_count = settings.chart_column_count = n;
    updateToolBar();
    updateLayout();
  }
}

void ChartsWidget::updateLayout(bool force) {
  auto charts_layout = charts_container->charts_layout;
  int n = MAX_COLUMN_COUNT;
  for (; n > 1; --n) {
    if ((n * CHART_MIN_WIDTH + (n - 1) * charts_layout->horizontalSpacing()) < charts_layout->geometry().width()) break;
  }

  bool show_column_cb = n > 1;
  columns_action->setVisible(show_column_cb);

  n = std::min(column_count, n);
  auto &current_charts = currentCharts();
  if ((current_charts.size() != charts_layout->count() || n != current_column_count) || force) {
    current_column_count = n;
    charts_container->setUpdatesEnabled(false);
    for (auto c : charts) {
      c->setVisible(false);
    }
    for (int i = 0; i < current_charts.size(); ++i) {
      charts_layout->addWidget(current_charts[i], i / n, i % n);
      if (current_charts[i]->sigs.empty()) {
        // the chart will be resized after add signal. delay setVisible to reduce flicker.
        QTimer::singleShot(0, current_charts[i], [c = current_charts[i]]() { c->setVisible(true); });
      } else {
        current_charts[i]->setVisible(true);
      }
    }
    charts_container->setUpdatesEnabled(true);
  }
}

void ChartsWidget::startChartDrag(ChartView *chart, const QPoint &global_pos) {
  stopAutoScroll();
  drag = {.source = chart, .press_pos = global_pos};
  QPixmap px = chart->grab().scaledToWidth(CHART_MIN_WIDTH * chart->devicePixelRatio(), Qt::SmoothTransformation);
  drag_preview->setPixmap(px);
  drag_preview->resize(px.size() / px.devicePixelRatio());
}

void ChartsWidget::dragChartMove(const QPoint &global_pos) {
  if (!drag.active) {
    if ((global_pos - drag.press_pos).manhattanLength() < QApplication::startDragDistance()) return;
    drag.active = true;
    drag_preview->show();
    drag_preview->raise();
  }
  drag_preview->move(mapFromGlobal(global_pos) + QPoint(5, 5));

  // hovering a tab switches to it so the chart can be dropped into another tab
  int tab = tabbar->tabAt(tabbar->mapFromGlobal(global_pos));
  if (tab >= 0 && tab != tabbar->currentIndex()) {
    tabbar->setCurrentIndex(tab);
  }

  const QPoint container_pos = charts_container->mapFromGlobal(global_pos);
  ChartView *target = nullptr;
  for (auto c : currentCharts()) {
    if (c != drag.source && c->isVisible() && c->geometry().contains(container_pos)) {
      target = c;
      break;
    }
  }
  if (std::exchange(drop_target, target) != target) {
    for (auto c : charts) c->setDropHighlight(c == target);
  }
  bool in_viewport = charts_scroll->viewport()->rect().contains(charts_scroll->viewport()->mapFromGlobal(global_pos));
  bool on_background = !target && in_viewport && !charts_container->childAt(container_pos);
  charts_container->drawDropIndicator(on_background ? container_pos : QPoint());

  if (in_viewport) {
    startAutoScroll(global_pos);
  }
}

void ChartsWidget::cancelChartDrag() {
  drag = {};
  stopAutoScroll();
  drag_preview->hide();
  charts_container->drawDropIndicator({});
  if (auto target = std::exchange(drop_target, nullptr)) target->setDropHighlight(false);
}

void ChartsWidget::dragChartRelease(const QPoint &global_pos) {
  ChartView *source = drag.source;
  bool active = drag.active;
  ChartView *target = drop_target;
  cancelChartDrag();
  if (!active) return;

  const QPoint container_pos = charts_container->mapFromGlobal(global_pos);
  bool in_viewport = charts_scroll->viewport()->rect().contains(charts_scroll->viewport()->mapFromGlobal(global_pos));
  if (target) {
    // merge source into target
    target->takeSignalsFrom(source);
  } else if (in_viewport && !charts_container->childAt(container_pos)) {
    // reorder within the current tab
    auto w = charts_container->getDropAfter(container_pos);
    if (w != source) {
      for (auto &[_, list] : tab_charts) {
        list.erase(std::remove(list.begin(), list.end(), source), list.end());
      }
      auto &cur = currentCharts();
      int to = w ? std::find(cur.begin(), cur.end(), w) - cur.begin() + 1 : 0;
      cur.insert(cur.begin() + to, source);
      updateLayout(true);
      updateTabBar();
    }
  }
}

void ChartsWidget::startAutoScroll(const QPoint &global_pos) {
  auto_scroll_pos = global_pos;
  auto_scroll_timer->start(50);
}

void ChartsWidget::stopAutoScroll() {
  auto_scroll_timer->stop();
  auto_scroll_count = 0;
}

void ChartsWidget::doAutoScroll() {
  QScrollBar *scroll = charts_scroll->verticalScrollBar();
  if (auto_scroll_count < scroll->pageStep()) {
    ++auto_scroll_count;
  }

  int value = scroll->value();
  QPoint pos = charts_scroll->viewport()->mapFromGlobal(auto_scroll_pos);
  QRect area = charts_scroll->viewport()->rect();

  if (pos.y() - area.top() < settings.chart_height / 2) {
    scroll->setValue(value - auto_scroll_count);
  } else if (area.bottom() - pos.y() < settings.chart_height / 2) {
    scroll->setValue(value + auto_scroll_count);
  }
  if (value == scroll->value()) {
    stopAutoScroll();
  } else if (chartDragActive()) {
    // refresh the drop indicator/target at the new scroll position
    dragChartMove(auto_scroll_pos);
  }
}

QSize ChartsWidget::minimumSizeHint() const {
  return QSize(CHART_MIN_WIDTH * 1.5, QWidget::minimumSizeHint().height());
}

void ChartsWidget::newChart() {
  SignalSelector dlg(tr("New Chart"), this);
  if (dlg.exec() == QDialog::Accepted) {
    auto items = dlg.seletedItems();
    if (!items.empty()) {
      auto c = createChart();
      for (auto it : items) {
        c->addSignal(it->msg_id, it->sig);
      }
      updateState();
    }
  }
}

void ChartsWidget::removeChart(ChartView *chart) {
  if (drag.source == chart) cancelChartDrag();
  if (drop_target == chart) drop_target = nullptr;
  charts.erase(std::remove(charts.begin(), charts.end(), chart), charts.end());
  chart->deleteLater();
  for (auto &[_, list] : tab_charts) {
    list.erase(std::remove(list.begin(), list.end(), chart), list.end());
  }
  updateToolBar();
  updateLayout(true);
  alignCharts();
  emit seriesChanged();
}

void ChartsWidget::removeAll() {
  while (tabbar->count() > 1) {
    tabbar->removeTab(1);
  }
  tab_charts.clear();

  if (!charts.empty()) {
    for (auto c : charts) {
      delete c;
    }
    charts.clear();
    emit seriesChanged();
  }
  zoomReset();
}

void ChartsWidget::alignCharts() {
  int plot_left = 0;
  for (auto c : charts) {
    plot_left = std::max(plot_left, c->y_label_width);
  }
  plot_left = std::max((plot_left / 10) * 10 + 10, 50);
  for (auto c : charts) {
    c->updatePlotArea(plot_left);
  }
}

bool ChartsWidget::eventFilter(QObject *o, QEvent *e) {
  // route all mouse events to the chart drag, even when the source chart is hidden by a tab switch
  if (chartDragActive()) {
    if (e->type() == QEvent::MouseMove) {
      dragChartMove(static_cast<QMouseEvent *>(e)->globalPos());
      return true;
    } else if (e->type() == QEvent::MouseButtonRelease && static_cast<QMouseEvent *>(e)->button() == Qt::LeftButton) {
      dragChartRelease(static_cast<QMouseEvent *>(e)->globalPos());
      return false;  // let the release through so Qt clears the implicit mouse grab
    } else if (e->type() == QEvent::MouseButtonPress || e->type() == QEvent::MouseButtonRelease) {
      return true;  // swallow other buttons during the drag
    }
  }

  if (!value_tip_visible_) return false;

  if (e->type() == QEvent::MouseMove) {
    bool on_tip = qobject_cast<TipLabel *>(o) != nullptr;
    auto global_pos = static_cast<QMouseEvent *>(e)->globalPos();

    for (const auto &c : charts) {
      auto local_pos = c->mapFromGlobal(global_pos);
      if (c->plot_area.contains(local_pos)) {
        if (on_tip) {
          showValueTip(c->secondsAtPoint(local_pos));
        }
        return false;
      }
    }

    showValueTip(-1);
  } else if (e->type() == QEvent::Wheel) {
    if (auto tip = qobject_cast<TipLabel *>(o)) {
      // Forward the event to the parent widget
      QCoreApplication::sendEvent(tip->parentWidget(), e);
    }
  }
  return false;
}

bool ChartsWidget::event(QEvent *event) {
  bool back_button = false;
  switch (event->type()) {
    case QEvent::Resize:
      updateLayout();
      break;
    case QEvent::MouseButtonPress:
      back_button = static_cast<QMouseEvent *>(event)->button() == Qt::BackButton;
      break;
    case QEvent::NativeGesture:
      back_button = (static_cast<QNativeGestureEvent *>(event)->value() == 180);
      break;
    case QEvent::WindowDeactivate:
    case QEvent::FocusOut:
      if (chartDragActive()) cancelChartDrag();
      showValueTip(-1);
    default:
      break;
  }

  if (back_button) {
    zoom_undo_stack.undo();
    return true;  // Return true since the event has been handled
  }
  return QFrame::event(event);
}

// ChartsContainer

ChartsContainer::ChartsContainer(ChartsWidget *parent) : charts_widget(parent), QWidget(parent) {
  setBackgroundRole(QPalette::Window);
  QVBoxLayout *charts_main_layout = new QVBoxLayout(this);
  charts_main_layout->setContentsMargins(0, CHART_SPACING, 0, CHART_SPACING);
  charts_layout = new QGridLayout();
  charts_layout->setSpacing(CHART_SPACING);
  charts_main_layout->addLayout(charts_layout);
  charts_main_layout->addStretch(0);
}

void ChartsContainer::paintEvent(QPaintEvent *ev) {
  if (!drop_indictor_pos.isNull() && !childAt(drop_indictor_pos)) {
    QRect r = geometry();
    r.setHeight(CHART_SPACING);
    if (auto insert_after = getDropAfter(drop_indictor_pos)) {
      r.moveTop(insert_after->geometry().bottom());
    }

    QPainter p(this);
    p.fillRect(r, palette().highlight());
  }
}

ChartView *ChartsContainer::getDropAfter(const QPoint &pos) const {
  auto it = std::find_if(charts_widget->currentCharts().crbegin(), charts_widget->currentCharts().crend(), [&pos](auto c) {
    auto area = c->geometry();
    return pos.x() >= area.left() && pos.x() <= area.right() && pos.y() >= area.bottom();
  });
  return it == charts_widget->currentCharts().crend() ? nullptr : *it;
}
