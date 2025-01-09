#include "tools/cabana/chart/chartswidget.h"

#include <algorithm>

#include <QApplication>
#include <QFutureSynchronizer>
#include <QMenu>
#include <QScrollBar>
#include <QToolBar>
#include <QtConcurrent>

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
  QToolBar *toolbar = new QToolBar(tr("Charts"), this);
  int icon_size = style()->pixelMetric(QStyle::PM_SmallIconSize);
  toolbar->setIconSize({icon_size, icon_size});

  auto new_plot_btn = new ToolButton("file-plus", tr("New Chart"));
  auto new_tab_btn = new ToolButton("window-stack", tr("New Tab"));
  toolbar->addWidget(new_plot_btn);
  toolbar->addWidget(new_tab_btn);
  toolbar->addWidget(title_label = new QLabel());
  title_label->setContentsMargins(0, 0, style()->pixelMetric(QStyle::PM_LayoutHorizontalSpacing), 0);

  QMenu *menu = new QMenu(this);
  for (int i = 0; i < MAX_COLUMN_COUNT; ++i) {
    menu->addAction(tr("%1").arg(i + 1), [=]() { setColumnCount(i + 1); });
  }
  columns_action = toolbar->addAction("");
  columns_action->setMenu(menu);
  qobject_cast<QToolButton*>(toolbar->widgetForAction(columns_action))->setPopupMode(QToolButton::InstantPopup);

  QLabel *stretch_label = new QLabel(this);
  stretch_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  toolbar->addWidget(stretch_label);

  range_lb_action = toolbar->addWidget(range_lb = new QLabel(this));
  range_slider = new LogSlider(1000, Qt::Horizontal, this);
  range_slider->setMaximumWidth(200);
  range_slider->setToolTip(tr("Set the chart range"));
  range_slider->setRange(1, settings.max_cached_minutes * 60);
  range_slider->setSingleStep(1);
  range_slider->setPageStep(60);  // 1 min
  range_slider_action = toolbar->addWidget(range_slider);

  // zoom controls
  zoom_undo_stack = new QUndoStack(this);
  toolbar->addAction(undo_zoom_action = zoom_undo_stack->createUndoAction(this));
  undo_zoom_action->setIcon(utils::icon("arrow-counterclockwise"));
  toolbar->addAction(redo_zoom_action = zoom_undo_stack->createRedoAction(this));
  redo_zoom_action->setIcon(utils::icon("arrow-clockwise"));
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
  tabbar->setAcceptDrops(true);
  tabbar->setChangeCurrentOnDrag(true);
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
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &ChartsWidget::removeAll);
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
    <b>Chart view</b><br />
    <!-- TODO: add descprition here -->
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
    tabbar->setTabText(i, QString("Tab %1 (%2)").arg(i + 1).arg(charts_in_tab.count()));
  }
}

void ChartsWidget::eventsMerged(const MessageEventsMap &new_events) {
  QFutureSynchronizer<void> future_synchronizer;
  for (auto c : charts) {
    future_synchronizer.addFuture(QtConcurrent::run(c, &ChartView::updateSeries, nullptr, &new_events));
  }
}

void ChartsWidget::timeRangeChanged(const std::optional<std::pair<double, double>> &time_range) {
  updateToolBar();
  updateState();
}

void ChartsWidget::zoomReset() {
  can->setTimeRange(std::nullopt);
  zoom_undo_stack->clear();
}

QRect ChartsWidget::chartVisibleRect(ChartView *chart) {
  const QRect visible_rect(-charts_container->pos(), charts_scroll->viewport()->size());
  return chart->rect().intersected(QRect(chart->mapFrom(charts_container, visible_rect.topLeft()), visible_rect.size()));
}

void ChartsWidget::showValueTip(double sec) {
  if (sec < 0 && !value_tip_visible_) return;

  value_tip_visible_ = sec >= 0;
  for (auto c : currentCharts()) {
    value_tip_visible_ ? c->showTip(sec) : c->hideTip();
  }
}

void ChartsWidget::updateState() {
  if (charts.isEmpty()) return;

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
  columns_action->setText(tr("Column: %1").arg(column_count));
  range_lb->setText(utils::formatSeconds(max_chart_range));

  bool is_zoomed = can->timeRange().has_value();
  range_lb_action->setVisible(!is_zoomed);
  range_slider_action->setVisible(!is_zoomed);
  undo_zoom_action->setVisible(is_zoomed);
  redo_zoom_action->setVisible(is_zoomed);
  reset_zoom_action->setVisible(is_zoomed);
  reset_zoom_btn->setText(is_zoomed ? tr("%1-%2").arg(can->timeRange()->first, 0, 'f', 2).arg(can->timeRange()->second, 0, 'f', 2) : "");
  remove_all_btn->setEnabled(!charts.isEmpty());
}

void ChartsWidget::settingChanged() {
  if (std::exchange(current_theme, settings.theme) != current_theme) {
    undo_zoom_action->setIcon(utils::icon("arrow-counterclockwise"));
    redo_zoom_action->setIcon(utils::icon("arrow-clockwise"));
    auto theme = settings.theme == DARK_THEME ? QChart::QChart::ChartThemeDark : QChart::ChartThemeLight;
    for (auto c : charts) {
      c->setTheme(theme);
    }
  }
  range_slider->setRange(1, settings.max_cached_minutes * 60);
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
  pos = std::clamp(pos, 0, charts.size());
  charts.insert(pos, chart);
  currentCharts().insert(pos, chart);
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
    int pos = charts.indexOf(src_chart) + 1;
    for (auto it = src_chart->sigs.begin() + 1; it != src_chart->sigs.end(); /**/) {
      auto c = createChart(pos);
      src_chart->chart()->removeSeries(it->series);

      // Restore to the original color
      it->series->setColor(it->sig->color);

      c->addSeries(it->series);
      c->sigs.emplace_back(std::move(*it));
      c->updateAxisY();
      c->updateTitle();
      it = src_chart->sigs.erase(it);
    }
    src_chart->updateAxisY();
    src_chart->updateTitle();
    QTimer::singleShot(0, src_chart, &ChartView::resetChartCache);
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

void ChartsWidget::startAutoScroll() {
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
  QPoint pos = charts_scroll->viewport()->mapFromGlobal(QCursor::pos());
  QRect area = charts_scroll->viewport()->rect();

  if (pos.y() - area.top() < settings.chart_height / 2) {
    scroll->setValue(value - auto_scroll_count);
  } else if (area.bottom() - pos.y() < settings.chart_height / 2) {
    scroll->setValue(value + auto_scroll_count);
  }
  bool vertical_unchanged = value == scroll->value();
  if (vertical_unchanged) {
    stopAutoScroll();
  } else {
    // mouseMoveEvent to updates the drag-selection rectangle
    const QPoint globalPos = charts_scroll->viewport()->mapToGlobal(pos);
    const QPoint windowPos = charts_scroll->window()->mapFromGlobal(globalPos);
    QMouseEvent mm(QEvent::MouseMove, pos, windowPos, globalPos,
                   Qt::NoButton, Qt::LeftButton, Qt::NoModifier, Qt::MouseEventSynthesizedByQt);
    QApplication::sendEvent(charts_scroll->viewport(), &mm);
  }
}

QSize ChartsWidget::minimumSizeHint() const {
  return QSize(CHART_MIN_WIDTH, QWidget::minimumSizeHint().height());
}

void ChartsWidget::newChart() {
  SignalSelector dlg(tr("New Chart"), this);
  if (dlg.exec() == QDialog::Accepted) {
    auto items = dlg.seletedItems();
    if (!items.isEmpty()) {
      auto c = createChart();
      for (auto it : items) {
        c->addSignal(it->msg_id, it->sig);
      }
    }
  }
}

void ChartsWidget::removeChart(ChartView *chart) {
  charts.removeOne(chart);
  chart->deleteLater();
  for (auto &[_, list] : tab_charts) {
    list.removeOne(chart);
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

  if (!charts.isEmpty()) {
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
  if (!value_tip_visible_) return false;

  if (e->type() == QEvent::MouseMove) {
    bool on_tip = qobject_cast<TipLabel *>(o) != nullptr;
    auto global_pos = static_cast<QMouseEvent *>(e)->globalPos();

    for (const auto &c : charts) {
      auto local_pos = c->mapFromGlobal(global_pos);
      if (c->chart()->plotArea().contains(local_pos)) {
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
      showValueTip(-1);
    default:
      break;
  }

  if (back_button) {
    zoom_undo_stack->undo();
    return true;  // Return true since the event has been handled
  }
  return QFrame::event(event);
}

// ChartsContainer

ChartsContainer::ChartsContainer(ChartsWidget *parent) : charts_widget(parent), QWidget(parent) {
  setAcceptDrops(true);
  setBackgroundRole(QPalette::Window);
  QVBoxLayout *charts_main_layout = new QVBoxLayout(this);
  charts_main_layout->setContentsMargins(0, CHART_SPACING, 0, CHART_SPACING);
  charts_layout = new QGridLayout();
  charts_layout->setSpacing(CHART_SPACING);
  charts_main_layout->addLayout(charts_layout);
  charts_main_layout->addStretch(0);
}

void ChartsContainer::dragEnterEvent(QDragEnterEvent *event) {
  if (event->mimeData()->hasFormat(CHART_MIME_TYPE)) {
    event->acceptProposedAction();
    drawDropIndicator(event->pos());
  }
}

void ChartsContainer::dropEvent(QDropEvent *event) {
  if (event->mimeData()->hasFormat(CHART_MIME_TYPE)) {
    auto w = getDropAfter(event->pos());
    auto chart = qobject_cast<ChartView *>(event->source());
    if (w != chart) {
      for (auto &[_, list] : charts_widget->tab_charts) {
        list.removeOne(chart);
      }
      int to = w ? charts_widget->currentCharts().indexOf(w) + 1 : 0;
      charts_widget->currentCharts().insert(to, chart);
      charts_widget->updateLayout(true);
      charts_widget->updateTabBar();
      event->acceptProposedAction();
      chart->startAnimation();
    }
    drawDropIndicator({});
  }
}

void ChartsContainer::paintEvent(QPaintEvent *ev) {
  if (!drop_indictor_pos.isNull() && !childAt(drop_indictor_pos)) {
    QRect r;
    if (auto insert_after = getDropAfter(drop_indictor_pos)) {
      QRect area = insert_after->geometry();
      r = QRect(area.left(), area.bottom() + 1, area.width(), CHART_SPACING);
    } else {
      r = geometry();
      r.setHeight(CHART_SPACING);
    }

    QPainter p(this);
    p.setPen(QPen(palette().highlight(), 2));
    p.drawLine(r.topLeft() + QPoint(1, 0), r.bottomLeft() + QPoint(1, 0));
    p.drawLine(r.topLeft() + QPoint(0, r.height() / 2), r.topRight() + QPoint(0, r.height() / 2));
    p.drawLine(r.topRight(), r.bottomRight());
  }
}

ChartView *ChartsContainer::getDropAfter(const QPoint &pos) const {
  auto it = std::find_if(charts_widget->currentCharts().crbegin(), charts_widget->currentCharts().crend(), [&pos](auto c) {
    auto area = c->geometry();
    return pos.x() >= area.left() && pos.x() <= area.right() && pos.y() >= area.bottom();
  });
  return it == charts_widget->currentCharts().crend() ? nullptr : *it;
}
