#include "tools/cabana/chartswidget.h"

#include <QActionGroup>
#include <QApplication>
#include <QCompleter>
#include <QDialogButtonBox>
#include <QDrag>
#include <QFutureSynchronizer>
#include <QGraphicsLayout>
#include <QLineEdit>
#include <QMenu>
#include <QOpenGLWidget>
#include <QPushButton>
#include <QRubberBand>
#include <QToolBar>
#include <QToolTip>
#include <QtConcurrent>

const int MAX_COLUMN_COUNT = 4;
static inline bool xLessThan(const QPointF &p, float x) { return p.x() < x; }

// ChartsWidget

ChartsWidget::ChartsWidget(QWidget *parent) : align_timer(this), QFrame(parent) {
  setFrameStyle(QFrame::StyledPanel | QFrame::Plain);
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // toolbar
  QToolBar *toolbar = new QToolBar(tr("Charts"), this);
  int icon_size = style()->pixelMetric(QStyle::PM_SmallIconSize);
  toolbar->setIconSize({icon_size, icon_size});

  QAction *new_plot_btn = toolbar->addAction(utils::icon("file-plus"), tr("New Plot"));
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

  undo_zoom_action = toolbar->addAction(utils::icon("arrow-counterclockwise"), tr("Previous zoom"));
  qobject_cast<QToolButton*>(toolbar->widgetForAction(undo_zoom_action))->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

  reset_zoom_action = toolbar->addAction(utils::icon("zoom-out"), tr("Reset Zoom"));
  qobject_cast<QToolButton*>(toolbar->widgetForAction(reset_zoom_action))->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

  remove_all_btn = toolbar->addAction(utils::icon("x"), tr("Remove all charts"));
  dock_btn = toolbar->addAction("");
  main_layout->addWidget(toolbar);

  // charts
  charts_layout = new QGridLayout();
  charts_layout->setSpacing(10);

  QWidget *charts_container = new QWidget(this);
  QVBoxLayout *charts_main_layout = new QVBoxLayout(charts_container);
  charts_main_layout->setContentsMargins(0, 0, 0, 0);
  charts_main_layout->addLayout(charts_layout);
  charts_main_layout->addStretch(0);

  QScrollArea *charts_scroll = new QScrollArea(this);
  charts_scroll->setFrameStyle(QFrame::NoFrame);
  charts_scroll->setWidgetResizable(true);
  charts_scroll->setWidget(charts_container);
  charts_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(charts_scroll);

  // init settings
  use_dark_theme = QApplication::palette().color(QPalette::WindowText).value() >
                   QApplication::palette().color(QPalette::Background).value();
  column_count = std::clamp(settings.chart_column_count, 1, MAX_COLUMN_COUNT);
  max_chart_range = std::clamp(settings.chart_range, 1, settings.max_cached_minutes * 60);
  display_range = {0, max_chart_range};
  range_slider->setValue(max_chart_range);
  updateToolBar();

  align_timer.setSingleShot(true);
  QObject::connect(&align_timer, &QTimer::timeout, this, &ChartsWidget::alignCharts);
  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &ChartsWidget::removeAll);
  QObject::connect(can, &AbstractStream::eventsMerged, this, &ChartsWidget::eventsMerged);
  QObject::connect(can, &AbstractStream::updated, this, &ChartsWidget::updateState);
  QObject::connect(range_slider, &QSlider::valueChanged, this, &ChartsWidget::setMaxChartRange);
  QObject::connect(new_plot_btn, &QAction::triggered, this, &ChartsWidget::newChart);
  QObject::connect(remove_all_btn, &QAction::triggered, this, &ChartsWidget::removeAll);
  QObject::connect(undo_zoom_action, &QAction::triggered, this, &ChartsWidget::zoomUndo);
  QObject::connect(reset_zoom_action, &QAction::triggered, this, &ChartsWidget::zoomReset);
  QObject::connect(&settings, &Settings::changed, this, &ChartsWidget::settingChanged);
  QObject::connect(dock_btn, &QAction::triggered, [this]() {
    emit dock(!docking);
    docking = !docking;
    updateToolBar();
  });

  setWhatsThis(tr(R"(
    <b>Chart view</b><br />
    <!-- TODO: add descprition here -->
  )"));
}

void ChartsWidget::eventsMerged() {
  {
    QFutureSynchronizer<void> future_synchronizer;
    for (auto c : charts) {
      future_synchronizer.addFuture(QtConcurrent::run(c, &ChartView::updateSeries, nullptr));
    }
  }
  if (can->isPaused()) {
    updateState();
  }
}

void ChartsWidget::setZoom(double min, double max) {
  zoomed_range = {min, max};
  is_zoomed = zoomed_range != display_range;
  updateToolBar();
  updateState();
  emit rangeChanged(min, max, is_zoomed);
}

void ChartsWidget::zoomIn(double min, double max) {
  // Save previous zoom on undo stack
  if (is_zoomed) {
    zoom_stack.push({zoomed_range.first, zoomed_range.second});
  }
  setZoom(min, max);
}

void ChartsWidget::zoomReset() {
  setZoom(display_range.first, display_range.second);
  zoom_stack.clear();
}

void ChartsWidget::zoomUndo() {
  if (!zoom_stack.isEmpty()) {
    auto r = zoom_stack.pop();
    setZoom(r.first, r.second);
  } else {
    zoomReset();
  }
}

void ChartsWidget::updateState() {
  if (charts.isEmpty()) return;

  const double cur_sec = can->currentSec();
  if (!is_zoomed) {
    double pos = (cur_sec - display_range.first) / std::max<float>(1.0, max_chart_range);
    if (pos < 0 || pos > 0.8) {
      display_range.first = std::max(0.0, cur_sec - max_chart_range * 0.1);
    }
    double max_sec = std::min(std::floor(display_range.first + max_chart_range), can->lastEventSecond());
    display_range.first = std::max(0.0, max_sec - max_chart_range);
    display_range.second = display_range.first + max_chart_range;
  } else if (cur_sec < zoomed_range.first || cur_sec >= zoomed_range.second) {
    // loop in zoomed range
    can->seekTo(zoomed_range.first);
  }

  const auto &range = is_zoomed ? zoomed_range : display_range;
  for (auto c : charts) {
    c->updatePlot(cur_sec, range.first, range.second);
  }
}

void ChartsWidget::setMaxChartRange(int value) {
  max_chart_range = settings.chart_range = range_slider->value();
  updateToolBar();
  updateState();
}

void ChartsWidget::updateToolBar() {
  title_label->setText(tr("Charts: %1").arg(charts.size()));
  columns_action->setText(tr("Column: %1").arg(column_count));
  range_lb->setText(QString("Range: %1:%2 ").arg(max_chart_range / 60, 2, 10, QLatin1Char('0')).arg(max_chart_range % 60, 2, 10, QLatin1Char('0')));
  range_lb_action->setVisible(!is_zoomed);
  range_slider_action->setVisible(!is_zoomed);
  undo_zoom_action->setVisible(is_zoomed);
  reset_zoom_action->setVisible(is_zoomed);
  reset_zoom_action->setText(is_zoomed ? tr("Zoomin: %1-%2").arg(zoomed_range.first, 0, 'f', 1).arg(zoomed_range.second, 0, 'f', 1) : "");
  remove_all_btn->setEnabled(!charts.isEmpty());
  dock_btn->setIcon(utils::icon(docking ? "arrow-up-right-square" : "arrow-down-left-square"));
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
}

void ChartsWidget::settingChanged() {
  range_slider->setRange(1, settings.max_cached_minutes * 60);
  for (auto c : charts) {
    c->setFixedHeight(settings.chart_height);
    c->setSeriesType((SeriesType)settings.chart_series_type);
  }
}

ChartView *ChartsWidget::findChart(const MessageId &id, const cabana::Signal *sig) {
  for (auto c : charts)
    if (c->hasSeries(id, sig)) return c;
  return nullptr;
}

ChartView *ChartsWidget::createChart() {
  auto chart = new ChartView(this);
  chart->setFixedHeight(settings.chart_height);
  chart->setMinimumWidth(CHART_MIN_WIDTH);
  chart->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
  chart->chart()->setTheme(use_dark_theme ? QChart::QChart::ChartThemeDark : QChart::ChartThemeLight);
  QObject::connect(chart, &ChartView::remove, [=]() { removeChart(chart); });
  QObject::connect(chart, &ChartView::zoomIn, this, &ChartsWidget::zoomIn);
  QObject::connect(chart, &ChartView::zoomUndo, this, &ChartsWidget::zoomUndo);
  QObject::connect(chart, &ChartView::seriesRemoved, this, &ChartsWidget::seriesChanged);
  QObject::connect(chart, &ChartView::seriesAdded, this, &ChartsWidget::seriesChanged);
  QObject::connect(chart, &ChartView::axisYLabelWidthChanged, &align_timer, qOverload<>(&QTimer::start));
  charts.push_back(chart);
  updateLayout();
  return chart;
}

void ChartsWidget::showChart(const MessageId &id, const cabana::Signal *sig, bool show, bool merge) {
  ChartView *chart = findChart(id, sig);
  if (show && !chart) {
    chart = merge && charts.size() > 0 ? charts.back() : createChart();
    chart->addSeries(id, sig);
    updateState();
  } else if (!show && chart) {
    chart->removeIf([&](auto &s) { return s.msg_id == id && s.sig == sig; });
  }
  updateToolBar();
}

void ChartsWidget::setColumnCount(int n) {
  n = std::clamp(n, 1, MAX_COLUMN_COUNT);
  if (column_count != n) {
    column_count = settings.chart_column_count = n;
    updateToolBar();
    updateLayout();
  }
}

void ChartsWidget::updateLayout() {
  int n = MAX_COLUMN_COUNT;
  for (; n > 1; --n) {
    if ((n * CHART_MIN_WIDTH + (n - 1) * charts_layout->spacing()) < charts_layout->geometry().width()) break;
  }

  bool show_column_cb = n > 1;
  columns_action->setVisible(show_column_cb);

  n = std::min(column_count, n);
  if (charts.size() != charts_layout->count() || n != current_column_count) {
    current_column_count = n;
    charts_layout->parentWidget()->setUpdatesEnabled(false);
    for (int i = 0; i < charts.size(); ++i) {
      charts_layout->addWidget(charts[charts.size() - i - 1], i / n, i % n);
    }
    QTimer::singleShot(0, [this]() { charts_layout->parentWidget()->setUpdatesEnabled(true); });
  }
}

void ChartsWidget::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  updateLayout();
}

void ChartsWidget::newChart() {
  SeriesSelector dlg(tr("New Chart"), this);
  if (dlg.exec() == QDialog::Accepted) {
    auto items = dlg.seletedItems();
    if (!items.isEmpty()) {
      auto c = createChart();
      for (auto it : items) {
        c->addSeries(it->msg_id, it->sig);
      }
    }
  }
}

void ChartsWidget::removeChart(ChartView *chart) {
  charts.removeOne(chart);
  chart->deleteLater();
  updateToolBar();
  alignCharts();
  updateLayout();
  emit seriesChanged();
}

void ChartsWidget::removeAll() {
  for (auto c : charts) {
    c->deleteLater();
  }
  charts.clear();
  updateToolBar();
  emit seriesChanged();
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

bool ChartsWidget::eventFilter(QObject *obj, QEvent *event) {
  if (obj != this && event->type() == QEvent::Close) {
    emit dock_btn->triggered();
    return true;
  }
  return false;
}

bool ChartsWidget::event(QEvent *event) {
  bool back_button = false;
  if (event->type() == QEvent::MouseButtonPress) {
    QMouseEvent *ev = static_cast<QMouseEvent *>(event);
    back_button = ev->button() == Qt::BackButton;
  } else if (event->type() == QEvent::NativeGesture) { // MacOS emulates a back swipe on pressing the mouse back button
    QNativeGestureEvent *ev = static_cast<QNativeGestureEvent *>(event);
    back_button = (ev->value() == 180);
  }

  if (back_button) {
    zoomUndo();
    return true;
  }
  return QFrame::event(event);
}

// ChartView

ChartView::ChartView(QWidget *parent) : QChartView(nullptr, parent) {
  series_type = (SeriesType)settings.chart_series_type;
  QChart *chart = new QChart();
  chart->setBackgroundVisible(false);
  axis_x = new QValueAxis(this);
  axis_y = new QValueAxis(this);
  chart->addAxis(axis_x, Qt::AlignBottom);
  chart->addAxis(axis_y, Qt::AlignLeft);
  chart->legend()->layout()->setContentsMargins(0, 0, 0, 0);
  chart->legend()->setShowToolTips(true);
  chart->setMargins({0, 0, 0, 0});

  background = new QGraphicsRectItem(chart);
  background->setBrush(QApplication::palette().color(QPalette::Base));
  background->setPen(Qt::NoPen);
  background->setZValue(chart->zValue() - 1);

  setChart(chart);

  createToolButtons();
  setRenderHint(QPainter::Antialiasing);
  // TODO: enable zoomIn/seekTo in live streaming mode.
  setRubberBand(can->liveStreaming() ? QChartView::NoRubberBand : QChartView::HorizontalRubberBand);
  setMouseTracking(true);

  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &ChartView::signalRemoved);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &ChartView::signalUpdated);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &ChartView::msgRemoved);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &ChartView::msgUpdated);
}

void ChartView::createToolButtons() {
  move_icon = new QGraphicsPixmapItem(utils::icon("grip-horizontal"), chart());
  move_icon->setToolTip(tr("Drag and drop to combine charts"));

  QToolButton *remove_btn = toolButton("x", tr("Remove Chart"));
  close_btn_proxy = new QGraphicsProxyWidget(chart());
  close_btn_proxy->setWidget(remove_btn);
  close_btn_proxy->setZValue(chart()->zValue() + 11);

  // series types
  QMenu *menu = new QMenu(this);
  auto change_series_group = new QActionGroup(menu);
  change_series_group->setExclusive(true);
  QStringList types{tr("line"), tr("Step Line"), tr("Scatter")};
  for (int i = 0; i < types.size(); ++i) {
    QAction *act = new QAction(types[i], change_series_group);
    act->setData(i);
    act->setCheckable(true);
    act->setChecked(i == (int)series_type);
    menu->addAction(act);
  }
  menu->addSeparator();
  menu->addAction(tr("Manage series"), this, &ChartView::manageSeries);

  QToolButton *manage_btn = toolButton("list", "");
  manage_btn->setMenu(menu);
  manage_btn->setPopupMode(QToolButton::InstantPopup);
  manage_btn_proxy = new QGraphicsProxyWidget(chart());
  manage_btn_proxy->setWidget(manage_btn);
  manage_btn_proxy->setZValue(chart()->zValue() + 11);

  QObject::connect(remove_btn, &QToolButton::clicked, this, &ChartView::remove);
  QObject::connect(change_series_group, &QActionGroup::triggered, [this](QAction *action) {
    setSeriesType((SeriesType)action->data().toInt());
  });
}

void ChartView::addSeries(const MessageId &msg_id, const cabana::Signal *sig) {
  if (hasSeries(msg_id, sig)) return;

  QXYSeries *series = createSeries(series_type, getColor(sig));
  sigs.push_back({.msg_id = msg_id, .sig = sig, .series = series});
  updateTitle();
  updateSeries(sig);
  updateSeriesPoints();
  emit seriesAdded(msg_id, sig);
}

bool ChartView::hasSeries(const MessageId &msg_id, const cabana::Signal *sig) const {
  return std::any_of(sigs.begin(), sigs.end(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

void ChartView::removeIf(std::function<bool(const SigItem &s)> predicate) {
  int prev_size = sigs.size();
  for (auto it = sigs.begin(); it != sigs.end(); /**/) {
    if (predicate(*it)) {
      chart()->removeSeries(it->series);
      it->series->deleteLater();
      auto msg_id = it->msg_id;
      auto sig = it->sig;
      it = sigs.erase(it);
      emit seriesRemoved(msg_id, sig);
    } else {
      ++it;
    }
  }
  if (sigs.empty()) {
    emit remove();
  } else if (sigs.size() != prev_size) {
    updateAxisY();
  }
}

void ChartView::signalUpdated(const cabana::Signal *sig) {
  if (std::any_of(sigs.begin(), sigs.end(), [=](auto &s) { return s.sig == sig; })) {
    updateTitle();
    // TODO: don't update series if only name changed.
    updateSeries(sig);
  }
}

void ChartView::msgUpdated(MessageId id) {
  if (std::any_of(sigs.begin(), sigs.end(), [=](auto &s) { return s.msg_id == id; }))
    updateTitle();
}

void ChartView::manageSeries() {
  SeriesSelector dlg(tr("Mange Chart"), this);
  for (auto &s : sigs) {
    dlg.addSelected(s.msg_id, s.sig);
  }
  if (dlg.exec() == QDialog::Accepted) {
    auto items = dlg.seletedItems();
    for (auto s : items) {
      addSeries(s->msg_id, s->sig);
    }
    removeIf([&](auto &s) {
      return std::none_of(items.cbegin(), items.cend(), [&](auto &it) { return s.msg_id == it->msg_id && s.sig == it->sig; });
    });
  }
}

void ChartView::resizeEvent(QResizeEvent *event) {
  qreal left, top, right, bottom;
  chart()->layout()->getContentsMargins(&left, &top, &right, &bottom);
  move_icon->setPos(left, top);
  close_btn_proxy->setPos(rect().right() - right - close_btn_proxy->size().width(), top);
  int x = close_btn_proxy->pos().x() - manage_btn_proxy->size().width() - style()->pixelMetric(QStyle::PM_LayoutHorizontalSpacing);
  manage_btn_proxy->setPos(x, top);
  chart()->legend()->setGeometry({move_icon->sceneBoundingRect().topRight(), manage_btn_proxy->sceneBoundingRect().bottomLeft()});
  if (align_to > 0) {
    updatePlotArea(align_to);
  }
  QChartView::resizeEvent(event);
}

void ChartView::updatePlotArea(int left_pos) {
  if (align_to != left_pos || rect() != background->rect()) {
    align_to = left_pos;
    background->setRect(rect());

    qreal left, top, right, bottom;
    chart()->layout()->getContentsMargins(&left, &top, &right, &bottom);
    QSizeF x_label_size = QFontMetrics(axis_x->labelsFont()).size(Qt::TextSingleLine, QString::number(axis_x->max(), 'f', 2));
    x_label_size += QSizeF{5 * devicePixelRatioF(), 5 * devicePixelRatioF()};
    int adjust_top = chart()->legend()->geometry().height() + style()->pixelMetric(QStyle::PM_LayoutTopMargin);
    chart()->setPlotArea(rect().adjusted(align_to + left, adjust_top + top, -x_label_size.width() / 2 - right, -x_label_size.height() - bottom));
    chart()->layout()->invalidate();
    if (can->isPaused()) {
      update();
    }
  }
}

void ChartView::updateTitle() {
  for (QLegendMarker *marker : chart()->legend()->markers()) {
    QObject::connect(marker, &QLegendMarker::clicked, this, &ChartView::handleMarkerClicked, Qt::UniqueConnection);
  }
  for (auto &s : sigs) {
    auto decoration = s.series->isVisible() ? "none" : "line-through";
    s.series->setName(QString("<span style=\"text-decoration:%1\"><b>%2</b> <font color=\"gray\">%3 %4</font></span>").arg(decoration, s.sig->name, msgName(s.msg_id), s.msg_id.toString()));
  }
}

void ChartView::updatePlot(double cur, double min, double max) {
  cur_sec = cur;
  if (min != axis_x->min() || max != axis_x->max()) {
    axis_x->setRange(min, max);
    updateAxisY();
    updateSeriesPoints();
  }
  scene()->invalidate({}, QGraphicsScene::ForegroundLayer);
}

void ChartView::updateSeriesPoints() {
  // Show points when zoomed in enough
  for (auto &s : sigs) {
    auto begin = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), xLessThan);
    auto end = std::lower_bound(begin, s.vals.end(), axis_x->max(), xLessThan);

    int num_points = std::max<int>(end - begin, 1);
    int pixels_per_point = width() / num_points;

    if (series_type == SeriesType::Scatter) {
      ((QScatterSeries *)s.series)->setMarkerSize(std::clamp(pixels_per_point / 3, 2, 8) * devicePixelRatioF());
    } else {
      s.series->setPointsVisible(pixels_per_point > 20);
    }
  }
}

void ChartView::updateSeries(const cabana::Signal *sig) {
  for (auto &s : sigs) {
    if (!sig || s.sig == sig) {
      if (!can->liveStreaming()) {
        s.vals.clear();
        s.step_vals.clear();
        s.last_value_mono_time = 0;
      }
      s.series->setColor(getColor(s.sig));

      const auto &msgs = can->events().at(s.msg_id);
      auto first = std::upper_bound(msgs.cbegin(), msgs.cend(), CanEvent{.mono_time = s.last_value_mono_time});
      int new_size = std::max<int>(s.vals.size() + std::distance(first, msgs.cend()), settings.max_cached_minutes * 60 * 100);
      if (s.vals.capacity() <= new_size) {
        s.vals.reserve(new_size * 2);
        s.step_vals.reserve(new_size * 4);
      }

      const double route_start_time = can->routeStartTime();
      for (auto end = msgs.cend(); first != end; ++first) {
        double value = get_raw_value(first->dat, first->size, *s.sig);
        double ts = first->mono_time / 1e9 - route_start_time;  // seconds
        s.vals.append({ts, value});
        if (!s.step_vals.empty()) {
          s.step_vals.append({ts, s.step_vals.back().y()});
        }
        s.step_vals.append({ts, value});
        s.last_value_mono_time = first->mono_time;
      }
      if (!can->liveStreaming()) {
        s.segment_tree.build(s.vals);
      }
      s.series->replace(series_type == SeriesType::StepLine ? s.step_vals : s.vals);
    }
  }
  updateAxisY();
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  if (sigs.isEmpty()) return;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();
  QString unit = sigs[0].sig->unit;

  for (auto &s : sigs) {
    if (!s.series->isVisible()) continue;

    // Only show unit when all signals have the same unit
    if (unit != s.sig->unit) {
      unit.clear();
    }

    auto first = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), xLessThan);
    auto last = std::lower_bound(first, s.vals.end(), axis_x->max(), xLessThan);
    if (can->liveStreaming()) {
      for (auto it = first; it != last; ++it) {
        if (it->y() < min) min = it->y();
        if (it->y() > max) max = it->y();
      }
    } else {
      auto [min_y, max_y] = s.segment_tree.minmax(std::distance(s.vals.begin(), first), std::distance(s.vals.begin(), last));
      min = std::min(min, min_y);
      max = std::max(max, max_y);
    }
  }
  if (min == std::numeric_limits<double>::max()) min = 0;
  if (max == std::numeric_limits<double>::lowest()) max = 0;

  if (axis_y->titleText() != unit) {
    axis_y->setTitleText(unit);
    y_label_width = 0;  // recalc width
  }

  double delta = std::abs(max - min) < 1e-3 ? 1 : (max - min) * 0.05;
  auto [min_y, max_y, tick_count] = getNiceAxisNumbers(min - delta, max + delta, axis_y->tickCount());
  if (min_y != axis_y->min() || max_y != axis_y->max() || y_label_width == 0) {
    axis_y->setRange(min_y, max_y);
    axis_y->setTickCount(tick_count);

    int title_spacing = unit.isEmpty() ? 0 : QFontMetrics(axis_y->titleFont()).size(Qt::TextSingleLine, unit).height();
    QFontMetrics fm(axis_y->labelsFont());
    int n = qMax(int(-qFloor(std::log10((max_y - min_y) / (tick_count - 1)))), 0) + 1;
    y_label_width = title_spacing + qMax(fm.width(QString::number(min_y, 'f', n)), fm.width(QString::number(max_y, 'f', n))) + 15;
    axis_y->setLabelFormat(QString("%.%1f").arg(n));
    emit axisYLabelWidthChanged(y_label_width);
  }
}

std::tuple<double, double, int> ChartView::getNiceAxisNumbers(qreal min, qreal max, int tick_count) {
  qreal range = niceNumber((max - min), true);  // range with ceiling
  qreal step = niceNumber(range / (tick_count - 1), false);
  min = qFloor(min / step);
  max = qCeil(max / step);
  tick_count = int(max - min) + 1;
  return {min * step, max * step, tick_count};
}

// nice numbers can be expressed as form of 1*10^n, 2* 10^n or 5*10^n
qreal ChartView::niceNumber(qreal x, bool ceiling) {
  qreal z = qPow(10, qFloor(std::log10(x))); //find corresponding number of the form of 10^n than is smaller than x
  qreal q = x / z; //q<10 && q>=1;
  if (ceiling) {
    if (q <= 1.0) q = 1;
    else if (q <= 2.0) q = 2;
    else if (q <= 5.0) q = 5;
    else q = 10;
  } else {
    if (q < 1.5) q = 1;
    else if (q < 3.0) q = 2;
    else if (q < 7.0) q = 5;
    else q = 10;
  }
  return q * z;
}

void ChartView::leaveEvent(QEvent *event) {
  clearTrackPoints();
  scene()->update();
  QChartView::leaveEvent(event);
}

void ChartView::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton && move_icon->sceneBoundingRect().contains(event->pos())) {
    QMimeData *mimeData = new QMimeData;
    mimeData->setData(mime_type, QByteArray::number((qulonglong)this));
    QDrag *drag = new QDrag(this);
    drag->setMimeData(mimeData);
    drag->setPixmap(grab());
    drag->setHotSpot(event->pos());
    drag->exec(Qt::CopyAction | Qt::MoveAction, Qt::MoveAction);
  } else if (event->button() == Qt::LeftButton && QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier)) {
    if (!can->liveStreaming()) {
      // Save current playback state when scrubbing
      resume_after_scrub = !can->isPaused();
      if (resume_after_scrub) {
        can->pause(true);
      }
      is_scrubbing = true;
    }
  } else {
    QChartView::mousePressEvent(event);
  }
}

void ChartView::mouseReleaseEvent(QMouseEvent *event) {
  auto rubber = findChild<QRubberBand *>();
  if (event->button() == Qt::LeftButton && rubber && rubber->isVisible()) {
    rubber->hide();
    QRectF rect = rubber->geometry().normalized();
    double min = chart()->mapToValue(rect.topLeft()).x();
    double max = chart()->mapToValue(rect.bottomRight()).x();

    // Prevent zooming/seeking past the end of the route
    min = std::clamp(min, 0., can->totalSeconds());
    max = std::clamp(max, 0., can->totalSeconds());

    double min_rounded = std::floor(min * 10.0) / 10.0;
    double max_rounded = std::floor(max * 10.0) / 10.0;
    if (rubber->width() <= 0) {
      // no rubber dragged, seek to mouse position
      can->seekTo(min);
    } else if ((max_rounded - min_rounded) >= 0.5) {
      // zoom in if selected range is greater than 0.5s
      emit zoomIn(min_rounded, max_rounded);
    } else {
      scene()->invalidate({}, QGraphicsScene::ForegroundLayer);
    }
    event->accept();
  } else if (!can->liveStreaming() && event->button() == Qt::RightButton) {
    emit zoomUndo();
    event->accept();
  } else {
    QGraphicsView::mouseReleaseEvent(event);
  }

  // Resume playback if we were scrubbing
  is_scrubbing = false;
  if (resume_after_scrub) {
    can->pause(false);
    resume_after_scrub = false;
  }
}

void ChartView::mouseMoveEvent(QMouseEvent *ev) {
  const auto plot_area = chart()->plotArea();
  // Scrubbing
  if (is_scrubbing && QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier)) {
    if (plot_area.contains(ev->pos())) {
      can->seekTo(std::clamp(chart()->mapToValue(ev->pos()).x(), 0., can->totalSeconds()));
    }
    return;
  }

  auto rubber = findChild<QRubberBand *>();
  bool is_zooming = rubber && rubber->isVisible();
  is_scrubbing = false;
  clearTrackPoints();

  if (!is_zooming && plot_area.contains(ev->pos())) {
    QStringList text_list;
    const double sec = chart()->mapToValue(ev->pos()).x();
    qreal x = -1;
    for (auto &s : sigs) {
      if (!s.series->isVisible()) continue;

      // use reverse iterator to find last item <= sec.
      double value = 0;
      auto it = std::lower_bound(s.vals.rbegin(), s.vals.rend(), sec, [](auto &p, double x) { return p.x() > x; });
      if (it != s.vals.rend() && it->x() >= axis_x->min()) {
        value = it->y();
        s.track_pt = chart()->mapToPosition(*it);
        x = std::max(x, s.track_pt.x());
      }
      text_list.push_back(QString("<span style=\"color:%1;\">■ </span>%2: <b>%3</b>")
                              .arg(s.series->color().name(), s.sig->name,
                                   s.track_pt.isNull() ? "--" : QString::number(value)));
    }
    if (x < 0) {
      x = ev->pos().x();
    }
    text_list.push_front(QString::number(chart()->mapToValue({x, 0}).x(), 'f', 3));
    QPointF tooltip_pt(x + 12, plot_area.top() - 20);
    QToolTip::showText(mapToGlobal(tooltip_pt.toPoint()), text_list.join("<br />"), this, plot_area.toRect());
    scene()->invalidate({}, QGraphicsScene::ForegroundLayer);
  } else {
    QToolTip::hideText();
  }

  QChartView::mouseMoveEvent(ev);
  if (is_zooming) {
    QRect rubber_rect = rubber->geometry();
    rubber_rect.setLeft(std::max(rubber_rect.left(), (int)plot_area.left()));
    rubber_rect.setRight(std::min(rubber_rect.right(), (int)plot_area.right()));
    if (rubber_rect != rubber->geometry()) {
      rubber->setGeometry(rubber_rect);
    }
    scene()->invalidate({}, QGraphicsScene::ForegroundLayer);
  }
}

void ChartView::dragMoveEvent(QDragMoveEvent *event) {
  if (event->mimeData()->hasFormat(mime_type)) {
    event->setDropAction(event->source() == this ? Qt::MoveAction : Qt::CopyAction);
    event->accept();
  } else {
    event->ignore();
  }
}

void ChartView::dropEvent(QDropEvent *event) {
  if (event->mimeData()->hasFormat(mime_type)) {
    if (event->source() == this) {
      event->setDropAction(Qt::MoveAction);
      event->accept();
    } else {
      ChartView *source_chart = (ChartView *)event->source();
      for (auto &s : source_chart->sigs) {
        addSeries(s.msg_id, s.sig);
      }
      emit source_chart->remove();
      event->acceptProposedAction();
    }
  } else {
    event->ignore();
  }
}

void ChartView::drawForeground(QPainter *painter, const QRectF &rect) {
  // draw time line
  qreal x = chart()->mapToPosition(QPointF{cur_sec, 0}).x();
  x = std::clamp(x, chart()->plotArea().left(), chart()->plotArea().right());
  qreal y1 = chart()->plotArea().top() - 2;
  qreal y2 = chart()->plotArea().bottom() + 2;
  painter->setPen(QPen(chart()->titleBrush().color(), 2));
  painter->drawLine(QPointF{x, y1}, QPointF{x, y2});

  // draw track points
  painter->setPen(Qt::NoPen);
  qreal track_line_x = -1;
  for (auto &s : sigs) {
    if (!s.track_pt.isNull() && s.series->isVisible()) {
      painter->setBrush(s.series->color().darker(125));
      painter->drawEllipse(s.track_pt, 5.5, 5.5);
      track_line_x = std::max(track_line_x, s.track_pt.x());
    }
  }
  if (track_line_x > 0) {
    painter->setPen(QPen(Qt::darkGray, 1, Qt::DashLine));
    painter->drawLine(QPointF{track_line_x, y1}, QPointF{track_line_x, y2});
  }

  // paint points. OpenGL mode lacks certain features (such as showing points)
  painter->setPen(Qt::NoPen);
  for (auto &s : sigs) {
    if (s.series->useOpenGL() && s.series->isVisible() && s.series->pointsVisible()) {
      auto first = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), xLessThan);
      auto last = std::lower_bound(first, s.vals.end(), axis_x->max(), xLessThan);
      for (auto it = first; it != last; ++it) {
        painter->setBrush(s.series->color());
        painter->drawEllipse(chart()->mapToPosition(*it), 4, 4);
      }
    }
  }

  // paint zoom range
  auto rubber = findChild<QRubberBand *>();
  if (rubber && rubber->isVisible() && rubber->width() > 1) {
    painter->setPen(Qt::white);
    auto rubber_rect = rubber->geometry().normalized();
    for (const auto &pt : {rubber_rect.bottomLeft(), rubber_rect.bottomRight()}) {
      QString sec = QString::number(chart()->mapToValue(pt).x(), 'f', 1);
      // ChartAxisElement's padding is 4 (https://codebrowser.dev/qt5/qtcharts/src/charts/axis/chartaxiselement_p.h.html)
      auto r = painter->fontMetrics().boundingRect(sec).adjusted(-6, -4, 6, 4);
      pt == rubber_rect.bottomLeft() ? r.moveTopRight(pt + QPoint{0, 2}) : r.moveTopLeft(pt + QPoint{0, 2});
      painter->fillRect(r, Qt::gray);
      painter->drawText(r, Qt::AlignCenter, sec);
    }
  }
}

QXYSeries *ChartView::createSeries(SeriesType type, QColor color) {
  QXYSeries *series = nullptr;
  if (type == SeriesType::Line) {
    series = new QLineSeries(this);
    chart()->legend()->setMarkerShape(QLegend::MarkerShapeRectangle);
  } else if (type == SeriesType::StepLine) {
    series = new QLineSeries(this);
    chart()->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
  } else {
    series = new QScatterSeries(this);
    chart()->legend()->setMarkerShape(QLegend::MarkerShapeCircle);
  }
  series->setColor(color);
  // TODO: Due to a bug in CameraWidget the camera frames
  // are drawn instead of the graphs on MacOS. Re-enable OpenGL when fixed
#ifndef __APPLE__
  series->setUseOpenGL(true);
  // Qt doesn't properly apply device pixel ratio in OpenGL mode
  QPen pen = series->pen();
  pen.setWidthF(2.0 * devicePixelRatioF());
  series->setPen(pen);
#endif
  chart()->addSeries(series);
  series->attachAxis(axis_x);
  series->attachAxis(axis_y);

  // disables the delivery of mouse events to the opengl widget.
  // this enables the user to select the zoom area when the mouse press on the data point.
  auto glwidget = findChild<QOpenGLWidget *>();
  if (glwidget && !glwidget->testAttribute(Qt::WA_TransparentForMouseEvents)) {
    glwidget->setAttribute(Qt::WA_TransparentForMouseEvents);
  }
  return series;
}

void ChartView::setSeriesType(SeriesType type) {
  if (type != series_type) {
    series_type = type;
    for (auto &s : sigs) {
      chart()->removeSeries(s.series);
      s.series->deleteLater();
    }
    for (auto &s : sigs) {
      auto series = createSeries(series_type, getColor(s.sig));
      series->replace(series_type == SeriesType::StepLine ? s.step_vals : s.vals);
      s.series = series;
    }
    updateSeriesPoints();
    updateTitle();
  }
}

void ChartView::handleMarkerClicked() {
  auto marker = qobject_cast<QLegendMarker *>(sender());
  Q_ASSERT(marker);
  if (sigs.size() > 1) {
    auto series = marker->series();
    series->setVisible(!series->isVisible());
    marker->setVisible(true);
    updateAxisY();
    updateTitle();
  }
}

// SeriesSelector

SeriesSelector::SeriesSelector(QString title, QWidget *parent) : QDialog(parent) {
  setWindowTitle(title);
  QGridLayout *main_layout = new QGridLayout(this);

  // left column
  main_layout->addWidget(new QLabel(tr("Available Signals")), 0, 0);
  main_layout->addWidget(msgs_combo = new QComboBox(this), 1, 0);
  msgs_combo->setEditable(true);
  msgs_combo->lineEdit()->setPlaceholderText(tr("Select a msg..."));
  msgs_combo->setInsertPolicy(QComboBox::NoInsert);
  msgs_combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  msgs_combo->completer()->setFilterMode(Qt::MatchContains);

  main_layout->addWidget(available_list = new QListWidget(this), 2, 0);

  // buttons
  QVBoxLayout *btn_layout = new QVBoxLayout();
  QPushButton *add_btn = new QPushButton(utils::icon("chevron-right"), "", this);
  add_btn->setEnabled(false);
  QPushButton *remove_btn = new QPushButton(utils::icon("chevron-left"), "", this);
  remove_btn->setEnabled(false);
  btn_layout->addStretch(0);
  btn_layout->addWidget(add_btn);
  btn_layout->addWidget(remove_btn);
  btn_layout->addStretch(0);
  main_layout->addLayout(btn_layout, 0, 1, 3, 1);

  // right column
  main_layout->addWidget(new QLabel(tr("Selected Signals")), 0, 2);
  main_layout->addWidget(selected_list = new QListWidget(this), 1, 2, 2, 1);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox, 3, 2);

  for (auto it = can->last_msgs.cbegin(); it != can->last_msgs.cend(); ++it) {
    if (auto m = dbc()->msg(it.key())) {
      msgs_combo->addItem(QString("%1 (%2)").arg(m->name).arg(it.key().toString()), QVariant::fromValue(it.key()));
    }
  }
  msgs_combo->model()->sort(0);
  msgs_combo->setCurrentIndex(-1);

  QObject::connect(msgs_combo, qOverload<int>(&QComboBox::currentIndexChanged), this, &SeriesSelector::updateAvailableList);
  QObject::connect(available_list, &QListWidget::currentRowChanged, [=](int row) { add_btn->setEnabled(row != -1); });
  QObject::connect(selected_list, &QListWidget::currentRowChanged, [=](int row) { remove_btn->setEnabled(row != -1); });
  QObject::connect(available_list, &QListWidget::itemDoubleClicked, this, &SeriesSelector::add);
  QObject::connect(selected_list, &QListWidget::itemDoubleClicked, this, &SeriesSelector::remove);
  QObject::connect(add_btn, &QPushButton::clicked, [this]() { if (auto item = available_list->currentItem()) add(item); });
  QObject::connect(remove_btn, &QPushButton::clicked, [this]() { if (auto item = selected_list->currentItem()) remove(item); });
  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void SeriesSelector::add(QListWidgetItem *item) {
  auto it = (ListItem *)item;
  addItemToList(selected_list, it->msg_id, it->sig, true);
  delete item;
}

void SeriesSelector::remove(QListWidgetItem *item) {
  auto it = (ListItem *)item;
  if (it->msg_id == msgs_combo->currentData().value<MessageId>()) {
    addItemToList(available_list, it->msg_id, it->sig);
  }
  delete item;
}

void SeriesSelector::updateAvailableList(int index) {
  if (index == -1) return;
  available_list->clear();
  MessageId msg_id = msgs_combo->itemData(index).value<MessageId>();
  auto selected_items = seletedItems();
  for (auto s : dbc()->msg(msg_id)->getSignals()) {
    bool is_selected = std::any_of(selected_items.begin(), selected_items.end(), [=, sig=s](auto it) { return it->msg_id == msg_id && it->sig == sig; });
    if (!is_selected) {
      addItemToList(available_list, msg_id, s);
    }
  }
}

void SeriesSelector::addItemToList(QListWidget *parent, const MessageId id, const cabana::Signal *sig, bool show_msg_name) {
  QString text = QString("<span style=\"color:%0;\">■ </span> %1").arg(getColor(sig).name(), sig->name);
  if (show_msg_name) text += QString(" <font color=\"gray\">%0 %1</font>").arg(msgName(id), id.toString());

  QLabel *label = new QLabel(text);
  label->setContentsMargins(5, 0, 5, 0);
  auto new_item = new ListItem(id, sig, parent);
  new_item->setSizeHint(label->sizeHint());
  parent->setItemWidget(new_item, label);
}

QList<SeriesSelector::ListItem *> SeriesSelector::seletedItems() {
  QList<SeriesSelector::ListItem *> ret;
  for (int i = 0; i < selected_list->count(); ++i) ret.push_back((ListItem *)selected_list->item(i));
  return ret;
}
