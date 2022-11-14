#include "tools/cabana/chartswidget.h"

#include <QFutureSynchronizer>
#include <QGraphicsLayout>
#include <QRubberBand>
#include <QTimer>
#include <QToolButton>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtConcurrent>
#include <QToolBar>

// ChartsWidget

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // toolbar
  QToolBar *toolbar = new QToolBar(tr("Charts"), this);
  title_label = new QLabel();
  title_label->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Preferred);
  toolbar->addWidget(title_label);
  toolbar->addWidget(range_label = new QLabel());
  reset_zoom_btn = toolbar->addAction("⟲");
  reset_zoom_btn->setToolTip(tr("Reset zoom (drag on chart to zoom X-Axis)"));
  remove_all_btn = toolbar->addAction("✖");
  remove_all_btn->setToolTip(tr("Remove all charts"));
  dock_btn = toolbar->addAction("");
  main_layout->addWidget(toolbar);
  updateToolBar();

  // charts
  QWidget *charts_container = new QWidget(this);
  charts_layout = new QVBoxLayout(charts_container);
  charts_layout->addStretch();

  QScrollArea *charts_scroll = new QScrollArea(this);
  charts_scroll->setWidgetResizable(true);
  charts_scroll->setWidget(charts_container);
  charts_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  main_layout->addWidget(charts_scroll);

  QObject::connect(dbc(), &DBCManager::DBCFileChanged, [this]() { removeAll(nullptr); });
  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &ChartsWidget::removeAll);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &ChartsWidget::signalUpdated);
  QObject::connect(dbc(), &DBCManager::msgRemoved, [this](uint32_t address) {
    for (auto c : charts.toVector())
      if (DBCManager::parseId(c->id).second == address) removeChart(c);
  });
  QObject::connect(dbc(), &DBCManager::msgUpdated, [this](uint32_t address) {
    for (auto c : charts) {
      if (DBCManager::parseId(c->id).second == address) c->updateTitle();
    }
  });
  QObject::connect(can, &CANMessages::eventsMerged, this, &ChartsWidget::eventsMerged);
  QObject::connect(can, &CANMessages::updated, this, &ChartsWidget::updateState);
  QObject::connect(remove_all_btn, &QAction::triggered, [this]() { removeAll(); });
  QObject::connect(reset_zoom_btn, &QAction::triggered, this, &ChartsWidget::zoomReset);
  QObject::connect(dock_btn, &QAction::triggered, [this]() {
    emit dock(!docking);
    docking = !docking;
    updateToolBar();
  });
}

void ChartsWidget::eventsMerged() {
  if (auto events = can->events(); events && !events->empty()) {
    auto it = std::find_if(events->begin(), events->end(), [=](const Event *e) { return e->which == cereal::Event::Which::CAN; });
    event_range.first = it == events->end() ? 0 : (*it)->mono_time / (double)1e9 - can->routeStartTime();
    event_range.second = it == events->end() ? 0 : events->back()->mono_time / (double)1e9 - can->routeStartTime();
    if (display_range.first == 0 && event_range.second == 0) {
      display_range.first = event_range.first;
      display_range.second = std::min(event_range.first + settings.max_chart_x_range, event_range.second);
    }
  }
}

void ChartsWidget::zoomIn(double min, double max) {
  zoomed_range = {min, max};
  is_zoomed = zoomed_range != display_range;
  updateToolBar();
  emit rangeChanged(min, max, is_zoomed);
  updateState();
}

void ChartsWidget::zoomReset() {
  zoomIn(display_range.first, display_range.second);
}

void ChartsWidget::updateState() {
  if (charts.isEmpty()) return;

  const double current_sec = can->currentSec();
  if (is_zoomed) {
    if (current_sec < zoomed_range.first || current_sec >= zoomed_range.second) {
      can->seekTo(zoomed_range.first);
    }
  } else {
    auto prev_range = display_range;
    if (current_sec < display_range.first || current_sec >= (display_range.second - 5)) {
      // line marker reached the end, or seeked to a timestamp out of range.
      display_range.first = current_sec - 5;
    }
    display_range.first = std::max(display_range.first, event_range.first);
    display_range.second = std::min(display_range.first + settings.max_chart_x_range, event_range.second);
    if (prev_range != display_range) {
      QFutureSynchronizer<void> future_synchronizer;
      for (auto c : charts)
        future_synchronizer.addFuture(QtConcurrent::run(c, &ChartView::updateSeries, display_range));
    }
  }

  const auto &range = is_zoomed ? zoomed_range : display_range;
  for (auto c : charts) {
    c->setRange(range.first, range.second);
    c->updateLineMarker(current_sec);
  }
}

void ChartsWidget::updateToolBar() {
  remove_all_btn->setEnabled(!charts.isEmpty());
  reset_zoom_btn->setEnabled(is_zoomed);
  range_label->setText(is_zoomed ? tr("%1 - %2").arg(zoomed_range.first, 0, 'f', 2).arg(zoomed_range.second, 0, 'f', 2) : "");
  title_label->setText(charts.size() > 0 ? tr("Charts (%1)").arg(charts.size()) : tr("Charts"));
  dock_btn->setText(docking ? "⬈" : "⬋");
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
}

void ChartsWidget::showChart(const QString &id, const Signal *sig, bool show) {
  auto it = std::find_if(charts.begin(), charts.end(), [=](auto c) { return c->id == id && c->signal == sig; });
  if (it != charts.end()) {
    if (!show) removeChart((*it));
  } else if (show) {
    auto chart = new ChartView(id, sig, this);
    chart->updateSeries(display_range);
    QObject::connect(chart, &ChartView::remove, [=]() { removeChart(chart); });
    QObject::connect(chart, &ChartView::zoomIn, this, &ChartsWidget::zoomIn);
    QObject::connect(chart, &ChartView::zoomReset, this, &ChartsWidget::zoomReset);
    charts_layout->insertWidget(0, chart);
    charts.push_back(chart);
    emit chartOpened(chart->id, chart->signal);
    updateState();
  }
  updateToolBar();
}

bool ChartsWidget::isChartOpened(const QString &id, const Signal *sig) {
  auto it = std::find_if(charts.begin(), charts.end(), [=](auto c) { return c->id == id && c->signal == sig; });
  return it != charts.end();
}

void ChartsWidget::removeChart(ChartView *chart) {
  charts.removeOne(chart);
  chart->deleteLater();
  updateToolBar();
  emit chartClosed(chart->id, chart->signal);
}

void ChartsWidget::removeAll(const Signal *sig) {
  for (auto c : charts.toVector())
    if (!sig || c->signal == sig) removeChart(c);
}

void ChartsWidget::signalUpdated(const Signal *sig) {
  for (auto c : charts) {
    if (c->signal == sig) {
      c->updateTitle();
      c->updateSeries(display_range);
      c->setRange(display_range.first, display_range.second, true);
    }
  }
}

bool ChartsWidget::eventFilter(QObject *obj, QEvent *event) {
  if (obj != this && event->type() == QEvent::Close) {
    emit dock_btn->triggered();
    return true;
  }
  return false;
}

// ChartView

ChartView::ChartView(const QString &id, const Signal *sig, QWidget *parent)
    : id(id), signal(sig), QChartView(nullptr, parent) {
  QLineSeries *series = new QLineSeries();
  QChart *chart = new QChart();
  chart->setBackgroundRoundness(0);
  chart->addSeries(series);
  chart->createDefaultAxes();
  chart->legend()->hide();
  chart->layout()->setContentsMargins(0, 0, 0, 0);
  // top margin for title
  chart->setMargins({0, 11, 0, 0});

  line_marker = new QGraphicsLineItem(chart);
  line_marker->setZValue(chart->zValue() + 10);

  track_line = new QGraphicsLineItem(chart);
  track_line->setPen(QPen(Qt::darkGray, 1, Qt::DashLine));
  track_ellipse = new QGraphicsEllipseItem(chart);
  track_ellipse->setBrush(Qt::darkGray);
  value_text = new QGraphicsTextItem(chart);
  item_group = scene()->createItemGroup({track_line, track_ellipse, value_text});
  item_group->setZValue(chart->zValue() + 10);

  // title
  QToolButton *remove_btn = new QToolButton();
  remove_btn->setText("X");
  remove_btn->setAutoRaise(true);
  remove_btn->setToolTip(tr("Remove Chart"));
  close_btn_proxy = new QGraphicsProxyWidget(chart);
  close_btn_proxy->setWidget(remove_btn);

  setChart(chart);
  setRenderHint(QPainter::Antialiasing);
  setRubberBand(QChartView::HorizontalRubberBand);
  updateFromSettings();
  updateTitle();

  QTimer *timer = new QTimer(this);
  timer->setInterval(100);
  timer->setSingleShot(true);
  timer->callOnTimeout(this, &ChartView::adjustChartMargins);

  QObject::connect(&settings, &Settings::changed, this, &ChartView::updateFromSettings);
  QObject::connect(remove_btn, &QToolButton::clicked, [=]() { emit remove(id, sig); });
  QObject::connect(chart, &QChart::plotAreaChanged, [=](const QRectF &plotArea) {
    // use a singleshot timer to avoid recursion call.
    timer->start();
  });
}

void ChartView::resizeEvent(QResizeEvent *event) {
  QChartView::resizeEvent(event);
  close_btn_proxy->setPos(event->size().width() - close_btn_proxy->size().width() - 11, 8);
}

void ChartView::updateTitle() {
  chart()->setTitle(tr("<font color=\"gray\" text-align:left>%1 %2</font> <b>%3</b>").arg(dbc()->msg(id)->name).arg(id).arg(signal->name.c_str()));
}

void ChartView::updateFromSettings() {
  setFixedHeight(settings.chart_height);
  chart()->setTheme(settings.chart_theme == 0 ? QChart::ChartThemeLight : QChart::QChart::ChartThemeDark);
  auto color = chart()->titleBrush().color();
  line_marker->setPen(QPen(color, 2));
}

void ChartView::setRange(double min, double max, bool force_update) {
  auto axis_x = dynamic_cast<QValueAxis *>(chart()->axisX());
  if (force_update || (min != axis_x->min() || max != axis_x->max())) {
    axis_x->setRange(min, max);
    updateAxisY();
  }
}

void ChartView::adjustChartMargins() {
  // TODO: Remove hardcoded aligned_pos
  const int aligned_pos = 60;
  if (chart()->plotArea().left() != aligned_pos) {
    const float left_margin = chart()->margins().left() + aligned_pos - chart()->plotArea().left();
    chart()->setMargins(QMargins(left_margin, 11, 0, 0));
    updateLineMarker(can->currentSec());
  }
}

void ChartView::updateLineMarker(double current_sec) {
  auto axis_x = dynamic_cast<QValueAxis *>(chart()->axisX());
  int x = chart()->plotArea().left() +
          chart()->plotArea().width() * (current_sec - axis_x->min()) / (axis_x->max() - axis_x->min());
  if (int(line_marker->line().x1()) != x) {
    line_marker->setLine(x, chart()->plotArea().top() - chart()->margins().top() + 3, x, height());
  }
}

void ChartView::updateSeries(const std::pair<double, double> range) {
  auto events = can->events();
  if (!events) return;

  vals.clear();
  vals.reserve((range.second - range.first) * 1000);  // [n]seconds * 1000hz
  auto [bus, address] = DBCManager::parseId(id);
  double route_start_time = can->routeStartTime();
  Event begin_event(cereal::Event::Which::INIT_DATA, (route_start_time + range.first) * 1e9);
  auto begin = std::lower_bound(events->begin(), events->end(), &begin_event, Event::lessThan());
  double end_ns = (route_start_time + range.second) * 1e9;
  for (auto it = begin; it != events->end() && (*it)->mono_time <= end_ns; ++it) {
    if ((*it)->which == cereal::Event::Which::CAN) {
      for (const auto &c : (*it)->event.getCan()) {
        if (bus == c.getSrc() && address == c.getAddress()) {
          auto dat = c.getDat();
          double value = get_raw_value((uint8_t *)dat.begin(), dat.size(), *signal);
          double ts = ((*it)->mono_time / (double)1e9) - route_start_time;  // seconds
          vals.push_back({ts, value});
        }
      }
    }
  }
  QLineSeries *series = (QLineSeries *)chart()->series()[0];
  series->replace(vals);
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  const auto axis_x = dynamic_cast<QValueAxis *>(chart()->axisX());
  const auto axis_y = dynamic_cast<QValueAxis *>(chart()->axisY());
  auto begin = std::lower_bound(vals.begin(), vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
  if (begin == vals.end())
    return;

  auto end = std::upper_bound(vals.begin(), vals.end(), axis_x->max(), [](double x, auto &p) { return x < p.x(); });
  const auto [min, max] = std::minmax_element(begin, end, [](auto &p1, auto &p2) { return p1.y() < p2.y(); });
  if (max->y() == min->y()) {
    axis_y->setRange(min->y() - 1, max->y() + 1);
  } else {
    double range = max->y() - min->y();
    axis_y->setRange(min->y() - range * 0.05, max->y() + range * 0.05);
    axis_y->applyNiceNumbers();
  }
}

void ChartView::leaveEvent(QEvent *event) {
  item_group->setVisible(false);
  QChartView::leaveEvent(event);
}

void ChartView::mouseReleaseEvent(QMouseEvent *event) {
  auto rubber = findChild<QRubberBand *>();
  if (event->button() == Qt::LeftButton && rubber && rubber->isVisible()) {
    rubber->hide();
    QRectF rect = rubber->geometry().normalized();
    rect.translate(-chart()->plotArea().topLeft());
    const auto axis_x = dynamic_cast<QValueAxis *>(chart()->axisX());
    double min = axis_x->min() + (rect.left() / chart()->plotArea().width()) * (axis_x->max() - axis_x->min());
    double max = axis_x->min() + (rect.right() / chart()->plotArea().width()) * (axis_x->max() - axis_x->min());
    if (rubber->width() <= 0) {
      // no rubber dragged, seek to mouse position
      can->seekTo(min);
    } else if ((max - min) >= 0.5) {
      // zoom in if selected range is greater than 0.5s
      emit zoomIn(min, max);
    }
    viewport()->update();
    event->accept();
  } else if (event->button() == Qt::RightButton) {
    emit zoomReset();
    event->accept();
  } else {
    QGraphicsView::mouseReleaseEvent(event);
  }
  setViewportUpdateMode(QGraphicsView::MinimalViewportUpdate);
}

void ChartView::mouseMoveEvent(QMouseEvent *ev) {
  auto rubber = findChild<QRubberBand *>();
  bool is_zooming = rubber && rubber->isVisible();
  if (!is_zooming) {
    const auto plot_area = chart()->plotArea();
    auto axis_x = dynamic_cast<QValueAxis *>(chart()->axisX());
    double x = std::clamp((double)ev->pos().x(), plot_area.left(), plot_area.right() - 1);
    double sec = axis_x->min() + (axis_x->max() - axis_x->min()) * (x - plot_area.left()) / plot_area.width();
    auto value = std::upper_bound(vals.begin(), vals.end(), sec, [](double x, auto &p) { return x < p.x(); });
    if (value != vals.end()) {
      QPointF pos = chart()->mapToPosition((*value));
      track_line->setLine(pos.x(), plot_area.top(), pos.x(), plot_area.bottom());
      track_ellipse->setRect(pos.x() - 5, pos.y() - 5, 10, 10);
      value_text->setHtml(tr("<div style='background-color:darkGray'><font color='white'>%1, %2)</font></div>")
                              .arg(value->x(), 0, 'f', 3).arg(value->y()));
      int text_x = pos.x() + 8;
      if ((text_x + value_text->boundingRect().width()) > plot_area.right()) {
        text_x = pos.x() - value_text->boundingRect().width() - 8;
      }
      value_text->setPos(text_x, pos.y() - 10);
    }
    item_group->setVisible(value != vals.end());
  } else {
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
  }
  QChartView::mouseMoveEvent(ev);
}
