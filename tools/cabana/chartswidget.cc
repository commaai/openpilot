#include "tools/cabana/chartswidget.h"

#include <QApplication>
#include <QCompleter>
#include <QDialogButtonBox>
#include <QDrag>
#include <QFutureSynchronizer>
#include <QGraphicsLayout>
#include <QLineEdit>
#include <QMenu>
#include <QRubberBand>
#include <QPushButton>
#include <QToolBar>
#include <QToolTip>
#include <QtConcurrent>

// ChartsWidget

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // toolbar
  QToolBar *toolbar = new QToolBar(tr("Charts"), this);
  toolbar->setIconSize({16, 16});

  QAction *new_plot_btn = toolbar->addAction(utils::icon("file-plus"), "");
  new_plot_btn->setToolTip(tr("New Plot"));
  toolbar->addWidget(title_label = new QLabel());
  title_label->setContentsMargins(0, 0, 12, 0);
  columns_cb = new QComboBox(this);
  columns_cb->addItems({"1", "2", "3", "4"});
  columns_lb_action = toolbar->addWidget(new QLabel(tr("Columns:")));
  columns_cb_action = toolbar->addWidget(columns_cb);

  QLabel *stretch_label = new QLabel(this);
  stretch_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  toolbar->addWidget(stretch_label);

  range_lb_action = toolbar->addWidget(range_lb = new QLabel(this));
  range_slider = new QSlider(Qt::Horizontal, this);
  range_slider->setToolTip(tr("Set the chart range"));
  range_slider->setRange(1, settings.max_cached_minutes * 60);
  range_slider->setSingleStep(1);
  range_slider->setPageStep(60);  // 1 min
  range_slider_action = toolbar->addWidget(range_slider);

  reset_zoom_action = toolbar->addWidget(reset_zoom_btn = new QToolButton());
  reset_zoom_btn->setIcon(utils::icon("zoom-out"));
  reset_zoom_btn->setToolTip(tr("Reset zoom"));
  reset_zoom_btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

  remove_all_btn = toolbar->addAction(utils::icon("x"), "");
  remove_all_btn->setToolTip(tr("Remove all charts"));
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
  charts_scroll->setWidgetResizable(true);
  charts_scroll->setWidget(charts_container);
  charts_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(charts_scroll);

  // init settings
  use_dark_theme = QApplication::style()->standardPalette().color(QPalette::WindowText).value() >
                   QApplication::style()->standardPalette().color(QPalette::Background).value();
  column_count = std::clamp(settings.chart_column_count, 1, columns_cb->count());
  max_chart_range = std::clamp(settings.chart_range, 1, settings.max_cached_minutes * 60);
  display_range = {0, max_chart_range};
  columns_cb->setCurrentIndex(column_count - 1);
  range_slider->setValue(max_chart_range);
  updateToolBar();

  QObject::connect(dbc(), &DBCManager::DBCFileChanged, this, &ChartsWidget::removeAll);
  QObject::connect(can, &AbstractStream::eventsMerged, this, &ChartsWidget::eventsMerged);
  QObject::connect(can, &AbstractStream::updated, this, &ChartsWidget::updateState);
  QObject::connect(range_slider, &QSlider::valueChanged, this, &ChartsWidget::setMaxChartRange);
  QObject::connect(new_plot_btn, &QAction::triggered, this, &ChartsWidget::newChart);
  QObject::connect(remove_all_btn, &QAction::triggered, this, &ChartsWidget::removeAll);
  QObject::connect(reset_zoom_btn, &QToolButton::clicked, this, &ChartsWidget::zoomReset);
  QObject::connect(columns_cb, SIGNAL(activated(int)), SLOT(setColumnCount(int)));
  QObject::connect(&settings, &Settings::changed, this, &ChartsWidget::settingChanged);
  QObject::connect(dock_btn, &QAction::triggered, [this]() {
    emit dock(!docking);
    docking = !docking;
    updateToolBar();
  });
}

void ChartsWidget::eventsMerged() {
  {
    assert(!can->liveStreaming());
    QFutureSynchronizer<void> future_synchronizer;
    const auto events = can->events();
    for (auto c : charts) {
      future_synchronizer.addFuture(QtConcurrent::run(c, &ChartView::updateSeries, nullptr, events, true));
    }
  }
  updateState();
}

void ChartsWidget::zoomIn(double min, double max) {
  zoomed_range = {min, max};
  is_zoomed = zoomed_range != display_range;
  updateToolBar();
  updateState();
  emit rangeChanged(min, max, is_zoomed);
}

void ChartsWidget::zoomReset() {
  zoomIn(display_range.first, display_range.second);
}

void ChartsWidget::updateState() {
  if (charts.isEmpty()) return;

  const auto events = can->events();
  if (can->liveStreaming()) {
    // appends incoming events to the end of series
    for (auto c : charts) {
      c->updateSeries(nullptr, events, false);
    }
  }

  const double cur_sec = can->currentSec();
  if (!is_zoomed) {
    double pos = (cur_sec - display_range.first) / std::max(1.0, (display_range.second - display_range.first));
    if (pos < 0 || pos > 0.8) {
      display_range.first = std::max(0.0, cur_sec - max_chart_range * 0.1);
    }
    double max_event_sec = events->empty() ? 0 : (events->back()->mono_time / 1e9 - can->routeStartTime());
    double max_sec = std::min(std::floor(display_range.first + max_chart_range), max_event_sec);
    display_range.first = std::max(0.0, max_sec - max_chart_range);
    display_range.second = display_range.first + max_chart_range;
  } else if (cur_sec < zoomed_range.first || cur_sec >= zoomed_range.second) {
    // loop in zoommed range
    can->seekTo(zoomed_range.first);
  }

  charts_layout->parentWidget()->setUpdatesEnabled(false);
  const auto &range = is_zoomed ? zoomed_range : display_range;
  for (auto c : charts) {
    c->updatePlot(cur_sec, range.first, range.second);
  }
  charts_layout->parentWidget()->setUpdatesEnabled(true);
}

void ChartsWidget::setMaxChartRange(int value) {
  max_chart_range = settings.chart_range = value;
  updateToolBar();
  updateState();
}

void ChartsWidget::updateToolBar() {
  title_label->setText(tr("Charts: %1").arg(charts.size()));
  range_lb->setText(QString("Range: %1:%2 ").arg(max_chart_range / 60, 2, 10, QLatin1Char('0')).arg(max_chart_range % 60, 2, 10, QLatin1Char('0')));
  range_lb_action->setVisible(!is_zoomed);
  range_slider_action->setVisible(!is_zoomed);
  reset_zoom_action->setVisible(is_zoomed);
  reset_zoom_btn->setText(is_zoomed ? tr("Zoomin: %1-%2").arg(zoomed_range.first, 0, 'f', 1).arg(zoomed_range.second, 0, 'f', 1) : "");
  remove_all_btn->setEnabled(!charts.isEmpty());
  dock_btn->setIcon(utils::icon(docking ? "arrow-up-right-square" : "arrow-down-left-square"));
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
}

void ChartsWidget::settingChanged() {
  range_slider->setRange(1, settings.max_cached_minutes * 60);
  for (auto c : charts) {
    c->setFixedHeight(settings.chart_height);
    c->setSeriesType(settings.chart_series_type == 0 ? QAbstractSeries::SeriesTypeLine : QAbstractSeries::SeriesTypeScatter);
  }
}

ChartView *ChartsWidget::findChart(const QString &id, const Signal *sig) {
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
  QObject::connect(chart, &ChartView::zoomReset, this, &ChartsWidget::zoomReset);
  QObject::connect(chart, &ChartView::seriesRemoved, this, &ChartsWidget::seriesChanged);
  QObject::connect(chart, &ChartView::seriesAdded, this, &ChartsWidget::seriesChanged);
  QObject::connect(chart, &ChartView::axisYLabelWidthChanged, this, &ChartsWidget::alignCharts);
  charts.push_back(chart);
  updateLayout();
  return chart;
}

void ChartsWidget::showChart(const QString &id, const Signal *sig, bool show, bool merge) {
  setUpdatesEnabled(false);
  ChartView *chart = findChart(id, sig);
  if (show && !chart) {
    chart = merge && charts.size() > 0 ? charts.back() : createChart();
    chart->addSeries(id, sig);
    updateState();
  } else if (!show && chart) {
    chart->removeSeries(id, sig);
  }
  updateToolBar();
  setUpdatesEnabled(true);
}

void ChartsWidget::setColumnCount(int n) {
  n = std::clamp(n + 1, 1, columns_cb->count());
  if (column_count != n) {
    column_count = settings.chart_column_count = n;
    updateLayout();
  }
}

void ChartsWidget::updateLayout() {
  int n = columns_cb->count();
  for (; n > 1; --n) {
    if ((n * CHART_MIN_WIDTH + (n - 1) * charts_layout->spacing()) < charts_layout->geometry().width()) break;
  }

  bool show_column_cb = n > 1;
  columns_lb_action->setVisible(show_column_cb);
  columns_cb_action->setVisible(show_column_cb);

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

// ChartView

ChartView::ChartView(QWidget *parent) : QChartView(nullptr, parent) {
  series_type = settings.chart_series_type == 0 ? QAbstractSeries::SeriesTypeLine : QAbstractSeries::SeriesTypeScatter;

  QChart *chart = new QChart();
  chart->setBackgroundVisible(false);
  axis_x = new QValueAxis(this);
  axis_y = new QValueAxis(this);
  chart->addAxis(axis_x, Qt::AlignBottom);
  chart->addAxis(axis_y, Qt::AlignLeft);
  chart->legend()->layout()->setContentsMargins(16, 0, 40, 0);
  chart->legend()->setShowToolTips(true);
  chart->setMargins({0, 0, 0, 0});

  background = new QGraphicsRectItem(chart);
  background->setBrush(Qt::white);
  background->setPen(Qt::NoPen);
  background->setZValue(chart->zValue() - 1);

  move_icon = new QGraphicsPixmapItem(utils::icon("grip-horizontal"), chart);
  move_icon->setToolTip(tr("Drag and drop to combine charts"));

  QToolButton *remove_btn = new QToolButton();
  remove_btn->setIcon(utils::icon("x"));
  remove_btn->setAutoRaise(true);
  remove_btn->setToolTip(tr("Remove Chart"));
  close_btn_proxy = new QGraphicsProxyWidget(chart);
  close_btn_proxy->setWidget(remove_btn);
  close_btn_proxy->setZValue(chart->zValue() + 11);

  QToolButton *manage_btn = new QToolButton();
  manage_btn->setToolButtonStyle(Qt::ToolButtonIconOnly);
  manage_btn->setIcon(utils::icon("list"));
  manage_btn->setAutoRaise(true);
  QMenu *menu = new QMenu(this);
  line_series_action = menu->addAction(tr("Line"), [this]() { setSeriesType(QAbstractSeries::SeriesTypeLine); });
  line_series_action->setCheckable(true);
  line_series_action->setChecked(series_type == QAbstractSeries::SeriesTypeLine);
  scatter_series_action = menu->addAction(tr("Scatter"), [this]() { setSeriesType(QAbstractSeries::SeriesTypeScatter); });
  scatter_series_action->setCheckable(true);
  scatter_series_action->setChecked(series_type == QAbstractSeries::SeriesTypeScatter);
  menu->addSeparator();
  menu->addAction(tr("Manage series"), this, &ChartView::manageSeries);
  manage_btn->setMenu(menu);
  manage_btn->setPopupMode(QToolButton::InstantPopup);
  manage_btn_proxy = new QGraphicsProxyWidget(chart);
  manage_btn_proxy->setWidget(manage_btn);
  manage_btn_proxy->setZValue(chart->zValue() + 11);

  setChart(chart);
  setRenderHint(QPainter::Antialiasing);
  // TODO: enable zoomIn/seekTo in live streaming mode.
  setRubberBand(can->liveStreaming() ? QChartView::NoRubberBand : QChartView::HorizontalRubberBand);

  QObject::connect(dbc(), &DBCManager::signalRemoved, this, &ChartView::signalRemoved);
  QObject::connect(dbc(), &DBCManager::signalUpdated, this, &ChartView::signalUpdated);
  QObject::connect(dbc(), &DBCManager::msgRemoved, this, &ChartView::msgRemoved);
  QObject::connect(dbc(), &DBCManager::msgUpdated, this, &ChartView::msgUpdated);
  QObject::connect(remove_btn, &QToolButton::clicked, this, &ChartView::remove);
}

void ChartView::addSeries(const QString &msg_id, const Signal *sig) {
  if (hasSeries(msg_id, sig)) return;

  QXYSeries *series = createSeries(series_type, getColor(sig));
  chart()->addSeries(series);
  series->attachAxis(axis_x);
  series->attachAxis(axis_y);
  auto [source, address] = DBCManager::parseId(msg_id);
  sigs.push_back({.msg_id = msg_id, .address = address, .source = source, .sig = sig, .series = series});
  updateTitle();
  updateSeries(sig);
  updateSeriesPoints();
  emit seriesAdded(msg_id, sig);
}

void ChartView::removeSeries(const QString &msg_id, const Signal *sig) {
  auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
  if (it != sigs.end()) {
    it = removeItem(it);
  }
}

bool ChartView::hasSeries(const QString &msg_id, const Signal *sig) const {
  return std::any_of(sigs.begin(), sigs.end(), [&](auto &s) { return s.msg_id == msg_id && s.sig == sig; });
}

QList<ChartView::SigItem>::iterator ChartView::removeItem(const QList<ChartView::SigItem>::iterator &it) {
  chart()->removeSeries(it->series);
  it->series->deleteLater();
  QString msg_id = it->msg_id;
  const Signal *sig = it->sig;
  auto ret = sigs.erase(it);
  emit seriesRemoved(msg_id, sig);
  if (!sigs.isEmpty()) {
    updateAxisY();
  } else {
    emit remove();
  }
  return ret;
}

void ChartView::signalUpdated(const Signal *sig) {
  if (std::any_of(sigs.begin(), sigs.end(), [=](auto &s) { return s.sig == sig; })) {
    updateTitle();
    // TODO: don't update series if only name changed.
    updateSeries(sig);
  }
}

void ChartView::signalRemoved(const Signal *sig) {
  for (auto it = sigs.begin(); it != sigs.end(); /**/) {
    it = (it->sig == sig) ? removeItem(it) : ++it;
  }
}

void ChartView::msgUpdated(uint32_t address) {
  if (std::any_of(sigs.begin(), sigs.end(), [=](auto &s) { return s.address == address; }))
    updateTitle();
}

void ChartView::msgRemoved(uint32_t address) {
  for (auto it = sigs.begin(); it != sigs.end(); /**/) {
    it = (it->address == address) ? removeItem(it) : ++it;
  }
}

void ChartView::manageSeries() {
  SeriesSelector dlg(tr("Mange Chart"), this);
  for (auto &s : sigs) {
    dlg.addSelected(s.msg_id, s.sig);
  }
  if (dlg.exec() == QDialog::Accepted) {
    auto items = dlg.seletedItems();
    if (items.isEmpty()) {
      emit remove();
    } else {
      for (auto s : items) {
        addSeries(s->msg_id, s->sig);
      }
      for (auto it = sigs.begin(); it != sigs.end(); /**/) {
        bool exists = std::any_of(items.cbegin(), items.cend(), [&](auto &s) {
          return s->msg_id == it->msg_id && s->sig == it->sig;
        });
        it = exists ? ++it : removeItem(it);
      }
    }
  }
}

void ChartView::resizeEvent(QResizeEvent *event) {
  QChartView::resizeEvent(event);
  updatePlotArea(align_to);
  int x = event->size().width() - close_btn_proxy->size().width() - 11;
  close_btn_proxy->setPos(x, 8);
  manage_btn_proxy->setPos(x - manage_btn_proxy->size().width() - 5, 8);
  move_icon->setPos(11, 8);
}

void ChartView::updatePlotArea(int left) {
  QRect r = rect();
  if (align_to != left || r != background->rect()) {
    align_to = left;
    background->setRect(r);
    chart()->legend()->setGeometry(QRect(r.left(), r.top(), r.width(), 45));
    chart()->setPlotArea(QRect(align_to, r.top() + 45, r.width() - align_to - 22, r.height() - 80));
    chart()->layout()->invalidate();
  }
}

void ChartView::updateTitle() {
  for (QLegendMarker *marker : chart()->legend()->markers()) {
    QObject::connect(marker, &QLegendMarker::clicked, this, &ChartView::handleMarkerClicked, Qt::UniqueConnection);
  }
  for (auto &s : sigs) {
    auto decoration = s.series->isVisible() ? "none" : "line-through";
    s.series->setName(QString("<span style=\"text-decoration:%1\"><b>%2</b> <font color=\"gray\">%3 %4</font></span>").arg(decoration, s.sig->name.c_str(), msgName(s.msg_id), s.msg_id));
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
    auto begin = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
    auto end = std::lower_bound(begin, s.vals.end(), axis_x->max(), [](auto &p, double x) { return p.x() < x; });

    int num_points = std::max<int>(end - begin, 1);
    int pixels_per_point = width() / num_points;

    if (series_type == QAbstractSeries::SeriesTypeScatter) {
      ((QScatterSeries *)s.series)->setMarkerSize(std::clamp(pixels_per_point / 3, 1, 8));
    } else {
      s.series->setPointsVisible(pixels_per_point > 20);
    }
  }
}

void ChartView::updateSeries(const Signal *sig, const std::vector<Event *> *events, bool clear) {
  events = events ? events : can->events();
  for (auto &s : sigs) {
    if (!sig || s.sig == sig) {
      if (clear) {
        s.vals.clear();
        s.vals.reserve(settings.max_cached_minutes * 60 * 100);  // [n]seconds * 100hz
        s.last_value_mono_time = 0;
      }
      s.series->setColor(getColor(s.sig));

      struct Chunk {
        std::vector<Event *>::const_iterator first, second;
        QVector<QPointF> vals;
      };
      // split into one minitue chunks
      QVector<Chunk> chunks;
      Event begin_event(cereal::Event::Which::INIT_DATA, s.last_value_mono_time);
      auto begin = std::upper_bound(events->begin(), events->end(), &begin_event, Event::lessThan());
      for (auto it = begin, second = begin; it != events->end(); it = second) {
        second = std::lower_bound(it, events->end(), (*it)->mono_time + 1e9 * 60, [](auto &e, uint64_t ts) { return e->mono_time < ts; });
        chunks.push_back({it, second});
      }

      QtConcurrent::blockingMap(chunks, [&](Chunk &chunk) {
        chunk.vals.reserve(60 * 100);  // 100 hz
        double route_start_time = can->routeStartTime();
        for (auto it = chunk.first; it != chunk.second; ++it) {
          if ((*it)->which == cereal::Event::Which::CAN) {
            for (const auto &c : (*it)->event.getCan()) {
              if (s.address == c.getAddress() && s.source == c.getSrc()) {
                auto dat = c.getDat();
                double value = get_raw_value((uint8_t *)dat.begin(), dat.size(), *s.sig);
                double ts = ((*it)->mono_time / (double)1e9) - route_start_time;  // seconds
                chunk.vals.push_back({ts, value});
              }
            }
          }
        }
      });
      for (auto &c : chunks) {
        s.vals.append(c.vals);
      }
      if (events->size()) {
        s.last_value_mono_time = events->back()->mono_time;
      }
      s.series->replace(s.vals);
    }
  }
  updateAxisY();
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  if (sigs.isEmpty()) return;

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();
  for (auto &s : sigs) {
    if (!s.series->isVisible()) continue;

    auto first = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
    auto last = std::lower_bound(first, s.vals.end(), axis_x->max(), [](auto &p, double x) { return p.x() < x; });
    for (auto it = first; it != last; ++it) {
      if (it->y() < min) min = it->y();
      if (it->y() > max) max = it->y();
    }
  }
  if (min == std::numeric_limits<double>::max()) min = 0;
  if (max == std::numeric_limits<double>::lowest()) max = 0;

  double delta = std::abs(max - min) < 1e-3 ? 1 : (max - min) * 0.05;
  auto [min_y, max_y, tick_count] = getNiceAxisNumbers(min - delta, max + delta, axis_y->tickCount());
  if (min_y != axis_y->min() || max_y != axis_y->max() || y_label_width == 0) {
    axis_y->setRange(min_y, max_y);
    axis_y->setTickCount(tick_count);

    QFontMetrics fm(axis_y->labelsFont());
    int n = qMax(int(-qFloor(std::log10((max_y - min_y) / (tick_count - 1)))), 0) + 1;
    y_label_width = qMax(fm.width(QString::number(min_y, 'f', n)), fm.width(QString::number(max_y, 'f', n))) + 20;  // left margin 20
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
  track_pts.clear();
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
    Qt::DropAction dropAction = drag->exec(Qt::CopyAction | Qt::MoveAction, Qt::MoveAction);
    if (dropAction == Qt::MoveAction) {
      return;
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
    }
    event->accept();
  } else if (!can->liveStreaming() && event->button() == Qt::RightButton) {
    emit zoomReset();
    event->accept();
  } else {
    QGraphicsView::mouseReleaseEvent(event);
  }
}

void ChartView::mouseMoveEvent(QMouseEvent *ev) {
  auto rubber = findChild<QRubberBand *>();
  bool is_zooming = rubber && rubber->isVisible();
  const auto plot_area = chart()->plotArea();
  track_pts.clear();
  if (!is_zooming && plot_area.contains(ev->pos())) {
    track_pts.resize(sigs.size());
    QStringList text_list;
    const double sec = chart()->mapToValue(ev->pos()).x();
    for (int i = 0; i < sigs.size(); ++i) {
      QString value = "--";
      // use reverse iterator to find last item <= sec.
      auto it = std::lower_bound(sigs[i].vals.rbegin(), sigs[i].vals.rend(), sec, [](auto &p, double x) { return p.x() > x; });
      if (it != sigs[i].vals.rend() && it->x() >= axis_x->min()) {
        value = QString::number(it->y());
        track_pts[i] = chart()->mapToPosition(*it);
      }
      text_list.push_back(QString("<span style=\"color:%1;\">■ </span>%2: <b>%3</b>").arg(sigs[i].series->color().name(), sigs[i].sig->name.c_str(), value));
    }
    auto max = std::max_element(track_pts.begin(), track_pts.end(), [](auto &a, auto &b) { return a.x() < b.x(); });
    auto pt = (max == track_pts.end()) ? ev->pos() : *max;
    text_list.push_front(QString::number(chart()->mapToValue(pt).x(), 'f', 3));
    QPointF tooltip_pt(pt.x() + 12, plot_area.top() - 20);
    QToolTip::showText(mapToGlobal(tooltip_pt.toPoint()), pt.isNull() ? "" : text_list.join("<br />"), this, plot_area.toRect());
    scene()->update();
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
  qreal x = chart()->mapToPosition(QPointF{cur_sec, 0}).x();
  x = std::clamp(x, chart()->plotArea().left(), chart()->plotArea().right());
  qreal y1 = chart()->plotArea().top() - 2;
  qreal y2 = chart()->plotArea().bottom() + 2;
  painter->setPen(QPen(chart()->titleBrush().color(), 2));
  painter->drawLine(QPointF{x, y1}, QPointF{x, y2});

  auto max = std::max_element(track_pts.begin(), track_pts.end(), [](auto &a, auto &b) { return a.x() < b.x(); });
  if (max != track_pts.end() && !max->isNull()) {
    painter->setPen(QPen(Qt::darkGray, 1, Qt::DashLine));
    painter->drawLine(QPointF{max->x(), y1}, QPointF{max->x(), y2});
    painter->setPen(Qt::NoPen);
    for (int i = 0; i < track_pts.size(); ++i) {
      if (!track_pts[i].isNull() && i < sigs.size()) {
        painter->setBrush(sigs[i].series->color().darker(125));
        painter->drawEllipse(track_pts[i], 5.5, 5.5);
      }
    }
  }

  // paint points. OpenGL mode lacks certain features (such as showing points)
  painter->setPen(Qt::NoPen);
  for (auto &s : sigs) {
    if (s.series->useOpenGL() && s.series->isVisible() && s.series->pointsVisible()) {
      auto first = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
      auto last = std::lower_bound(first, s.vals.end(), axis_x->max(), [](auto &p, double x) { return p.x() < x; });
      for (auto it = first; it != last; ++it) {
        painter->setBrush(s.series->color());
        painter->drawEllipse(chart()->mapToPosition(*it), 4, 4);
      }
    }
  }
}

QXYSeries *ChartView::createSeries(QAbstractSeries::SeriesType type, QColor color) {
  QXYSeries *series = nullptr;
  if (type == QAbstractSeries::SeriesTypeLine) {
    series = new QLineSeries(this);
    chart()->legend()->setMarkerShape(QLegend::MarkerShapeRectangle);
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
  pen.setWidth(2.0 * qApp->devicePixelRatio());
  series->setPen(pen);
#endif
  return series;
}

void ChartView::setSeriesType(QAbstractSeries::SeriesType type) {
  line_series_action->setChecked(type == QAbstractSeries::SeriesTypeLine);
  scatter_series_action->setChecked(type == QAbstractSeries::SeriesTypeScatter);
  if (type != series_type) {
    series_type = type;
    for (auto &s : sigs) {
      chart()->removeSeries(s.series);
      s.series->deleteLater();
    }
    for (auto &s : sigs) {
      auto series = createSeries(series_type, getColor(s.sig));
      chart()->addSeries(series);
      series->attachAxis(axis_x);
      series->attachAxis(axis_y);
      series->replace(s.vals);
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

  for (auto it = can->can_msgs.cbegin(); it != can->can_msgs.cend(); ++it) {
    if (auto m = dbc()->msg(it.key())) {
      msgs_combo->addItem(QString("%1 (%2)").arg(m->name).arg(it.key()), it.key());
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
  QObject::connect(remove_btn, &QPushButton::clicked, [this]() { if (auto item = selected_list->currentItem()) remove(item);});
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
  if (it->msg_id == msgs_combo->currentData().toString()) {
    addItemToList(available_list, it->msg_id, it->sig);
  }
  delete item;
}

void SeriesSelector::updateAvailableList(int index) {
  if (index == -1) return;
  available_list->clear();
  QString msg_id = msgs_combo->itemData(index).toString();
  auto selected_items = seletedItems();
  for (auto &[name, s] : dbc()->msg(msg_id)->sigs) {
    bool is_selected = std::any_of(selected_items.begin(), selected_items.end(), [=, sig=&s](auto it) { return it->msg_id == msg_id && it->sig == sig; });
    if (!is_selected) {
      addItemToList(available_list, msg_id, &s);
    }
  }
}

void SeriesSelector::addItemToList(QListWidget *parent, const QString id, const Signal *sig, bool show_msg_name) {
  QString text = QString("<span style=\"color:%0;\">■ </span> %1").arg(getColor(sig).name(), sig->name.c_str());
  if (show_msg_name) text += QString(" <font color=\"gray\">%0 %1</font>").arg(msgName(id), id);

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
