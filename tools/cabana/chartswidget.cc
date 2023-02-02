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
#include <QToolBar>
#include <QToolButton>
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

  toolbar->addWidget(new QLabel(tr("Range:")));
  toolbar->addWidget(range_lb = new QLabel(this));
  range_slider = new QSlider(Qt::Horizontal, this);
  range_slider->setToolTip(tr("Set the chart range"));
  range_slider->setRange(1, settings.max_cached_minutes * 60);
  range_slider->setSingleStep(1);
  range_slider->setPageStep(60);  // 1 min
  toolbar->addWidget(range_slider);

  reset_zoom_btn = toolbar->addAction(utils::icon("zoom-out"), "");
  reset_zoom_btn->setToolTip(tr("Reset zoom (drag on chart to zoom X-Axis)"));
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

  align_charts_timer = new QTimer(this);
  align_charts_timer->setSingleShot(true);
  align_charts_timer->callOnTimeout(this, &ChartsWidget::alignCharts);

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
  QObject::connect(reset_zoom_btn, &QAction::triggered, this, &ChartsWidget::zoomReset);
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

  if (can->liveStreaming()) {
    // appends incoming events to the end of series
    const auto events = can->events();
    for (auto c : charts) {
      c->updateSeries(nullptr, events, false);
    }
  }

  const double cur_sec = can->currentSec();
  if (!is_zoomed) {
    double pos = (cur_sec - display_range.first) / max_chart_range;
    if (pos < 0 || pos > 0.8) {
      const double min_event_sec = (can->events()->front()->mono_time / (double)1e9) - can->routeStartTime();
      display_range.first = std::floor(std::max(min_event_sec, cur_sec - max_chart_range * 0.2));
    }
    display_range.second = std::floor(display_range.first + max_chart_range);
  } else if (cur_sec < zoomed_range.first || cur_sec >= zoomed_range.second) {
    // loop in zoommed range
    can->seekTo(zoomed_range.first);
  }

  setUpdatesEnabled(false);
  const auto &range = is_zoomed ? zoomed_range : display_range;
  for (auto c : charts) {
    c->updatePlot(cur_sec, range.first, range.second);
  }
  setUpdatesEnabled(true);
}

void ChartsWidget::setMaxChartRange(int value) {
  max_chart_range = settings.chart_range = value;
  double current_sec = can->currentSec();
  const double min_event_sec = (can->events()->front()->mono_time / (double)1e9) - can->routeStartTime();
  // keep current_sec's pos
  double pos = (current_sec - display_range.first) / (display_range.second - display_range.first);
  display_range.first = std::floor(std::max(min_event_sec, current_sec - max_chart_range * (1.0 - pos)));
  display_range.second = std::floor(display_range.first + max_chart_range);
  updateToolBar();
  updateState();
}

void ChartsWidget::updateToolBar() {
  range_lb->setText(QString(" %1:%2 ").arg(max_chart_range / 60, 2, 10, QLatin1Char('0')).arg(max_chart_range % 60, 2, 10, QLatin1Char('0')));
  title_label->setText(tr("Charts: %1").arg(charts.size()));
  dock_btn->setIcon(utils::icon(docking ? "arrow-up-right" : "arrow-down-left"));
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
  remove_all_btn->setEnabled(!charts.isEmpty());
  reset_zoom_btn->setEnabled(is_zoomed);
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
  chart->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  chart->chart()->setTheme(use_dark_theme ? QChart::QChart::ChartThemeDark : QChart::ChartThemeLight);
  QObject::connect(chart, &ChartView::remove, [=]() { removeChart(chart); });
  QObject::connect(chart, &ChartView::zoomIn, this, &ChartsWidget::zoomIn);
  QObject::connect(chart, &ChartView::zoomReset, this, &ChartsWidget::zoomReset);
  QObject::connect(chart, &ChartView::seriesRemoved, this, &ChartsWidget::seriesChanged);
  QObject::connect(chart, &ChartView::seriesAdded, this, &ChartsWidget::seriesChanged);
  QObject::connect(chart, &ChartView::axisYUpdated, [this]() { align_charts_timer->start(100); });
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
  for (int i = 0; i < charts.size(); ++i) {
    charts_layout->addWidget(charts[charts.size() - i - 1], i / n, i % n);
  }
}

void ChartsWidget::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  updateLayout();
}

void ChartsWidget::newChart() {
  SeriesSelector dlg(this);
  if (dlg.exec() == QDialog::Accepted) {
    QList<QStringList> series_list = dlg.series();
    if (!series_list.isEmpty()) {
      auto c = createChart();
      c->addSeries(series_list);
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
    plot_left = qMax((qreal)plot_left, c->getYAsixLabelWidth());
  }
  for (auto c : charts) {
    c->setPlotAreaLeftPosition(plot_left);
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
  chart->setBackgroundRoundness(0);
  axis_x = new QValueAxis(this);
  axis_y = new QValueAxis(this);
  chart->addAxis(axis_x, Qt::AlignBottom);
  chart->addAxis(axis_y, Qt::AlignLeft);
  chart->legend()->layout()->setContentsMargins(0, 0, 40, 0);
  chart->legend()->setShowToolTips(true);
  chart->layout()->setContentsMargins(0, 0, 0, 0);
  chart->setMargins({20, 11, 11, 11});

  QToolButton *remove_btn = new QToolButton();
  remove_btn->setIcon(utils::icon("x"));
  remove_btn->setAutoRaise(true);
  remove_btn->setToolTip(tr("Remove Chart"));
  close_btn_proxy = new QGraphicsProxyWidget(chart);
  close_btn_proxy->setWidget(remove_btn);
  close_btn_proxy->setZValue(chart->zValue() + 11);

  QToolButton *manage_btn = new QToolButton();
  manage_btn->setToolButtonStyle(Qt::ToolButtonIconOnly);
  manage_btn->setIcon(utils::icon("gear"));
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

qreal ChartView::getYAsixLabelWidth() const {
  if (axis_y->max() <= axis_y->min() || axis_y->tickCount() <= 1) {
    return 0;
  }
  QFontMetrics fm(axis_y->labelsFont());
  int n = qMax(int(-qFloor(std::log10((axis_y->max() - axis_y->min()) / (axis_y->tickCount() - 1)))), 0) + 1;
  return qMax(fm.width(QString::number(axis_y->min(), 'f', n)), fm.width(QString::number(axis_y->max(), 'f', n))) + 20;
}

void ChartView::setPlotAreaLeftPosition(int pos) {
  if (std::ceil(chart()->plotArea().left()) != pos) {
    const float left_margin = chart()->margins().left() + pos - chart()->plotArea().left();
    chart()->setMargins(QMargins(left_margin, 11, 11, 11));
  }
}

void ChartView::addSeries(const QString &msg_id, const Signal *sig) {
  QXYSeries *series = createSeries(series_type);
  chart()->addSeries(series);
  series->attachAxis(axis_x);
  series->attachAxis(axis_y);
  auto [source, address] = DBCManager::parseId(msg_id);
  sigs.push_back({.msg_id = msg_id, .address = address, .source = source, .sig = sig, .series = series});
  updateTitle();
  updateSeries(sig);
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

void ChartView::addSeries(const QList<QStringList> &series_list) {
  for (auto &s : series_list) {
    if (auto m = dbc()->msg(s[0])) {
      auto it = m->sigs.find(s[2]);
      if (it != m->sigs.end() && !hasSeries(s[0], &(it->second))) {
        addSeries(s[0], &(it->second));
      }
    }
  }
}

void ChartView::manageSeries() {
  SeriesSelector dlg(this);
  for (auto &s : sigs) {
    dlg.addSeries(s.msg_id, msgName(s.msg_id), QString::fromStdString(s.sig->name));
  }

  int ret = dlg.exec();
  if (ret == QDialog::Accepted) {
    QList<QStringList> series_list = dlg.series();
    if (series_list.isEmpty()) {
      emit remove();
    } else {
      addSeries(series_list);
      for (auto it = sigs.begin(); it != sigs.end(); /**/) {
        bool exists = std::any_of(series_list.cbegin(), series_list.cend(), [&](auto &s) {
          return s[0] == it->msg_id && s[2] == it->sig->name.c_str();
        });
        it = exists ? ++it : removeItem(it);
      }
    }
  }
}

void ChartView::resizeEvent(QResizeEvent *event) {
  QChartView::resizeEvent(event);
  int x = event->size().width() - close_btn_proxy->size().width() - 11;
  close_btn_proxy->setPos(x, 8);
  manage_btn_proxy->setPos(x - manage_btn_proxy->size().width() - 5, 8);
}

void ChartView::updateTitle() {
  for (auto &s : sigs) {
    s.series->setName(QString("<b>%1</b> <font color=\"gray\">%2 %3</font>").arg(s.sig->name.c_str()).arg(msgName(s.msg_id)).arg(s.msg_id));
  }
}

void ChartView::updatePlot(double cur, double min, double max) {
  cur_sec = cur;
  if (min != axis_x->min() || max != axis_x->max()) {
    axis_x->setRange(min, max);
    updateAxisY();

    // Show points when zoomed in enough
    for (auto &s : sigs) {
      auto begin = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
      auto end = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->max(), [](auto &p, double x) { return p.x() < x; });

      int num_points = std::max<int>(end - begin, 1);
      int pixels_per_point = width() / num_points;

      if (series_type == QAbstractSeries::SeriesTypeScatter) {
        ((QScatterSeries *)s.series)->setMarkerSize(std::clamp(pixels_per_point / 3, 1, 8));
      } else {
        s.series->setPointsVisible(pixels_per_point > 20);

        // TODO: On MacOS QChartWidget doesn't work with the OpenGL settings that CameraWidget needs.
#ifndef __APPLE
        // OpenGL mode lacks certain features (such as showing points), only use when drawing many points
        bool use_opengl = pixels_per_point < 1;
        s.series->setUseOpenGL(use_opengl);

        // Qt doesn't properly apply device pixel ratio in OpenGL mode
        QApplication *application = static_cast<QApplication *>(QApplication::instance());
        float scale = use_opengl ? application->devicePixelRatio() : 1.0;

        QPen pen = s.series->pen();
        pen.setWidth(2.0 * scale);
        s.series->setPen(pen);
#endif
      }
    }
  }

  scene()->invalidate({}, QGraphicsScene::ForegroundLayer);
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
      double route_start_time = can->routeStartTime();
      Event begin_event(cereal::Event::Which::INIT_DATA, s.last_value_mono_time);
      auto begin = std::upper_bound(events->begin(), events->end(), &begin_event, Event::lessThan());
      for (auto it = begin; it != events->end(); ++it) {
        if ((*it)->which == cereal::Event::Which::CAN) {
          for (const auto &c : (*it)->event.getCan()) {
            if (s.source == c.getSrc() && s.address == c.getAddress()) {
              auto dat = c.getDat();
              double value = get_raw_value((uint8_t *)dat.begin(), dat.size(), *s.sig);
              double ts = ((*it)->mono_time / (double)1e9) - route_start_time;  // seconds
              s.vals.push_back({ts, value});
            }
          }
        }
      }
      if (events->size()) {
        s.last_value_mono_time = events->back()->mono_time;
      }
      s.series->replace(s.vals);
      updateAxisY();
    }
  }
}

// auto zoom on yaxis
void ChartView::updateAxisY() {
  if (sigs.isEmpty()) return;

  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  for (auto &s : sigs) {
    auto begin = std::lower_bound(s.vals.begin(), s.vals.end(), axis_x->min(), [](auto &p, double x) { return p.x() < x; });
    for (auto it = begin; it != s.vals.end() && it->x() <= axis_x->max(); ++it) {
      if (it->y() < min_y) min_y = it->y();
      if (it->y() > max_y) max_y = it->y();
    }
  }

  if (min_y == std::numeric_limits<double>::max()) min_y = 0;
  if (max_y == std::numeric_limits<double>::lowest()) max_y = 0;
  if (std::abs(max_y - min_y) < 1e-3) {
    applyNiceNumbers(min_y - 1, max_y + 1);
  } else {
    double range = max_y - min_y;
    applyNiceNumbers(min_y - range * 0.05, max_y + range * 0.05);
  }
  emit axisYUpdated();
}

void ChartView::applyNiceNumbers(qreal min, qreal max) {
  int tick_count = axis_y->tickCount();
  qreal range = niceNumber((max - min), true);  // range with ceiling
  qreal step = niceNumber(range / (tick_count - 1), false);
  min = qFloor(min / step);
  max = qCeil(max / step);
  tick_count = int(max - min) + 1;
  axis_y->setRange(min * step, max * step);
  axis_y->setTickCount(tick_count);
  axis_y->setLabelFormat("%.1f");
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
  if (event->button() == Qt::LeftButton && !chart()->plotArea().contains(event->pos()) &&
      !manage_btn_proxy->widget()->underMouse() && !close_btn_proxy->widget()->underMouse()) {
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
    min = std::clamp(min, can->routeStartTime(), can->routeStartTime() + can->totalSeconds());
    max = std::clamp(max, can->routeStartTime(), can->routeStartTime() + can->totalSeconds());

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
      QList<QStringList> series;
      for (auto &s : source_chart->sigs) {
        series.push_back({s.msg_id, msgName(s.msg_id), QString::fromStdString(s.sig->name)});
      }
      addSeries(series);
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
}

QXYSeries *ChartView::createSeries(QAbstractSeries::SeriesType type) {
  QXYSeries *series = nullptr;
  if (type == QAbstractSeries::SeriesTypeLine) {
    series = new QLineSeries(this);
  } else {
    series = new QScatterSeries(this);
  }
    // TODO: Due to a bug in CameraWidget the camera frames
    // are drawn instead of the graphs on MacOS. Re-enable OpenGL when fixed
#ifndef __APPLE__
  series->setUseOpenGL(true);
#endif
  return series;
}

void ChartView::setSeriesType(QAbstractSeries::SeriesType type) {
  if (type != series_type) {
    series_type = type;
    line_series_action->setChecked(type == QAbstractSeries::SeriesTypeLine);
    scatter_series_action->setChecked(type == QAbstractSeries::SeriesTypeScatter);

    for (auto &s : sigs) {
      chart()->removeSeries(s.series);
      s.series->deleteLater();
    }
    for (auto &s : sigs) {
      auto series = createSeries(series_type);
      chart()->addSeries(series);
      series->attachAxis(axis_x);
      series->attachAxis(axis_y);
      series->replace(s.vals);
      s.series = series;
    }
    updateTitle();
  }
}

// SeriesSelector

SeriesSelector::SeriesSelector(QWidget *parent) {
  setWindowTitle(tr("Manage Chart Series"));
  QHBoxLayout *contents_layout = new QHBoxLayout();

  QVBoxLayout *left_layout = new QVBoxLayout();
  left_layout->addWidget(new QLabel(tr("Select Signals:")));

  msgs_combo = new QComboBox(this);
  msgs_combo->setEditable(true);
  msgs_combo->lineEdit()->setPlaceholderText(tr("Select Msg"));
  msgs_combo->setInsertPolicy(QComboBox::NoInsert);
  msgs_combo->completer()->setCompletionMode(QCompleter::PopupCompletion);
  msgs_combo->completer()->setFilterMode(Qt::MatchContains);

  left_layout->addWidget(msgs_combo);
  sig_list = new QListWidget(this);
  sig_list->setSortingEnabled(true);
  sig_list->setToolTip(tr("Double click on an item to add signal to chart"));
  left_layout->addWidget(sig_list);

  QVBoxLayout *right_layout = new QVBoxLayout();
  right_layout->addWidget(new QLabel(tr("Chart Signals:")));
  chart_series = new QListWidget(this);
  chart_series->setSortingEnabled(true);
  chart_series->setToolTip(tr("Double click on an item to remove signal from chart"));
  right_layout->addWidget(chart_series);
  contents_layout->addLayout(left_layout);
  contents_layout->addLayout(right_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(contents_layout);
  main_layout->addWidget(buttonBox);

  for (auto it = can->can_msgs.cbegin(); it != can->can_msgs.cend(); ++it) {
    if (auto m = dbc()->msg(it.key())) {
      msgs_combo->addItem(QString("%1 (%2)").arg(m->name).arg(it.key()), it.key());
    }
  }
  msgs_combo->model()->sort(0);

  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(msgs_combo, SIGNAL(currentIndexChanged(int)), SLOT(msgSelected(int)));
  QObject::connect(sig_list, &QListWidget::itemDoubleClicked, this, &SeriesSelector::addSignal);
  QObject::connect(chart_series, &QListWidget::itemDoubleClicked, [](QListWidgetItem *item) { delete item; });

  if (int index = msgs_combo->currentIndex(); index >= 0) {
    msgSelected(index);
  }
}

void SeriesSelector::msgSelected(int index) {
  QString msg_id = msgs_combo->itemData(index).toString();
  sig_list->clear();
  if (auto m = dbc()->msg(msg_id)) {
    for (auto &[name, s] : m->sigs) {
      QStringList data({msg_id, m->name, name});
      QListWidgetItem *item = new QListWidgetItem(name, sig_list);
      item->setData(Qt::UserRole, data);
      sig_list->addItem(item);
    }
  }
}

void SeriesSelector::addSignal(QListWidgetItem *item) {
  QStringList data = item->data(Qt::UserRole).toStringList();
  addSeries(data[0], data[1], data[2]);
}

void SeriesSelector::addSeries(const QString &id, const QString &msg_name, const QString &sig_name) {
  QStringList data({id, msg_name, sig_name});
  for (int i = 0; i < chart_series->count(); ++i) {
    if (chart_series->item(i)->data(Qt::UserRole).toStringList() == data) {
      return;
    }
  }
  QListWidgetItem *new_item = new QListWidgetItem(chart_series);
  new_item->setData(Qt::UserRole, data);
  chart_series->addItem(new_item);
  QLabel *label = new QLabel(QString("%0 <font color=\"gray\">%1 %2</font>").arg(data[2]).arg(data[1]).arg(data[0]), chart_series);
  label->setContentsMargins(5, 0, 5, 0);
  new_item->setSizeHint(label->sizeHint());
  chart_series->setItemWidget(new_item, label);
}

QList<QStringList> SeriesSelector::series() {
  QList<QStringList> ret;
  for (int i = 0; i < chart_series->count(); ++i) {
    ret.push_back(chart_series->item(i)->data(Qt::UserRole).toStringList());
  }
  return ret;
}
