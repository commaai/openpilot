#include "tools/cabana/chart/chartswidget.h"

#include <QApplication>
#include <QFutureSynchronizer>
#include <QToolBar>
#include <QToolTip>
#include <QtConcurrent>

#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/chart/seriesdlg.h"

// ChartsWidget

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // toolbar
  QToolBar *toolbar = new QToolBar(tr("Charts"), this);
  toolbar->setIconSize({16, 16});

  QAction *new_plot_btn = toolbar->addAction(bootstrapPixmap("file-plus"), "");
  new_plot_btn->setToolTip(tr("New Plot"));
  toolbar->addWidget(title_label = new QLabel());
  title_label->setContentsMargins(0, 0, 12, 0);
  columns_cb = new QComboBox(this);
  columns_cb->addItems({"1", "2", "3", "4"});
  toolbar->addWidget(new QLabel(tr("Columns:")));
  toolbar->addWidget(columns_cb);

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

  toolbar->addWidget(zoom_range_lb = new QLabel());
  reset_zoom_btn = toolbar->addAction(bootstrapPixmap("arrow-counterclockwise"), "");
  reset_zoom_btn->setToolTip(tr("Reset zoom (drag on chart to zoom X-Axis)"));
  remove_all_btn = toolbar->addAction(bootstrapPixmap("x"), "");
  remove_all_btn->setToolTip(tr("Remove all charts"));
  dock_btn = toolbar->addAction("");
  main_layout->addWidget(toolbar);

  // charts
  QWidget *charts_container = new QWidget(this);
  QVBoxLayout *charts_main_layout = new QVBoxLayout(charts_container);
  charts_main_layout->setContentsMargins(0, 0, 0, 0);
  charts_main_layout->addLayout(charts_layout = new QGridLayout);
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
  zoom_range_lb->setText(is_zoomed ? tr("Zooming: %1 - %2").arg(zoomed_range.first, 0, 'f', 2).arg(zoomed_range.second, 0, 'f', 2) : "");
  title_label->setText(tr("Charts: %1").arg(charts.size()));
  dock_btn->setIcon(bootstrapPixmap(docking ? "arrow-up-right" : "arrow-down-left"));
  dock_btn->setToolTip(docking ? tr("Undock charts") : tr("Dock charts"));
  remove_all_btn->setEnabled(!charts.isEmpty());
  reset_zoom_btn->setEnabled(is_zoomed);
}

void ChartsWidget::settingChanged() {
  range_slider->setRange(1, settings.max_cached_minutes * 60);
  for (auto c : charts) {
    c->setFixedHeight(settings.chart_height);
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
  int n = column_count;
  for (; n > 1; --n) {
    if ((n * (CHART_MIN_WIDTH + charts_layout->spacing())) < rect().width()) break;
  }
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
