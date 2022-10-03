#include "tools/cabana/chartswidget.h"

#include <QtCharts/QLineSeries>

using namespace QtCharts;

int64_t get_raw_value(const QByteArray &msg, const Signal &sig) {
  int64_t ret = 0;

  int i = sig.msb / 8;
  int bits = sig.size;
  while (i >= 0 && i < msg.size() && bits > 0) {
    int lsb = (int)(sig.lsb / 8) == i ? sig.lsb : i * 8;
    int msb = (int)(sig.msb / 8) == i ? sig.msb : (i + 1) * 8 - 1;
    int size = msb - lsb + 1;

    uint64_t d = (msg[i] >> (lsb - (i * 8))) & ((1ULL << size) - 1);
    ret |= d << (bits - size);

    bits -= size;
    i = sig.is_little_endian ? i - 1 : i + 1;
  }
  return ret;
}

ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  connect(parser, &Parser::updated, this, &ChartsWidget::updateState);
  connect(parser, &Parser::showPlot, this, &ChartsWidget::addChart);
  connect(parser, &Parser::hidePlot, this, &ChartsWidget::removeChart);
}

void ChartsWidget::addChart(uint32_t address, const QString &name) {
  address_ = address;
  if (charts.find(name) == charts.end()) {
    QLineSeries *series = new QLineSeries();
    series->setUseOpenGL(true);
    auto chart = new QChart();
    chart->setTitle(name);
    chart->addSeries(series);
    chart->createDefaultAxes();
    chart->legend()->hide();
    auto chart_view = new QChartView(chart);
    chart_view->setMinimumSize({width(), 300});
    chart_view->setMaximumSize({width(), 300});
    chart_view->setRenderHint(QPainter::Antialiasing);
    main_layout->addWidget(chart_view);
    charts[name] = {.chart_view = chart_view};
  }
}

void ChartsWidget::removeChart(uint32_t address, const QString &name) {
  auto it = charts.find(name);
  if (it == charts.end()) return;

  delete it->second.chart_view;
  charts.erase(it);
}

void ChartsWidget::updateState() {
  static double last_update = millis_since_boot();
  auto msg = parser->getMsg(address_);
  if (!msg) return;
  auto it = parser->items.find(address_);
  if (it == parser->items.end()) return;

  double current_ts = millis_since_boot();
  bool update = (current_ts - last_update) > 500;
  if (update) {
    last_update = current_ts;
  }
  for (auto &sig : msg->sigs) {
    if (charts.find(sig.name.c_str()) == charts.end()) continue;

    auto &dat = it->second.back().dat;
    int64_t val = get_raw_value(dat, sig);
    if (sig.is_signed) {
      val -= ((val >> (sig.size - 1)) & 0x1) ? (1ULL << sig.size) : 0;
    }
    double value = val * sig.factor + sig.offset;
    auto &signal_chart = charts[sig.name.c_str()];

    if (value > signal_chart.max_y) signal_chart.max_y = value;
    if (value < signal_chart.min_y) signal_chart.min_y = value;

    while (signal_chart.data.size() > DATA_LIST_SIZE) {
      signal_chart.data.pop_front();
    }
    signal_chart.data.push_back({(millis_since_boot() - signal_chart.ts_begin), value});

    if (update) {
      QChart *chart = signal_chart.chart_view->chart();
      QLineSeries *series = (QLineSeries *)chart->series()[0];
      series->replace(signal_chart.data);
      chart->axisX()->setRange(signal_chart.data.front().x(), signal_chart.data.back().x());
      chart->axisY()->setRange(signal_chart.min_y, signal_chart.max_y);
    }
  }
}
