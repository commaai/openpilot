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
  connect(parser, &Parser::updated, this, &ChartsWidget::updateState);
  connect(parser, &Parser::showPlot, this, &ChartsWidget::addChart);
}

void ChartsWidget::addChart(uint32_t address, const QString &name) {
  address_ = address;
  if (charts.find(name) == charts.end()) {
    QLineSeries *series = new QLineSeries();
    auto chart = new QChart();
    chart->setTitle(name);
    chart->addSeries(series);
    chart->createDefaultAxes();
    auto chart_view = new QChartView(chart);
    chart_view->setRenderHint(QPainter::Antialiasing);
    main_layout->addWidget(chart_view);
    charts[name] = chart_view;
  }
}

void ChartsWidget::removeChart(uint32_t address, const QString &name) {
}

void ChartsWidget::updateState() {
  static double last_update = millis_since_boot();
  auto msg = parser->msg_map.find(address_);
  if (msg == parser->msg_map.end()) return;
  auto it = parser->items.find(address_);
  if (it == parser->items.end()) return;

  double current_ts = millis_since_boot();
  bool update = (current_ts - last_update) > 500;
  if (update) {
    last_update = current_ts;
  }
  for (auto &sig : msg->second->sigs) {
    if (charts.find(sig.name.c_str()) == charts.end()) continue;

    auto &dat = it->second.back().dat;
    int64_t val = get_raw_value(dat, sig);
    if (sig.is_signed) {
      val -= ((val >> (sig.size - 1)) & 0x1) ? (1ULL << sig.size) : 0;
    }
    double value = val * sig.factor + sig.offset;
    QChart *chart = charts[sig.name.c_str()]->chart();
    QLineSeries *series = (QLineSeries *)chart->series()[0];
    // while (series->count() >= DATA_LIST_SIZE) {
    //   series->remove(0);
    // }

    series->append(series->count(), value);
    if (update) {
      chart->axisX()->setRange(series->at(0).x(), series->at(series->count() - 1).x());
      chart->axisY()->setRange(-1, 1);
    }
  }
}
