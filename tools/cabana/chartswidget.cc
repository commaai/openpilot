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

void ChartsWidget::addChart(const QString &id, const QString &sig_name) {
  const QString char_name = id + sig_name;
  if (charts.find(char_name) == charts.end()) {
    QLineSeries *series = new QLineSeries();
    series->setUseOpenGL(true);
    auto chart = new QChart();
    chart->setTitle(id + ": " + sig_name);
    chart->addSeries(series);
    chart->createDefaultAxes();
    chart->legend()->hide();
    auto chart_view = new QChartView(chart);
    chart_view->setMinimumSize({width(), 300});
    chart_view->setMaximumSize({width(), 300});
    chart_view->setRenderHint(QPainter::Antialiasing);
    main_layout->addWidget(chart_view);
    charts[char_name] = {.id = id, .sig_name = sig_name, .chart_view = chart_view};
  }
}

void ChartsWidget::removeChart(const QString &id, const QString &sig_name) {
  auto it = charts.find(id + sig_name);
  if (it == charts.end()) return;

  delete it->second.chart_view;
  charts.erase(it);
}

void ChartsWidget::updateState() {
  static double last_update = millis_since_boot();
  double current_ts = millis_since_boot();
  bool update = (current_ts - last_update) > 500;
  if (update) {
    last_update = current_ts;
  }

  auto getSig = [=](const QString &id, const QString &name) -> const Signal * {
    for (auto &sig : parser->getMsg(id)->sigs) {
      if (name == sig.name.c_str()) return &sig;
    }
    return nullptr;
  };

  for (auto &[_, c] : charts) {
    if (auto sig = getSig(c.id, c.sig_name)) {
      const auto &can_data = parser->can_msgs[c.id].back();
      int64_t val = get_raw_value(can_data.dat, *sig);
      if (sig->is_signed) {
        val -= ((val >> (sig->size - 1)) & 0x1) ? (1ULL << sig->size) : 0;
      }
      double value = val * sig->factor + sig->offset;

      if (value > c.max_y) c.max_y = value;
      if (value < c.min_y) c.min_y = value;

      while (c.data.size() > DATA_LIST_SIZE) {
        c.data.pop_front();
      }
      c.data.push_back({can_data.ts / 1000., value});

      if (update) {
        QChart *chart = c.chart_view->chart();
        QLineSeries *series = (QLineSeries *)chart->series()[0];
        series->replace(c.data);
        chart->axisX()->setRange(c.data.front().x(), c.data.back().x());
        chart->axisY()->setRange(c.min_y, c.max_y);
      }
    }
  }
}
