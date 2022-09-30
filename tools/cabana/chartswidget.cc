#include "tools/cabana/chartswidget.h"

#include <QtCharts/QLineSeries>

using namespace QtCharts;
ChartsWidget::ChartsWidget(QWidget *parent) : QWidget(parent) {
  main_layout = new QVBoxLayout(this);
  QLineSeries *series = new QLineSeries();
  series->append(0, 6);
  series->append(2, 4);
  series->append(3, 8);
  series->append(7, 4);
  series->append(10, 5);
  *series << QPointF(11, 1) << QPointF(13, 3) << QPointF(17, 6) << QPointF(18, 3) << QPointF(20, 2);
  auto *chart = new QChart();
  chart->addSeries(series);
  chart->createDefaultAxes();
  chart->setTitle("Simple line chart example");

  auto *chartView = new QChartView(chart);
  chartView->setRenderHint(QPainter::Antialiasing);

  main_layout->addWidget(chartView);
}

void ChartsWidget::addChart(uint32_t address) {
}

void ChartsWidget::removeChart(uint32_t address) {
}
