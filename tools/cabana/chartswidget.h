#pragma once

#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>
#include <map>

#include "tools/cabana/parser.h"

class ChartsWidget : public QWidget {
  Q_OBJECT

 public:
  ChartsWidget(QWidget *parent = nullptr);
  void addChart(uint32_t address);
  void removeChart(uint32_t address);

 protected:
  QVBoxLayout *main_layout;
  std::map<uint32_t, QtCharts::QChartView *> charts;
};
