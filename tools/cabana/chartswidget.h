#pragma once

#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <map>

#include "tools/cabana/parser.h"

class ChartsWidget : public QWidget {
  Q_OBJECT

 public:
  ChartsWidget(QWidget *parent = nullptr);
  void addChart(uint32_t address, const QString &name);
  void removeChart(uint32_t address, const QString &name);
  void updateState();

 protected:
  QVBoxLayout *main_layout;
  uint32_t address_ = 0;
  std::map<QString, QtCharts::QChartView *> charts;
};
