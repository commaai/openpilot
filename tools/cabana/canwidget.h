#pragma once

#include <QApplication>
#include <QTableWidget>
#include <QWidget>

#include "tools/cabana/parser.h"

class CanWidget : public QWidget {
  Q_OBJECT

 public:
  CanWidget(QWidget *parent);

 public slots:
  void updateState();

 signals:
  void addressChanged(uint32_t address);

 protected:
  QTableWidget *table_widget;
};
