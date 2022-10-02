#pragma once

#include <QApplication>
#include <QTableWidget>
#include <QWidget>

#include "tools/cabana/parser.h"

class MessagesWidget : public QWidget {
  Q_OBJECT

 public:
  MessagesWidget(QWidget *parent);

 public slots:
  void updateState();

 signals:
  void addressChanged(uint32_t address);

 protected:
  QTableWidget *table_widget;
};
