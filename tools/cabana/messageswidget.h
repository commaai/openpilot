#pragma once

#include <QLineEdit>
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
  void msgChanged(const CanData *id);

 protected:
  QLineEdit *filter;
  QTableWidget *table_widget;
};
