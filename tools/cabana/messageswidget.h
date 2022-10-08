#pragma once

#include <QLineEdit>
#include <QTableWidget>
#include <QWidget>

#include "tools/cabana/canmessages.h"

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);

public slots:
  void updateState();
  void dbcSelectionChanged(const QString &dbc_file);

signals:
  void msgSelectionChanged(const QString &message_id);

protected:
  QLineEdit *filter;
  QTableWidget *table_widget;
};
