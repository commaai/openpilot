#pragma once

#include <QAbstractTableModel>
#include <QTableView>

#include "tools/cabana/canmessages.h"

class MessageListModel : public QAbstractTableModel {
Q_OBJECT

public:
  MessageListModel(QObject *parent) : QAbstractTableModel(parent) {}
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 4; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return row_count; }
  void updateState();

private:
  int row_count = 0;
};

class MessagesWidget : public QWidget {
  Q_OBJECT

public:
  MessagesWidget(QWidget *parent);

public slots:
  void dbcSelectionChanged(const QString &dbc_file);

signals:
  void msgSelectionChanged(const QString &message_id);

protected:
  QTableView *table_widget;
  MessageListModel *model;
};
