#pragma once

#include <QTableView>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

class HistoryLogModel : public QAbstractTableModel {
Q_OBJECT

public:
  HistoryLogModel(QObject *parent) : QAbstractTableModel(parent) {}
  void setMessage(const QString &message_id);
  void updateState();
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return column_count; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return row_count; }

private:
  QString msg_id;
  int row_count = 0;
  int column_count = 0;
};

class HistoryLog : public QWidget {
  Q_OBJECT

public:
  HistoryLog(QWidget *parent);
  void setMessage(const QString &message_id);
  void updateState();

private:
  QTableView *table;
  HistoryLogModel *model;
};
