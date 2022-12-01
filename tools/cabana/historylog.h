#pragma once

#include <QHeaderView>
#include <QTableView>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

class HeaderView : public QHeaderView {
public:
  HeaderView(Qt::Orientation orientation, QWidget *parent = nullptr) : QHeaderView(orientation, parent) {}
  QSize sectionSizeFromContents(int logicalIndex) const override;
  void paintSection(QPainter *painter, const QRect &rect, int logicalIndex) const;
};

class HistoryLogModel : public QAbstractTableModel {
  Q_OBJECT

public:
  HistoryLogModel(QObject *parent) : QAbstractTableModel(parent) {}
  void setMessage(const QString &message_id);
  void updateState();
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return row_count; }
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return column_count; }

private:
  QString msg_id;
  int row_count = 0;
  int column_count = 2;
  const DBCMsg *dbc_msg = nullptr;
  std::deque<CanData> messages;
};

class HistoryLog : public QTableView {
  Q_OBJECT

public:
  HistoryLog(QWidget *parent);
  void setMessage(const QString &message_id) { model->setMessage(message_id); }
  void updateState() { model->updateState(); }
private:
  int sizeHintForColumn(int column) const override;
  HistoryLogModel *model;
};
