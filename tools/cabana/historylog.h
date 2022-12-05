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
public:
  HistoryLogModel(QObject *parent) : QAbstractTableModel(parent) {}
  void setMessage(const QString &message_id);
  void updateState();
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return messages.size(); }
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return std::max(1ul, sigs.size()) + 1; }

private:
  QString msg_id;
  std::deque<CanData> messages;
  std::vector<const Signal*> sigs;
};

class HistoryLog : public QTableView {
public:
  HistoryLog(QWidget *parent);
  void setMessage(const QString &message_id) { model->setMessage(message_id); }
  void updateState() { model->updateState(); }

private:
  int sizeHintForColumn(int column) const override;
  HistoryLogModel *model;
};
