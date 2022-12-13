#pragma once

#include <deque>
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
  HistoryLogModel(QObject *parent);
  void setMessage(const QString &message_id);
  void updateState();
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  void fetchMore(const QModelIndex &parent) override;
  inline bool canFetchMore(const QModelIndex &parent) const override { return has_more_data; }
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return messages.size(); }
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return std::max(1ul, sigs.size()) + 1; }

  struct Message {
    uint64_t mono_time = 0;
    QVector<double> sig_values;
    QByteArray data;
  };

  std::deque<Message> fetchData(uint64_t min_mono_time, uint64_t max_mono_time);
  QString msg_id;
  bool has_more_data = true;
  const int batch_size = 50;
  std::deque<Message> messages;
  std::vector<const Signal*> sigs;
};

class HistoryLog : public QTableView {
public:
  HistoryLog(QWidget *parent);
  void setMessage(const QString &message_id) { model->setMessage(message_id); }
  void updateState() { model->updateState(); }

private:
  int sizeHintForColumn(int column) const override { return -1; };
  void showEvent(QShowEvent *event) override { model->setMessage(model->msg_id); };
  HistoryLogModel *model;
};
