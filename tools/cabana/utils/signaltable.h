#pragma once

#include <map>
#include <set>
#include <utility>
#include <vector>
#include <QAbstractTableModel>
#include <QTableView>

#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/streams/abstractstream.h"

class SignalTableModel : public QAbstractTableModel {
  Q_OBJECT
public:
  SignalTableModel(QObject *parent) : QAbstractTableModel(parent) {}
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return signal_items_.size(); }
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return 4; }
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  void updateState(const std::set<MessageId> *new_msgs, bool has_new_ids);
  void setSignals(const std::set<std::pair<MessageId, QString>> &sigs);

protected:
  void updateMessage(const MessageId &id, const CanData &data);

  struct Item {
    double seconds;
    QString msg_name;
    QString sig_name;
    QString unit;
    QString value;
  };
  std::vector<const Item *> signal_items_;
  std::set<std::pair<MessageId, QString>> signals_;
  std::map<std::pair<MessageId, QString>, Item> signals_map_;
  bool display_all_ = false;
  friend class SignalTable;
};

class SignalTable : public QTableView {
  Q_OBJECT
public:
  SignalTable(QWidget *parent);
  void wheelEvent(QWheelEvent *event) override;
  void setSignals(const std::set<std::pair<MessageId, QString>> &sigs) { model_->setSignals(sigs); }
  void setDisplayALlSignal(bool all) {
    model_->display_all_ = all;
    model_->updateState(nullptr, true);
  }
  void refresh() { model_->updateState(nullptr, true); }

protected:
  SignalTableModel *model_;
};

