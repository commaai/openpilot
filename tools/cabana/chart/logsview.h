
#pragma once

#include <QAbstractTableModel>
#include <QTableView>

#include "tools/cabana/dbc/dbcmanager.h"

class MultipleSignalsLogModel : public QAbstractTableModel {
  Q_OBJECT

public:
  struct Signal {
    MessageId msg_id;
    const cabana::Signal *sig;
  };

  MultipleSignalsLogModel(QObject *parent) : QAbstractTableModel(parent) {}
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return values_.size(); }
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return sigs_.size() + 1; }
  void refresh() { setSignals(sigs_); }
  void setSignals(const std::vector<MultipleSignalsLogModel::Signal> &sigs);
  void updateState();

public slots:
  void signalUpdated(const cabana::Signal *sig);
  void msgUpdated(MessageId id);

private:
  uint64_t last_ts_ = 0;
  std::vector<Signal> sigs_;
  std::map<uint64_t, std::vector<std::optional<double>>> values_;
  friend class MultipleSignalsLogView;
};

class MultipleSignalsLogView : public QTableView {
  Q_OBJECT

public:
  MultipleSignalsLogView(QWidget *parent);
  QSize minimumSizeHint() const override;
  void setSignals(const std::vector<MultipleSignalsLogModel::Signal> &sigs);

 private:
  MultipleSignalsLogModel *model;
};
