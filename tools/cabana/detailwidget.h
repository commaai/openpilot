#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QWidget>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/signaledit.h"

class BinaryView : public QWidget {
  Q_OBJECT

public:
  BinaryView(QWidget *parent);
  void setMessage(const QString &message_id);
  void updateState();

private:
  QString msg_id;
  QTableWidget *table;
};

class EditMessageDialog : public QDialog {
  Q_OBJECT

public:
  EditMessageDialog(const QString &msg_id, QWidget *parent);

protected:
  void save();

  QString msg_id;
  QLineEdit *name_edit;
  QSpinBox *size_spin;
};

class HistoryLogModel : public QAbstractTableModel {
Q_OBJECT

public:
  HistoryLogModel(QObject *parent) : QAbstractTableModel(parent) {}
  void setMessage(const QString &message_id);
  void clear();
  void updateState();
  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override { return column_count; }
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override { return CAN_MSG_LOG_SIZE; }

private:
  QString msg_id;
  QList<QPair<double, QStringList>> values;
  uint64_t previous_count = 0;
  int column_count = 0;
};

class HistoryLog : public QWidget {
  Q_OBJECT

public:
  HistoryLog(QWidget *parent);
  void setMessage(const QString &message_id);
  void updateState();

private:
  QString msg_id;
  uint64_t previous_count = 0;
  QTableView *table;
  HistoryLogModel *model;
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(QWidget *parent);
  void setMessage(const QString &message_id);

signals:
  void showChart(const QString &msg_id, const QString &sig_name);

private:
  void addSignal();
  void editMsg();
  void updateState();

  QString msg_id;
  QLabel *name_label, *time_label;
  QPushButton *edit_btn;
  QVBoxLayout *signal_edit_layout;
  QWidget *signals_header;
  HistoryLog *history_log;
  BinaryView *binary_view;
};
