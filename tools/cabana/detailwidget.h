#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QWidget>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/cabana/parser.h"
#include "tools/cabana/signaledit.h"

class BinaryView : public QWidget {
  Q_OBJECT

public:
  BinaryView(QWidget *parent);
  void updateDBCMsg(const QString &message_id);
  void setData(const QByteArray &binary);

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


class HistoryLog : public QWidget {
  Q_OBJECT

public:
  HistoryLog(QWidget *parent);
  void updateDBCMsg(const QString &message_id);
  void updateState();

private:
  QString msg_id;
  uint64_t previous_count = 0;
  QTableWidget *table;
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(QWidget *parent);
  void updateDBCMsg(const QString &message_id);

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
