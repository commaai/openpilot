#pragma once

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QWidget>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/historylog.h"
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

class ScrollArea : public QScrollArea {
  Q_OBJECT

public:
  ScrollArea(QWidget *parent) : QScrollArea(parent) {}
  bool eventFilter(QObject *obj, QEvent *ev) override;
  void setWidget(QWidget *w);
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(QWidget *parent);
  void setMessage(const QString &message_id);

signals:
  void showChart(const QString &msg_id, const QString &sig_name);

private slots:
  void showForm();

private:
  void addSignal();
  void editMsg();
  void updateState();

  QString msg_id;
  QLabel *name_label, *time_label;
  QPushButton *edit_btn;
  QVBoxLayout *signal_edit_layout;
  QWidget *signals_header;
  QList<SignalEdit *> signal_forms;
  HistoryLog *history_log;
  BinaryView *binary_view;
  ScrollArea *scroll;
};
