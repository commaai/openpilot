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

class HistoryLog : public QWidget {
  Q_OBJECT

public:
  HistoryLog(QWidget *parent);
  void clear();
  void updateState();

private:
  QLabel *labels[LOG_SIZE] = {};
};

class BinaryView : public QWidget {
  Q_OBJECT

public:
  BinaryView(QWidget *parent);
  void setMsg(const CanData *can_data);
  void setData(const QByteArray &binary);

  QTableWidget *table;
};

class EditMessageDialog : public QDialog {
  Q_OBJECT

public:
  EditMessageDialog(const QString &id, QWidget *parent);

protected:
  void save();

  QLineEdit *name_edit;
  QSpinBox *size_spin;
  QString id;
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(QWidget *parent);
  void setMsg(const CanData *c);

private:
  void updateState();
  void addSignal();
  void editMsg();

  const CanData *can_data = nullptr;
  QLabel *name_label, *time_label;
  QPushButton *edit_btn, *add_sig_btn;
  QVBoxLayout *signal_edit_layout;
  HistoryLog *history_log;
  BinaryView *binary_view;
};
