#pragma once

#include <QScrollArea>
#include <QTabBar>

#include "tools/cabana/binaryview.h"
#include "tools/cabana/historylog.h"
#include "tools/cabana/signaledit.h"

class EditMessageDialog : public QDialog {
  Q_OBJECT

public:
  EditMessageDialog(const QString &msg_id, const QString &title, int size, QWidget *parent);

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
  void dbcMsgChanged();

signals:
  void showChart(const QString &msg_id, const Signal *sig);
  void removeChart(const Signal *sig);

private:
  void addSignal(int start_bit, int size);
  void saveSignal();
  void removeSignal();
  void editMsg();
  void showForm();
  void updateState();

  QString msg_id;
  QLabel *name_label, *time_label, *warning_label;
  QWidget *warning_widget;
  QPushButton *edit_btn;
  QWidget *signals_container;
  QTabBar *tabbar;
  QStringList messages;
  HistoryLog *history_log;
  BinaryView *binary_view;
  ScrollArea *scroll;
};
