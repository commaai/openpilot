#pragma once

#include <QScrollArea>
#include <QTabBar>
#include <QToolBar>

#include "tools/cabana/binaryview.h"
#include "tools/cabana/chartswidget.h"
#include "tools/cabana/historylog.h"
#include "tools/cabana/signaledit.h"

class EditMessageDialog : public QDialog {
  Q_OBJECT

public:
  EditMessageDialog(const QString &msg_id, const QString &title, int size, QWidget *parent);

  QLineEdit *name_edit;
  QSpinBox *size_spin;
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(ChartsWidget *charts, QWidget *parent);
  void setMessage(const QString &message_id);
  void dbcMsgChanged(int show_form_idx = -1);

private:
  void updateChartState(const QString &id, const Signal *sig, bool opened);
  void showTabBarContextMenu(const QPoint &pt);
  void addSignal(int start_bit, int size, bool little_endian);
  void resizeSignal(const Signal *sig, int from, int to);
  void saveSignal(const Signal *sig, const Signal &new_sig);
  void removeSignal(const Signal *sig);
  void editMsg();
  void removeMsg();
  void showForm();
  void updateState();

  QString msg_id;
  QLabel *name_label, *time_label, *warning_label;
  QWidget *warning_widget;
  QWidget *signals_container;
  QTabBar *tabbar;
  QToolBar *toolbar;
  QAction *remove_msg_act;
  HistoryLog *history_log;
  BinaryView *binary_view;
  QScrollArea *scroll;
  ChartsWidget *charts;
  QList<SignalEdit *> signal_list;
};
