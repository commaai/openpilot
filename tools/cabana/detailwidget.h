#pragma once

#include <QScrollArea>
#include <QTabWidget>
#include <QToolBar>
#include <QUndoStack>

#include "tools/cabana/binaryview.h"
#include "tools/cabana/chartswidget.h"
#include "tools/cabana/historylog.h"
#include "tools/cabana/signaledit.h"

class EditMessageDialog : public QDialog {
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
  QUndoStack *undo_stack = nullptr;

private:
  void showFormClicked();
  void updateChartState(const QString &id, const Signal *sig, bool opened);
  void showTabBarContextMenu(const QPoint &pt);
  void addSignal(int start_bit, int size, bool little_endian);
  void resizeSignal(const Signal *sig, int from, int to);
  void saveSignal(const Signal *sig, const Signal &new_sig);
  void removeSignal(const Signal *sig);
  void editMsg();
  void removeMsg();
  void updateState(const QHash<QString, CanData> * msgs = nullptr);

  QString msg_id;
  QLabel *name_label, *time_label, *warning_label;
  QWidget *warning_widget;
  QVBoxLayout *signals_layout;
  QTabBar *tabbar;
  QTabWidget *tab_widget;
  QToolBar *toolbar;
  QAction *remove_msg_act;
  HistoryLog *history_log;
  BinaryView *binary_view;
  QScrollArea *scroll;
  ChartsWidget *charts;
  QList<SignalEdit *> signal_list;
};
