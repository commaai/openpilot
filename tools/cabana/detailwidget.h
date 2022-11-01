#pragma once

#include <QScrollArea>
#include <QTabBar>
#include <QVBoxLayout>

#include "tools/cabana/binaryview.h"
#include "tools/cabana/chartswidget.h"
#include "tools/cabana/historylog.h"
#include "tools/cabana/signaledit.h"

class TitleFrame : public QFrame {
  Q_OBJECT
public:
  TitleFrame(QWidget *parent) : QFrame(parent) {}
  void mouseDoubleClickEvent(QMouseEvent *e) { emit doubleClicked(); }
signals:
  void doubleClicked();
};

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
  DetailWidget(ChartsWidget *charts, QWidget *parent);
  void setMessage(const QString &message_id);
  void dbcMsgChanged(int show_form_idx = -1);

signals:
  void binaryViewMoved(bool in);

private:
  void updateChartState(const QString &id, const Signal *sig, bool opened);
  void showTabBarContextMenu(const QPoint &pt);
  void addSignal(int start_bit, int to);
  void resizeSignal(const Signal *sig, int from, int to);
  void saveSignal(const Signal *sig, const Signal &new_sig);
  void removeSignal(const Signal *sig);
  void editMsg();
  void showForm();
  void updateState();
  void moveBinaryView();

  QString msg_id;
  QLabel *name_label, *time_label, *warning_label;
  QWidget *warning_widget;
  QPushButton *edit_btn;
  QWidget *signals_container;
  QTabBar *tabbar;
  QHBoxLayout *main_layout;
  QVBoxLayout *right_column;
  bool binview_in_left_col = false;
  QWidget *binary_view_container;
  QPushButton *split_btn;
  HistoryLog *history_log;
  BinaryView *binary_view;
  ScrollArea *scroll;
  ChartsWidget *charts;
};
