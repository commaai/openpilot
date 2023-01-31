#pragma once

#include <QSplitter>
#include <QStackedLayout>
#include <QTabWidget>
#include <QToolBar>

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

class WelcomeWidget : public QWidget {
public:
  WelcomeWidget(QWidget *parent);
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(ChartsWidget *charts, QWidget *parent);
  void setMessage(const QString &message_id);
  void refresh();
  QSize minimumSizeHint() const override { return binary_view->minimumSizeHint(); }

private:
  void showTabBarContextMenu(const QPoint &pt);
  void editMsg();
  void removeMsg();
  void updateState(const QHash<QString, CanData> * msgs = nullptr);

  QString msg_id;
  QLabel *name_label, *time_label, *warning_icon, *warning_label;
  QWidget *warning_widget;
  QTabBar *tabbar;
  QTabWidget *tab_widget;
  QAction *remove_msg_act;
  LogsWidget *history_log;
  BinaryView *binary_view;
  SignalView *signal_view;
  ChartsWidget *charts;
  QSplitter *splitter;
  QStackedLayout *stacked_layout;
};
