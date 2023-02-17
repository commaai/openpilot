#pragma once

#include <QDialogButtonBox>
#include <QSplitter>
#include <QStackedLayout>
#include <QTabWidget>
#include <QToolBar>

#include "selfdrive/ui/qt/widgets/controls.h"
#include "tools/cabana/binaryview.h"
#include "tools/cabana/chartswidget.h"
#include "tools/cabana/historylog.h"
#include "tools/cabana/signaledit.h"

class EditMessageDialog : public QDialog {
public:
  EditMessageDialog(const MessageId &msg_id, const QString &title, int size, QWidget *parent);
  void validateName(const QString &text);

  QString original_name;
  QDialogButtonBox *btn_box;
  QLineEdit *name_edit;
  QLabel *error_label;
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
  void setMessage(const MessageId &message_id);
  void refresh();
  void removeAll();
  QSize minimumSizeHint() const override { return binary_view->minimumSizeHint(); }

private:
  void showTabBarContextMenu(const QPoint &pt);
  void editMsg();
  void removeMsg();
  void updateState(const QHash<MessageId, CanData> * msgs = nullptr);

  std::optional<MessageId> msg_id;
  QLabel *time_label, *warning_icon, *warning_label;
  ElidedLabel *name_label;
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
