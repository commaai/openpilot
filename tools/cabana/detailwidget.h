#pragma once

#include <QDialogButtonBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTextEdit>
#include <set>

#include "selfdrive/ui/qt/widgets/controls.h"
#include "tools/cabana/binaryview.h"
#include "tools/cabana/chart/chartswidget.h"
#include "tools/cabana/historylog.h"
#include "tools/cabana/signalview.h"

class EditMessageDialog : public QDialog {
public:
  EditMessageDialog(const MessageId &msg_id, const QString &title, int size, QWidget *parent);
  void validateName(const QString &text);

  MessageId msg_id;
  QString original_name;
  QDialogButtonBox *btn_box;
  QLineEdit *name_edit;
  QLineEdit *node;
  QTextEdit *comment_edit;
  QLabel *error_label;
  QSpinBox *size_spin;
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(ChartsWidget *charts, QWidget *parent);
  void setMessage(const MessageId &message_id);
  void refresh();

private:
  void showTabBarContextMenu(const QPoint &pt);
  void editMsg();
  void removeMsg();
  void updateState(const std::set<MessageId> *msgs = nullptr);

  MessageId msg_id;
  QLabel *warning_icon, *warning_label;
  ElidedLabel *name_label;
  QWidget *warning_widget;
  TabBar *tabbar;
  QTabWidget *tab_widget;
  QToolButton *remove_btn;
  LogsWidget *history_log;
  BinaryView *binary_view;
  SignalView *signal_view;
  ChartsWidget *charts;
  QSplitter *splitter;
};

class CenterWidget : public QWidget {
  Q_OBJECT
public:
  CenterWidget(QWidget *parent);
  void setMessage(const MessageId &msg_id);
  void clear();

private:
  QWidget *createWelcomeWidget();
  DetailWidget *detail_widget = nullptr;
  QWidget *welcome_widget = nullptr;
};
