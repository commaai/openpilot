#pragma once

#include <QScrollArea>
#include <QTableWidget>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/historylog.h"
#include "tools/cabana/signaledit.h"

class BinarySelectionModel : public QItemSelectionModel {
public:
  BinarySelectionModel(QAbstractItemModel *model = nullptr) : QItemSelectionModel(model) {}
  void select(const QItemSelection &selection, QItemSelectionModel::SelectionFlags command) override;
};

class BinaryView : public QTableWidget {
  Q_OBJECT
public:
  BinaryView(QWidget *parent = nullptr);
  void mouseReleaseEvent(QMouseEvent *event) override;
  void setMessage(const QString &message_id);
  void updateState();
signals:
  void cellsSelected(int start_bit, int size);

private:
  QString msg_id;
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
  QLabel *name_label, *time_label;
  QPushButton *edit_btn;
  QWidget *signals_container;
  HistoryLog *history_log;
  BinaryView *binary_view;
  ScrollArea *scroll;
};
