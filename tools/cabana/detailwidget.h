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

class MessagesView : public QWidget {
  Q_OBJECT

public:
  MessagesView(QWidget *parent);
  void setMessages(const std::list<CanData> &data);

  std::vector<QLabel *> messages;
  QVBoxLayout *message_layout;
};

class BinaryView : public QWidget {
  Q_OBJECT

public:
  BinaryView(QWidget *parent);
  void setMsg(const QString &id);
  void setData(const QByteArray &binary);

  QTableWidget *table;
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  DetailWidget(QWidget *parent);
  void setMsg(const QString &id);

private:
  void updateState();
  void addSignal();
  void editMsg();

  QString msg_id;
  QLabel *name_label, *time_label;
  QPushButton *edit_btn, *add_sig_btn;
  QVBoxLayout *signal_edit_layout;
  MessagesView *messages_view;
  BinaryView *binary_view;
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
