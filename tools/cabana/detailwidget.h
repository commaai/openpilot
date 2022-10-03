#pragma once
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QWidget>
#include <optional>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "tools/cabana/parser.h"

class SignalForm : public QWidget {
  Q_OBJECT

 public:
  SignalForm(const Signal &sig, QWidget *parent);
  std::optional<Signal> getSignal();
  QLineEdit *name, *unit, *comment, *val_desc;
  QSpinBox *size, *msb, *lsb, *factor, *offset, *min_val, *max_val;
  QComboBox *sign, *endianness;
};

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

class SignalEdit : public QWidget {
  Q_OBJECT

 public:
  SignalEdit(const QString &id, const Signal &sig, int idx, QWidget *parent);
  void save();

signals:
  void removed();
 protected:
  void remove();
  QString id;
  QString name_;
  ElidedLabel *title;
  SignalForm *form;
  QWidget *edit_container;
  QPushButton *remove_btn;
};

class DetailWidget : public QWidget {
  Q_OBJECT
 public:
  DetailWidget(QWidget *parent);
  void setMsg(const QString &id);

 public slots:
  void updateState();

 protected:
  QLabel *name_label = nullptr;
  QPushButton *edit_btn, *add_sig_btn;
  QVBoxLayout *signal_edit_layout;
  Signal *sig = nullptr;
  MessagesView *messages_view;
  QString msg_id;
  BinaryView *binary_view;
  std::vector<SignalEdit *> signal_edit;
};

class EditMessageDialog : public QDialog {
  Q_OBJECT

 public:
  EditMessageDialog(const QString &id, QWidget *parent);
};

class AddSignalDialog : public QDialog {
  Q_OBJECT

 public:
  AddSignalDialog(const QString &id, QWidget *parent);
};
