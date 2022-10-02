#pragma once

#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QWidget>

#include "opendbc/can/common.h"
#include "opendbc/can/common_dbc.h"
#include "tools/cabana/parser.h"

#include "selfdrive/ui/qt/widgets/controls.h"

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
  void setData(const QByteArray &binary);

  QTableWidget *table;
};

class SignalEdit : public QWidget {
  Q_OBJECT

 public:
  SignalEdit(QWidget *parent);
  void setSig(uint32_t address, const Signal &sig);

 signals:
  void showPlot(uint32_t address, const QString &name);

 protected:
  uint32_t address_ = 0;
  QString name_;
  ElidedLabel *title;
  QWidget *edit_container;
  QLineEdit *name, *unit, *comment, *val_desc;
  QSpinBox *size, *msb, *lsb, *factor, *offset, *min_val, *max_val;
  QComboBox *sign, *endianness;
  QPushButton *remove_btn;
};

class DetailWidget : public QWidget {
  Q_OBJECT
 public:
  DetailWidget(QWidget *parent);
  void setItem(uint32_t addr);

 public slots:
  void updateState();

 protected:
  QLabel *name_label = nullptr;
  QVBoxLayout *signal_edit_layout;
  Signal *sig = nullptr;
  MessagesView *messages_view;
  uint32_t address = 0;
  BinaryView *binary_view;
  std::vector<SignalEdit *> signal_edit;
};
