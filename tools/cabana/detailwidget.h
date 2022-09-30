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

class BinaryView : public QWidget {
  Q_OBJECT

 public:
  BinaryView(QWidget *parent);
  void setData(const std::string &binary);

  QTableWidget *table;
};

class SignalEdit : public QWidget {
  Q_OBJECT

 public:
  SignalEdit(QWidget *parent);
  void setSig(const Signal &sig);

  QLineEdit *name, *unit, *comment, *val_desc;
  QSpinBox *size, *significant_bit, *factor, *offset, *min_val, *max_val;
  QComboBox *sign, *endianness;
  QPushButton *remove_btn;
};

class DetailWidget : public QWidget {
  Q_OBJECT
 public:
  DetailWidget(QWidget *parent);
  void setItem(uint32_t addr);
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;

 public slots:
  void updateState();

 protected:
  QLabel *name_label = nullptr;
  QVBoxLayout *signal_edit_layout;
  uint32_t address = 0;
  BinaryView *binary_view;
  std::vector<SignalEdit *> signal_edit;
};
