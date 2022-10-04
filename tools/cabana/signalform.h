#pragma once

#include <optional>

#include <QComboBox>
#include <QLineEdit>
#include <QSpinBox>

#include "tools/cabana/parser.h"

class SignalForm : public QWidget {
 Q_OBJECT

public:
  SignalForm(const Signal &sig, QWidget *parent);
  std::optional<Signal> getSignal();

private:
  QLineEdit *name, *unit, *comment, *val_desc;
  QSpinBox *size, *msb, *lsb, *factor, *offset, *min_val, *max_val;
  QComboBox *sign, *endianness;
};
