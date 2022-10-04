#include "tools/cabana/signalform.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>

SignalForm::SignalForm(const Signal &sig, QWidget *parent) : QWidget(parent) {
  QVBoxLayout *v_layout = new QVBoxLayout(this);

  QHBoxLayout *h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Name")));
  name = new QLineEdit(sig.name.c_str());
  h->addWidget(name);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Size")));
  size = new QSpinBox();
  size->setValue(sig.size);
  h->addWidget(size);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Most significant bit")));
  msb = new QSpinBox();
  msb->setValue(sig.msb);
  h->addWidget(msb);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Endianness")));
  endianness = new QComboBox();
  endianness->addItems({"Little", "Big"});
  endianness->setCurrentIndex(sig.is_little_endian ? 0 : 1);
  h->addWidget(endianness);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("sign")));
  sign = new QComboBox();
  sign->addItems({"Signed", "Unsigned"});
  sign->setCurrentIndex(sig.is_signed ? 0 : 1);
  h->addWidget(sign);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Factor")));
  factor = new QSpinBox();
  factor->setValue(sig.factor);
  h->addWidget(factor);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Offset")));
  offset = new QSpinBox();
  offset->setValue(sig.offset);
  h->addWidget(offset);
  v_layout->addLayout(h);

  // TODO: parse the following parameters in opendbc
  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Unit")));
  unit = new QLineEdit();
  h->addWidget(unit);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Comment")));
  comment = new QLineEdit();
  h->addWidget(comment);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Minimum value")));
  min_val = new QSpinBox();
  h->addWidget(min_val);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Maximum value")));
  max_val = new QSpinBox();
  h->addWidget(max_val);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Value descriptions")));
  val_desc = new QLineEdit();
  h->addWidget(val_desc);
  v_layout->addLayout(h);
}

std::optional<Signal> SignalForm::getSignal() {
  Signal sig = {};
  sig.name = name->text().toStdString();
  sig.size = size->text().toInt();
  sig.offset = offset->text().toDouble();
  sig.factor = factor->text().toDouble();
  sig.msb = msb->text().toInt();
  sig.is_signed = sign->currentIndex() == 0;
  sig.is_little_endian = endianness->currentIndex() == 0;
  if (sig.is_little_endian) {
    sig.lsb = sig.start_bit;
    sig.msb = sig.start_bit + sig.size - 1;
  } else {
    sig.lsb = BIG_ENDIAN_START_BITS[bigEndianBitIndex(sig.start_bit) + sig.size - 1];
    sig.msb = sig.start_bit;
  }

  return (sig.name.empty() || sig.size <= 0) ? std::nullopt : std::optional(sig);
}
