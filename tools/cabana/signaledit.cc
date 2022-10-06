#include "tools/cabana/signaledit.h"

#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QVBoxLayout>

// SignalForm

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
    sig.lsb = bigEndianStartBitsIndex(bigEndianBitIndex(sig.start_bit) + sig.size - 1);
    sig.msb = sig.start_bit;
  }
  return (sig.name.empty() || sig.size <= 0) ? std::nullopt : std::optional(sig);
}

// SignalEdit

SignalEdit::SignalEdit(const QString &id, const Signal &sig, const QString &color, QWidget *parent) : id(id), name_(sig.name.c_str()), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // title
  QHBoxLayout *title_layout = new QHBoxLayout();
  QLabel *icon = new QLabel(">");
  icon->setStyleSheet("font-weight:bold");
  title_layout->addWidget(icon);
  title = new ElidedLabel(this);
  title->setText(sig.name.c_str());
  title->setStyleSheet(QString("font-weight:bold; color:%1").arg(color));
  connect(title, &ElidedLabel::clicked, [=]() {
    edit_container->isVisible() ? edit_container->hide() : edit_container->show();
    icon->setText(edit_container->isVisible() ? "▼" : ">");
  });
  title_layout->addWidget(title);
  title_layout->addStretch();
  plot_btn = new QPushButton("📈");
  plot_btn->setStyleSheet("font-size:16px");
  plot_btn->setToolTip(tr("Show Plot"));
  plot_btn->setContentsMargins(5, 5, 5, 5);
  plot_btn->setFixedSize(30, 30);
  QObject::connect(plot_btn, &QPushButton::clicked, [=]() { emit parser->showPlot(id, name_); });
  title_layout->addWidget(plot_btn);
  main_layout->addLayout(title_layout);

  edit_container = new QWidget(this);
  QVBoxLayout *v_layout = new QVBoxLayout(edit_container);
  form = new SignalForm(sig, this);
  v_layout->addWidget(form);

  QHBoxLayout *h = new QHBoxLayout();
  remove_btn = new QPushButton(tr("Remove Signal"));
  QObject::connect(remove_btn, &QPushButton::clicked, this, &SignalEdit::remove);
  h->addWidget(remove_btn);
  h->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save"));
  QObject::connect(save_btn, &QPushButton::clicked, this, &SignalEdit::save);
  h->addWidget(save_btn);
  v_layout->addLayout(h);

  edit_container->setVisible(false);
  main_layout->addWidget(edit_container);
}

void SignalEdit::save() {
  if (auto sig = const_cast<Signal *>(parser->getSig(id, name_))) {
    if (auto s = form->getSignal()) {
      *sig = *s;
      // TODO: reset the chart for sig
    }
  }
}

void SignalEdit::remove() {
  QMessageBox msgbox;
  msgbox.setText(tr("Remove signal"));
  msgbox.setInformativeText(tr("Are you sure you want to remove signal '%1'").arg(name_));
  msgbox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
  msgbox.setDefaultButton(QMessageBox::Cancel);
  if (msgbox.exec()) {
    parser->removeSignal(id, name_);
    deleteLater();
  }
}

// AddSignalDialog

AddSignalDialog::AddSignalDialog(const QString &id, QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Add signal to %1").arg(parser->getMsg(id)->name.c_str()));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  Signal sig = {.name = "untitled"};
  auto form = new SignalForm(sig, this);
  main_layout->addWidget(form);
  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  connect(buttonBox, &QDialogButtonBox::accepted, [=]() {
    if (auto msg = const_cast<Msg *>(parser->getMsg(id))) {
      if (auto signal = form->getSignal()) {
        msg->sigs.push_back(*signal);
      }
    }
    QDialog::accept();
  });
}
