#include "tools/cabana/signaledit.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QVBoxLayout>

// SignalForm

SignalForm::SignalForm(const Signal &sig, QWidget *parent) : QWidget(parent) {
  QFormLayout *form_layout = new QFormLayout(this);

  name = new QLineEdit(sig.name.c_str());
  form_layout->addRow(tr("Name"), name);

  size = new QSpinBox();
  size->setValue(sig.size);
  form_layout->addRow(tr("Size"), size);

  msb = new QSpinBox();
  msb->setValue(sig.msb);
  form_layout->addRow(tr("Most significant bit"), msb);

  endianness = new QComboBox();
  endianness->addItems({"Little", "Big"});
  endianness->setCurrentIndex(sig.is_little_endian ? 0 : 1);
  form_layout->addRow(tr("Endianness"), endianness);

  sign = new QComboBox();
  sign->addItems({"Signed", "Unsigned"});
  sign->setCurrentIndex(sig.is_signed ? 0 : 1);
  form_layout->addRow(tr("sign"), sign);

  factor = new QSpinBox();
  factor->setValue(sig.factor);
  form_layout->addRow(tr("Factor"), factor);

  offset = new QSpinBox();
  offset->setValue(sig.offset);
  form_layout->addRow(tr("Offset"), offset);

  // TODO: parse the following parameters in opendbc
  unit = new QLineEdit();
  form_layout->addRow(tr("Unit"), unit);
  comment = new QLineEdit();
  form_layout->addRow(tr("Comment"), comment);
  min_val = new QSpinBox();
  form_layout->addRow(tr("Minimum value"), min_val);
  max_val = new QSpinBox();
  form_layout->addRow(tr("Maximum value"), max_val);
  val_desc = new QLineEdit();
  form_layout->addRow(tr("Value descriptions"), val_desc);
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
  h->addWidget(remove_btn);
  h->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save"));
  h->addWidget(save_btn);
  v_layout->addLayout(h);

  edit_container->setVisible(false);
  main_layout->addWidget(edit_container);

  QObject::connect(remove_btn, &QPushButton::clicked, this, &SignalEdit::remove);
  QObject::connect(save_btn, &QPushButton::clicked, this, &SignalEdit::save);
  QObject::connect(title, &ElidedLabel::clicked, [=]() {
    edit_container->isVisible() ? edit_container->hide() : edit_container->show();
    icon->setText(edit_container->isVisible() ? "▼" : ">");
  });
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
