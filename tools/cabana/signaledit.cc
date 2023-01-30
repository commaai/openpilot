#include "tools/cabana/signaledit.h"

#include <QDoubleValidator>
#include <QFormLayout>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/util.h"

// SignalForm

SignalForm::SignalForm(QWidget *parent) : QWidget(parent) {
  auto double_validator = new QDoubleValidator(this);
  double_validator->setLocale(QLocale::C); // Match locale of QString::toDouble() instead of system

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QFormLayout *form_layout = new QFormLayout();
  main_layout->addLayout(form_layout);

  name = new QLineEdit();
  name->setValidator(new NameValidator(name));
  form_layout->addRow(tr("Name"), name);

  QHBoxLayout *hl = new QHBoxLayout(this);
  size = new QSpinBox();
  size->setMinimum(1);
  hl->addWidget(size);
  endianness = new QComboBox();
  endianness->addItems({"Little Endianness", "Big Endianness"});
  hl->addWidget(endianness);
  sign = new QComboBox();
  sign->addItems({"Signed", "Unsigned"});
  hl->addWidget(sign);
  form_layout->addRow(tr("Size"), hl);

  offset = new QLineEdit();
  offset->setValidator(double_validator);
  form_layout->addRow(tr("Offset"), offset);
  factor = new QLineEdit();
  factor->setValidator(double_validator);
  form_layout->addRow(tr("Factor"), factor);

  expand_btn = new QToolButton(this);
  expand_btn->setText(tr("more..."));
  main_layout->addWidget(expand_btn, 0, Qt::AlignRight);

  // TODO: parse the following parameters in opendbc
  QWidget *extra_container = new QWidget(this);
  QFormLayout *extra_layout = new QFormLayout(extra_container);
  unit = new QLineEdit();
  extra_layout->addRow(tr("Unit"), unit);
  comment = new QLineEdit();
  extra_layout->addRow(tr("Comment"), comment);
  min_val = new QLineEdit();
  min_val->setValidator(double_validator);
  extra_layout->addRow(tr("Minimum value"), min_val);
  max_val = new QLineEdit();
  max_val->setValidator(double_validator);
  extra_layout->addRow(tr("Maximum value"), max_val);
  val_desc = new QLineEdit();
  extra_layout->addRow(tr("Value descriptions"), val_desc);

  main_layout->addWidget(extra_container);
  extra_container->setVisible(false);

  QObject::connect(name, &QLineEdit::editingFinished, this, &SignalForm::textBoxEditingFinished);
  QObject::connect(factor, &QLineEdit::editingFinished, this, &SignalForm::textBoxEditingFinished);
  QObject::connect(offset, &QLineEdit::editingFinished, this, &SignalForm::textBoxEditingFinished);
  QObject::connect(size, &QSpinBox::editingFinished, this, &SignalForm::changed);
  QObject::connect(sign, SIGNAL(activated(int)), SIGNAL(changed()));
  QObject::connect(endianness, SIGNAL(activated(int)), SIGNAL(changed()));
  QObject::connect(expand_btn, &QToolButton::clicked, [=]() {
    extra_container->setVisible(!extra_container->isVisible());
    expand_btn->setText(extra_container->isVisible() ? tr("less...") : tr("more..."));
  });
}

void SignalForm::textBoxEditingFinished() {
  QLineEdit *edit = qobject_cast<QLineEdit *>(QObject::sender());
  if (edit && edit->isModified()) {
    edit->setModified(false);
    emit changed();
  }
}

// SignalEdit

SignalEdit::SignalEdit(int index, QWidget *parent) : form_idx(index), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(0);

  bg_color = QColor(getColor(form_idx));

  // title bar
  auto title_bar = new QWidget(this);
  title_bar->setFixedHeight(32);
  QHBoxLayout *title_layout = new QHBoxLayout(title_bar);
  title_layout->setContentsMargins(0, 0, 0, 0);
  title_bar->setStyleSheet("QToolButton {width:15px;height:15px;font-size:15px}");
  color_label = new QLabel(this);
  color_label->setFixedWidth(25);
  color_label->setContentsMargins(5, 0, 0, 0);
  title_layout->addWidget(color_label);
  icon = new QLabel(this);
  title_layout->addWidget(icon);
  title = new ElidedLabel(this);
  title->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  title_layout->addWidget(title);

  plot_btn = new QToolButton(this);
  plot_btn->setIcon(bootstrapPixmap("graph-up"));
  plot_btn->setCheckable(true);
  plot_btn->setAutoRaise(true);
  title_layout->addWidget(plot_btn);
  auto remove_btn = new QToolButton(this);
  remove_btn->setAutoRaise(true);
  remove_btn->setIcon(bootstrapPixmap("x"));
  remove_btn->setToolTip(tr("Remove signal"));
  title_layout->addWidget(remove_btn);
  main_layout->addWidget(title_bar);

  // signal form
  form = new SignalForm(this);
  form->setVisible(false);
  main_layout->addWidget(form);

  // bottom line
  QFrame *hline = new QFrame();
  hline->setFrameShape(QFrame::HLine);
  hline->setFrameShadow(QFrame::Sunken);
  main_layout->addWidget(hline);

  QObject::connect(title, &ElidedLabel::clicked, [this]() { emit showFormClicked(sig); });
  QObject::connect(plot_btn, &QToolButton::clicked, [this](bool checked) {
    emit showChart(msg_id, sig, checked, QGuiApplication::keyboardModifiers() & Qt::ShiftModifier);
  });
  QObject::connect(remove_btn, &QToolButton::clicked, [this]() { emit remove(sig); });
  QObject::connect(form, &SignalForm::changed, this, &SignalEdit::saveSignal);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
}

void SignalEdit::setSignal(const QString &message_id, const Signal *signal) {
  sig = signal;
  updateForm(msg_id == message_id && form->isVisible());
  msg_id = message_id;
  color_label->setText(QString::number(form_idx + 1));
  color_label->setStyleSheet(QString("color:black; background-color:%2").arg(bg_color.name()));
  title->setText(sig->name.c_str());
  show();
}

void SignalEdit::saveSignal() {
  Signal s = *sig;
  s.name = form->name->text().toStdString();
  s.size = form->size->text().toInt();
  s.offset = form->offset->text().toDouble();
  s.factor = form->factor->text().toDouble();
  s.is_signed = form->sign->currentIndex() == 0;
  bool little_endian = form->endianness->currentIndex() == 0;
  if (little_endian != s.is_little_endian) {
    int start = std::floor(s.start_bit / 8);
    if (little_endian) {
      int end = std::floor((s.start_bit - s.size + 1) / 8);
      s.start_bit = start == end ? s.start_bit - s.size + 1 : bigEndianStartBitsIndex(s.start_bit);
    } else {
      int end = std::floor((s.start_bit + s.size - 1) / 8);
      s.start_bit = start == end ? s.start_bit + s.size - 1 : bigEndianBitIndex(s.start_bit);
    }
    s.is_little_endian = little_endian;
  }
  if (s.is_little_endian) {
    s.lsb = s.start_bit;
    s.msb = s.start_bit + s.size - 1;
  } else {
    s.lsb = bigEndianStartBitsIndex(bigEndianBitIndex(s.start_bit) + s.size - 1);
    s.msb = s.start_bit;
  }
  if (s != *sig) {
    emit save(this->sig, s);
  }
}

void SignalEdit::setChartOpened(bool opened) {
  plot_btn->setToolTip(opened ? tr("Close Plot") : tr("Show Plot\nSHIFT click to add to previous opened chart"));
  plot_btn->setChecked(opened);
}

void SignalEdit::updateForm(bool visible) {
  if (visible && sig) {
    if (form->name->text() != sig->name.c_str()) {
      form->name->setText(sig->name.c_str());
    }
    form->endianness->setCurrentIndex(sig->is_little_endian ? 0 : 1);
    form->sign->setCurrentIndex(sig->is_signed ? 0 : 1);
    form->factor->setText(QString::number(sig->factor));
    form->offset->setText(QString::number(sig->offset));
    form->size->setValue(sig->size);
  }
  form->setVisible(visible);
  icon->setText(visible ? "▼ " : "> ");
}

void SignalEdit::signalHovered(const Signal *s) {
  auto text_color = sig == s ? "white" : "black";
  auto _bg_color = sig == s ? bg_color.darker(125) : bg_color;  // 4/5x brightness
  color_label->setStyleSheet(QString("color:%1; background-color:%2").arg(text_color).arg(_bg_color.name()));
}

void SignalEdit::enterEvent(QEvent *event) {
  emit highlight(sig);
  QWidget::enterEvent(event);
}

void SignalEdit::leaveEvent(QEvent *event) {
  emit highlight(nullptr);
  QWidget::leaveEvent(event);
}
