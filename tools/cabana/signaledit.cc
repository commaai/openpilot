#include "tools/cabana/signaledit.h"

#include <QDoubleValidator>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QTimer>
#include <QToolBar>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/util.h"

// SignalForm

SignalForm::SignalForm(QWidget *parent) : QWidget(parent) {
  QFormLayout *form_layout = new QFormLayout(this);

  name = new QLineEdit();
  form_layout->addRow(tr("Name"), name);

  size = new QSpinBox();
  size->setMinimum(1);
  form_layout->addRow(tr("Size"), size);

  endianness = new QComboBox();
  endianness->addItems({"Little", "Big"});
  form_layout->addRow(tr("Endianness"), endianness);

  form_layout->addRow(tr("lsb"), lsb = new QLabel());
  form_layout->addRow(tr("msb"), msb = new QLabel());

  sign = new QComboBox();
  sign->addItems({"Signed", "Unsigned"});
  form_layout->addRow(tr("sign"), sign);

  auto double_validator = new QDoubleValidator(this);

  factor = new QLineEdit();
  factor->setValidator(double_validator);
  form_layout->addRow(tr("Factor"), factor);

  offset = new QLineEdit();
  offset->setValidator(double_validator);
  form_layout->addRow(tr("Offset"), offset);

  // TODO: parse the following parameters in opendbc
  unit = new QLineEdit();
  form_layout->addRow(tr("Unit"), unit);
  comment = new QLineEdit();
  form_layout->addRow(tr("Comment"), comment);
  min_val = new QLineEdit();
  min_val->setValidator(double_validator);
  form_layout->addRow(tr("Minimum value"), min_val);
  max_val = new QLineEdit();
  max_val->setValidator(double_validator);
  form_layout->addRow(tr("Maximum value"), max_val);
  val_desc = new QLineEdit();
  form_layout->addRow(tr("Value descriptions"), val_desc);

  QObject::connect(name, &QLineEdit::textEdited, this, &SignalForm::changed);
  QObject::connect(factor, &QLineEdit::textEdited, this, &SignalForm::changed);
  QObject::connect(offset, &QLineEdit::textEdited, this, &SignalForm::changed);
  QObject::connect(sign, SIGNAL(activated(int)), SIGNAL(changed()));
  QObject::connect(endianness, SIGNAL(activated(int)), SIGNAL(changed()));
  QObject::connect(size, SIGNAL(valueChanged(int)), SIGNAL(changed()));
}

// SignalEdit

SignalEdit::SignalEdit(int index, QWidget *parent) : form_idx(index), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // title bar
  auto toolbar = new QToolBar(this);
  toolbar->setStyleSheet("QToolButton {width:15px;height:15px;font-size:15px}");
  icon = new QLabel();
  toolbar->addWidget(icon);
  title = new ElidedLabel(this);
  title->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  title->setStyleSheet(QString("font-weight:bold; color:%1").arg(getColor(index)));
  toolbar->addWidget(title);
  plot_btn = toolbar->addAction("", [this]() { emit showChart(msg_id, sig, !chart_opened); });
  auto seek_btn = toolbar->addAction(QIcon::fromTheme("edit-find"), "", [this]() { SignalFindDlg(msg_id, sig, this).exec(); });
  seek_btn->setToolTip(tr("Find signal values"));
  auto remove_btn = toolbar->addAction("x", [this]() { emit remove(sig); });
  remove_btn->setToolTip(tr("Remove signal"));
  main_layout->addWidget(toolbar);

  // signal form
  form = new SignalForm(this);
  form->setVisible(false);
  main_layout->addWidget(form);

  // bottom line
  QFrame *hline = new QFrame();
  hline->setFrameShape(QFrame::HLine);
  hline->setFrameShadow(QFrame::Sunken);
  main_layout->addWidget(hline);

  QObject::connect(form, &SignalForm::changed, this, &SignalEdit::saveSignal);
  QObject::connect(title, &ElidedLabel::clicked, this, &SignalEdit::showFormClicked);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
}

void SignalEdit::setSignal(const QString &message_id, const Signal *signal) {
  sig = signal;
  updateForm(msg_id == message_id && form->isVisible());
  msg_id = message_id;
  title->setText(QString("%1. %2").arg(form_idx + 1).arg(sig->name.c_str()));
  show();
}

void SignalEdit::saveSignal() {
  if (!sig || !form->changed_by_user) return;

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
  if (s != *sig)
    emit save(this->sig, s);
}

void SignalEdit::setChartOpened(bool opened) {
  plot_btn->setText(opened ? "☒" : "📈");
  plot_btn->setToolTip(opened ? tr("Close Plot") : tr("Show Plot"));
  chart_opened = opened;
}

void SignalEdit::updateForm(bool visible) {
  if (visible && sig) {
    form->changed_by_user = false;
    form->name->setText(sig->name.c_str());
    form->endianness->setCurrentIndex(sig->is_little_endian ? 0 : 1);
    form->sign->setCurrentIndex(sig->is_signed ? 0 : 1);
    form->factor->setText(QString::number(sig->factor));
    form->offset->setText(QString::number(sig->offset));
    form->msb->setText(QString::number(sig->msb));
    form->lsb->setText(QString::number(sig->lsb));
    form->size->setValue(sig->size);
    form->changed_by_user = true;
  }
  form->setVisible(visible);
  icon->setText(visible ? "▼ " : "> ");
}

void SignalEdit::showFormClicked() {
  parentWidget()->setUpdatesEnabled(false);
  for (auto &edit : parentWidget()->findChildren<SignalEdit*>())
    edit->updateForm(edit == this && !form->isVisible());
  QTimer::singleShot(1, [this]() { parentWidget()->setUpdatesEnabled(true); });
}

void SignalEdit::signalHovered(const Signal *s) {
  auto color = sig == s ? hoverColor(getColor(form_idx)) : QColor(getColor(form_idx));
  title->setStyleSheet(QString("font-weight:bold; color:%1").arg(color.name()));
}

void SignalEdit::hideEvent(QHideEvent *event) {
  msg_id = "";
  sig = nullptr;
  updateForm(false);
  QWidget::hideEvent(event);
}

void SignalEdit::enterEvent(QEvent *event) {
  emit highlight(sig);
  QWidget::enterEvent(event);
}

void SignalEdit::leaveEvent(QEvent *event) {
  emit highlight(nullptr);
  QWidget::leaveEvent(event);
}

// SignalFindDlg

SignalFindDlg::SignalFindDlg(const QString &id, const Signal *signal, QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Find signal values"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *h = new QHBoxLayout();
  h->addWidget(new QLabel(signal->name.c_str()));
  QComboBox *comp_box = new QComboBox();
  comp_box->addItems({">", "=", "<"});
  h->addWidget(comp_box);
  QLineEdit *value_edit = new QLineEdit("0", this);
  value_edit->setValidator(new QDoubleValidator(-500000, 500000, 6, this));
  h->addWidget(value_edit, 1);
  QPushButton *search_btn = new QPushButton(tr("Find"), this);
  h->addWidget(search_btn);
  main_layout->addLayout(h);

  QWidget *container = new QWidget(this);
  QVBoxLayout *signals_layout = new QVBoxLayout(container);
  QScrollArea *scroll = new QScrollArea(this);
  scroll->setWidget(container);
  scroll->setWidgetResizable(true);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(scroll);

  QObject::connect(search_btn, &QPushButton::clicked, [=]() {
    clearLayout(signals_layout);

    CANMessages::FindFlags comp = CANMessages::EQ;
    if (comp_box->currentIndex() == 0) {
      comp = CANMessages::GT;
    } else if (comp_box->currentIndex() == 2) {
      comp = CANMessages::LT;
    }
    double value = value_edit->text().toDouble();

    const int limit_results = 50;
    auto values = can->findSignalValues(id, signal, value, comp, limit_results);
    for (auto &v : values) {
      QHBoxLayout *item_layout = new QHBoxLayout();
      item_layout->addWidget(new QLabel(QString::number(v.x(), 'f', 2)));
      item_layout->addWidget(new QLabel(QString::number(v.y())));
      item_layout->addStretch(1);

      QPushButton *goto_btn = new QPushButton(tr("Goto"), this);
      QObject::connect(goto_btn, &QPushButton::clicked, [sec = v.x()]() { can->seekTo(sec); });
      item_layout->addWidget(goto_btn);
      signals_layout->addLayout(item_layout);
    }
    if (values.size() == limit_results) {
      QFrame *hline = new QFrame();
      hline->setFrameShape(QFrame::HLine);
      hline->setFrameShadow(QFrame::Sunken);
      signals_layout->addWidget(hline);
      QLabel *info = new QLabel(tr("Only display the first %1 results").arg(limit_results));
      info->setAlignment(Qt::AlignCenter);
      signals_layout->addWidget(info);
    }
    if (values.size() * 30 > container->height()) {
      scroll->setFixedHeight(std::min(values.size() * 30, 300));
    }
  });
}
