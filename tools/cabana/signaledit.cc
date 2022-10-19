#include "tools/cabana/signaledit.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QRadioButton>
#include <QScrollArea>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/util.h"

// SignalForm

SignalForm::SignalForm(const Signal &sig, QWidget *parent) : start_bit(sig.start_bit), QWidget(parent) {
  QFormLayout *form_layout = new QFormLayout(this);

  name = new QLineEdit(sig.name.c_str());
  form_layout->addRow(tr("Name"), name);

  size = new QSpinBox();
  size->setMinimum(1);
  size->setValue(sig.size);
  form_layout->addRow(tr("Size"), size);

  endianness = new QComboBox();
  endianness->addItems({"Little", "Big"});
  endianness->setCurrentIndex(sig.is_little_endian ? 0 : 1);
  form_layout->addRow(tr("Endianness"), endianness);

  form_layout->addRow(tr("lsb"), new QLabel(QString::number(sig.lsb)));
  form_layout->addRow(tr("msb"), new QLabel(QString::number(sig.msb)));

  sign = new QComboBox();
  sign->addItems({"Signed", "Unsigned"});
  sign->setCurrentIndex(sig.is_signed ? 0 : 1);
  form_layout->addRow(tr("sign"), sign);

  factor = new QDoubleSpinBox();
  factor->setDecimals(3);
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
  min_val = new QDoubleSpinBox();
  factor->setDecimals(3);
  form_layout->addRow(tr("Minimum value"), min_val);
  max_val = new QDoubleSpinBox();
  factor->setDecimals(3);
  form_layout->addRow(tr("Maximum value"), max_val);
  val_desc = new QLineEdit();
  form_layout->addRow(tr("Value descriptions"), val_desc);
}

Signal SignalForm::getSignal() {
  // TODO: Check if the size is valid, and no duplicate name
  Signal sig = {};
  sig.start_bit = start_bit;
  sig.name = name->text().toStdString();
  sig.size = size->text().toInt();
  sig.offset = offset->text().toDouble();
  sig.factor = factor->text().toDouble();
  sig.is_signed = sign->currentIndex() == 0;
  sig.is_little_endian = endianness->currentIndex() == 0;
  if (sig.is_little_endian) {
    sig.lsb = sig.start_bit;
    sig.msb = sig.start_bit + sig.size - 1;
  } else {
    sig.lsb = bigEndianStartBitsIndex(bigEndianBitIndex(sig.start_bit) + sig.size - 1);
    sig.msb = sig.start_bit;
  }
  return sig;
}

// SignalEdit

SignalEdit::SignalEdit(int index, const QString &msg_id, const Signal &sig, QWidget *parent)
    : sig_name(sig.name.c_str()), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // title bar
  QHBoxLayout *title_layout = new QHBoxLayout();
  icon = new QLabel(">");
  icon->setStyleSheet("font-weight:bold");
  title_layout->addWidget(icon);
  title = new ElidedLabel(this);
  title->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
  title->setText(QString("%1. %2").arg(index + 1).arg(sig_name));
  title->setStyleSheet(QString("font-weight:bold; color:%1").arg(getColor(index)));
  title_layout->addWidget(title, 1);

  QPushButton *seek_btn = new QPushButton("⌕");
  seek_btn->setStyleSheet("font-weight:bold;font-size:20px");
  seek_btn->setToolTip(tr("Find signal values"));
  seek_btn->setFixedSize(20, 20);
  title_layout->addWidget(seek_btn);

  QPushButton *plot_btn = new QPushButton("📈");
  plot_btn->setToolTip(tr("Show Plot"));
  plot_btn->setFixedSize(20, 20);
  QObject::connect(plot_btn, &QPushButton::clicked, this, &SignalEdit::showChart);
  title_layout->addWidget(plot_btn);
  main_layout->addLayout(title_layout);

  // signal form
  form_container = new QWidget(this);
  QVBoxLayout *v_layout = new QVBoxLayout(form_container);
  form = new SignalForm(sig, this);
  v_layout->addWidget(form);

  QHBoxLayout *h = new QHBoxLayout();
  QPushButton *remove_btn = new QPushButton(tr("Remove Signal"));
  h->addWidget(remove_btn);
  h->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save"));
  h->addWidget(save_btn);
  v_layout->addLayout(h);

  form_container->setVisible(false);
  main_layout->addWidget(form_container);

  // bottom line
  QFrame *hline = new QFrame();
  hline->setFrameShape(QFrame::HLine);
  hline->setFrameShadow(QFrame::Sunken);
  main_layout->addWidget(hline);

  QObject::connect(remove_btn, &QPushButton::clicked, this, &SignalEdit::remove);
  QObject::connect(title, &ElidedLabel::clicked, this, &SignalEdit::showFormClicked);
  QObject::connect(save_btn, &QPushButton::clicked, [=]() {
    QString new_name = form->getSignal().name.c_str();
    title->setText(QString("%1. %2").arg(index + 1).arg(new_name));
    emit save();
    sig_name = new_name;
  });
  QObject::connect(seek_btn, &QPushButton::clicked, [this, msg_id, s = &sig]() {
    SignalFindDlg dlg(msg_id, s, this);
    dlg.exec();
  });
}

void SignalEdit::setFormVisible(bool visible) {
  form_container->setVisible(visible);
  icon->setText(visible ? "▼" : ">");
}

// AddSignalDialog

AddSignalDialog::AddSignalDialog(const QString &id, int start_bit, int size, QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Add signal to %1").arg(dbc()->msg(id)->name.c_str()));
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  Signal sig = {
    .name = "untitled",
    .start_bit = bigEndianBitIndex(start_bit),
    .is_little_endian = false,
    .size = size,
  };
  form = new SignalForm(sig, this);
  main_layout->addWidget(form);
  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);
  setFixedWidth(parent->width() * 0.9);

  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
}

SignalFindDlg::SignalFindDlg(const QString &id, const Signal *signal, QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Find signal values"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *h = new QHBoxLayout();
  h->addWidget(new QLabel(signal->name.c_str()));
  QComboBox *comp_box = new QComboBox();
  comp_box->addItems({">", "=", "<"});
  h->addWidget(comp_box);
  QLineEdit *value_edit = new QLineEdit("0", this);
  value_edit->setValidator( new QDoubleValidator(-500000, 500000, 6, this) );
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
