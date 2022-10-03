#include "tools/cabana/detailwidget.h"

#include <QDebug>
#include <QHeaderView>
#include <QMessageBox>
#include <QTimer>
#include <QVBoxLayout>
#include <bitset>

#include "selfdrive/ui/qt/widgets/scrollview.h"

const QString SIGNAL_COLORS[] = {"#9FE2BF", "#40E0D0", "#6495ED", "#CCCCFF", "#FF7F50", "#FFBF00"};

static QVector<int> BIG_ENDIAN_START_BITS = []() {
  QVector<int> ret;
  for (int i = 0; i < 64; i++) {
    for (int j = 7; j >= 0; j--) {
      ret.push_back(j + i * 8);
    }
  }
  return ret;
}();

static int bigEndianBitIndex(int index) {
  // TODO: Add a helper function in dbc.h
  return BIG_ENDIAN_START_BITS.indexOf(index);
}

DetailWidget::DetailWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QLabel *title = new QLabel(tr("SELECTED MESSAGE:"), this);
  main_layout->addWidget(title);

  QHBoxLayout *name_layout = new QHBoxLayout();
  name_label = new QLabel(this);
  name_label->setStyleSheet("font-weight:bold;");
  name_layout->addWidget(name_label);
  name_layout->addStretch();
  edit_btn = new QPushButton(tr("Edit"), this);
  edit_btn->setVisible(false);
  QObject::connect(edit_btn, &QPushButton::clicked, [=]() {
    EditMessageDialog dlg(msg_id, this);
    int ret = dlg.exec();
    if (ret) {
      setMsg(msg_id);
    }
  });
  name_layout->addWidget(edit_btn);
  main_layout->addLayout(name_layout);

  binary_view = new BinaryView(this);
  main_layout->addWidget(binary_view);

  QHBoxLayout *signals_layout = new QHBoxLayout();
  signals_layout->addWidget(new QLabel(tr("Signals")));
  signals_layout->addStretch();
  add_sig_btn = new QPushButton(tr("Add signal"), this);
  add_sig_btn->setVisible(false);
  QObject::connect(add_sig_btn, &QPushButton::clicked, [=]() {
    AddSignalDialog dlg(msg_id, this);
    int ret = dlg.exec();
    if (ret) {
      setMsg(msg_id);
    }
  });
  signals_layout->addWidget(add_sig_btn);
  main_layout->addLayout(signals_layout);

  QWidget *container = new QWidget(this);
  QVBoxLayout *container_layout = new QVBoxLayout(container);
  signal_edit_layout = new QVBoxLayout();
  signal_edit_layout->setSpacing(2);
  container_layout->addLayout(signal_edit_layout);

  messages_view = new MessagesView(this);
  container_layout->addWidget(messages_view);

  QScrollArea *scroll = new QScrollArea(this);
  scroll->setWidget(container);
  scroll->setWidgetResizable(true);
  scroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

  main_layout->addWidget(scroll);
  setFixedWidth(600);

  connect(parser, &Parser::updated, this, &DetailWidget::updateState);
}

void DetailWidget::updateState() {
  if (msg_id.isEmpty()) return;

  auto &list = parser->can_msgs[msg_id];
  if (!list.empty()) {
    binary_view->setData(list.back().dat);
    messages_view->setMessages(list);
  }
}

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

void DetailWidget::setMsg(const QString &id) {
  msg_id = id;
  QString name = tr("untitled");

  for (auto edit : signal_edit) {
    delete edit;
  }
  signal_edit.clear();
  int i = 0;
  auto msg = parser->getMsg(id);
  if (msg) {
    for (auto &s : msg->sigs) {
      SignalEdit *edit = new SignalEdit(id, s, i++, this);
      connect(edit, &SignalEdit::removed, [=]() {
        QTimer::singleShot(0, [=]() { setMsg(id); });
      });
      signal_edit_layout->addWidget(edit);
      signal_edit.push_back(edit);
    }
    name = msg->name.c_str();
  }
  name_label->setText(name);
  binary_view->setMsg(msg_id);

  edit_btn->setVisible(true);
  add_sig_btn->setVisible(msg != nullptr);
}

SignalEdit::SignalEdit(const QString &id, const Signal &sig, int idx, QWidget *parent) : id(id), name_(sig.name.c_str()), QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // title
  QHBoxLayout *title_layout = new QHBoxLayout();
  QLabel *icon = new QLabel(">");
  icon->setStyleSheet("font-weight:bold");
  title_layout->addWidget(icon);
  title = new ElidedLabel(this);
  title->setText(sig.name.c_str());
  title->setStyleSheet(QString("font-weight:bold; color:%1").arg(SIGNAL_COLORS[idx % std::size(SIGNAL_COLORS)]));
  connect(title, &ElidedLabel::clicked, [=]() {
    edit_container->isVisible() ? edit_container->hide() : edit_container->show();
    icon->setText(edit_container->isVisible() ? "▼" : ">");
  });
  title_layout->addWidget(title);
  title_layout->addStretch();
  QPushButton *show_plot = new QPushButton(tr("Show Plot"));
  QObject::connect(show_plot, &QPushButton::clicked, [=]() {
    if (show_plot->text() == tr("Show Plot")) {
      emit parser->showPlot(id, name_);
      show_plot->setText(tr("Hide Plot"));
    } else {
      emit parser->hidePlot(id, name_);
      show_plot->setText(tr("Show Plot"));
    }
  });
  title_layout->addWidget(show_plot);
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
  Msg *msg = const_cast<Msg *>(parser->getMsg(id));
  if (!msg) return;

  for (auto &sig : msg->sigs) {
    if (name_ == sig.name.c_str()) {
      if (auto s = form->getSignal()) {
        sig = *s;
      }
      break;
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
    emit removed();
  }
}

BinaryView::BinaryView(QWidget *parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  table = new QTableWidget(this);
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->hide();
  table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  main_layout->addWidget(table);
  table->setColumnCount(9);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
}

void BinaryView::setMsg(const QString &id) {
  auto msg = parser->getMsg(Parser::addressFromId(id));
  int row_count = msg ? msg->size : parser->can_msgs[id].back().dat.size();

  table->setRowCount(row_count);
  table->setColumnCount(9);
  for (int i = 0; i < table->rowCount(); ++i) {
    for (int j = 0; j < table->columnCount(); ++j) {
      auto item = new QTableWidgetItem();
      item->setTextAlignment(Qt::AlignCenter);
      if (j == 8) {
        QFont font;
        font.setBold(true);
        item->setFont(font);
      }
      table->setItem(i, j, item);
    }
  }

  if (msg) {
    for (int i = 0; i < msg->sigs.size(); ++i) {
      const auto &sig = msg->sigs[i];
      int start = sig.is_little_endian ? sig.start_bit : bigEndianBitIndex(sig.start_bit);
      for (int j = start; j <= start + sig.size - 1; ++j) {
        table->item(j / 8, j % 8)->setBackground(QColor(SIGNAL_COLORS[i % std::size(SIGNAL_COLORS)]));
      }
    }
  }

  setFixedHeight(table->rowHeight(0) * table->rowCount() + 25);
  if (!parser->can_msgs.empty()) {
    setData(parser->can_msgs[id].back().dat);
  }
}

void BinaryView::setData(const QByteArray &binary) {
  std::string s;
  for (int j = 0; j < binary.size(); ++j) {
    s += std::bitset<8>(binary[j]).to_string();
  }

  char hex[3] = {'\0'};
  for (int i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      table->item(i, j)->setText(QChar(s[i * 8 + j]));
    }
    sprintf(&hex[0], "%02X", (unsigned char)binary[i]);
    table->item(i, 8)->setText(hex);
  }
}

MessagesView::MessagesView(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QLabel *title = new QLabel("MESSAGE    TIME       BYTES");
  main_layout->addWidget(title);

  message_layout = new QVBoxLayout();
  main_layout->addLayout(message_layout);
  main_layout->addStretch();
}

void MessagesView::setMessages(const std::list<CanData> &list) {
  auto begin = list.begin();
  std::advance(begin, std::max(0, (int)(list.size() - 100)));
  int j = 0;
  for (auto it = begin; it != list.end(); ++it) {
    QLabel *label;
    if (j >= messages.size()) {
      label = new QLabel();
      message_layout->addWidget(label);
      messages.push_back(label);
    } else {
      label = messages[j];
    }
    label->setText(it->hex_dat);
    ++j;
  }
}

EditMessageDialog::EditMessageDialog(const QString &id, QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Edit message"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addWidget(new QLabel(tr("ID: (%1)").arg(id)));

  auto msg = const_cast<Msg *>(parser->getMsg(Parser::addressFromId(id)));
  QHBoxLayout *h_layout = new QHBoxLayout();
  h_layout->addWidget(new QLabel(tr("Name")));
  h_layout->addStretch();
  QLineEdit *name_edit = new QLineEdit(this);
  name_edit->setText(msg ? msg->name.c_str() : "untitled");
  h_layout->addWidget(name_edit);
  main_layout->addLayout(h_layout);

  h_layout = new QHBoxLayout();
  h_layout->addWidget(new QLabel(tr("Size")));
  h_layout->addStretch();
  QSpinBox *size_spin = new QSpinBox(this);
  size_spin->setValue(msg ? msg->size : parser->can_msgs[id].back().dat.size());
  h_layout->addWidget(size_spin);
  main_layout->addLayout(h_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, [=]() {
    if (size_spin->value() <= 0 || name_edit->text().isEmpty()) return;

    if (msg) {
      msg->name = name_edit->text().toStdString();
      msg->size = size_spin->value();
    } else {
      Msg m = {};
      m.address = Parser::addressFromId(id);
      m.name = name_edit->text().toStdString();
      m.size = size_spin->value();
      parser->addNewMsg(m);
    }
    QDialog::accept();
  });
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

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
