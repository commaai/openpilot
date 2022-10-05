
#include "tools/cabana/detailwidget.h"

#include <QDebug>
#include <QDialogButtonBox>
#include <QHeaderView>
#include <QTimer>
#include <QVBoxLayout>
#include <bitset>

#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

inline const QString &getColor(int i) {
  static const QString SIGNAL_COLORS[] = {"#9FE2BF", "#40E0D0", "#6495ED", "#CCCCFF", "#FF7F50", "#FFBF00"};
  return SIGNAL_COLORS[i % std::size(SIGNAL_COLORS)];
}

// DetailWidget

DetailWidget::DetailWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  name_label = new QLabel(this);
  name_label->setStyleSheet("font-weight:bold;");
  name_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  name_label->setAlignment(Qt::AlignCenter);
  main_layout->addWidget(name_label);

  // title
  QHBoxLayout *title_layout = new QHBoxLayout();
  time_label = new QLabel(this);
  title_layout->addWidget(time_label);
  title_layout->addStretch();

  edit_btn = new QPushButton(tr("Edit"), this);
  edit_btn->setVisible(false);
  title_layout->addWidget(edit_btn);
  main_layout->addLayout(title_layout);

  // binary view
  binary_view = new BinaryView(this);
  main_layout->addWidget(binary_view);

  // scroll area
  QHBoxLayout *signals_layout = new QHBoxLayout();
  signals_layout->addWidget(new QLabel(tr("Signals")));
  signals_layout->addStretch();
  add_sig_btn = new QPushButton(tr("Add signal"), this);
  add_sig_btn->setVisible(false);
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
  scroll->setFrameShape(QFrame::NoFrame);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

  main_layout->addWidget(scroll);

  QObject::connect(add_sig_btn, &QPushButton::clicked, this, &DetailWidget::addSignal);
  QObject::connect(edit_btn, &QPushButton::clicked, this, &DetailWidget::editMsg);
  QObject::connect(parser, &Parser::updated, this, &DetailWidget::updateState);
}

void DetailWidget::setMsg(const QString &id) {
  msg_id = id;
  clearLayout(signal_edit_layout);

  if (auto msg = parser->getMsg(id)) {
    name_label->setText(msg->name.c_str());
    add_sig_btn->setVisible(true);
    for (int i = 0; i < msg->sigs.size(); ++i) {
      signal_edit_layout->addWidget(new SignalEdit(id, msg->sigs[i], getColor(i)));
    }
  } else {
    name_label->setText(tr("untitled"));
    add_sig_btn->setVisible(false);
  }

  binary_view->setMsg(msg_id);
  edit_btn->setVisible(true);
}

void DetailWidget::updateState() {
  if (msg_id.isEmpty()) return;

  auto list = parser->getCanMessages(msg_id);
  if (list && !list->empty()) {
    time_label->setText(QString("time: %1").arg(list->back().ts, 0, 'f', 3));
    binary_view->setData(list->back().dat);
    messages_view->setMessages(*list);
  }
}

void DetailWidget::editMsg() {
  EditMessageDialog dlg(msg_id, this);
  if (dlg.exec()) {
    setMsg(msg_id);
  }
}

void DetailWidget::addSignal() {
  AddSignalDialog dlg(msg_id, this);
  if (dlg.exec()) {
    setMsg(msg_id);
  }
}

// BinaryView

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
  const Msg *msg = parser->getMsg(id);
  const auto &dat = parser->can_msgs[id].back().dat;
  int row_count = msg ? msg->size : dat.size();

  table->setRowCount(row_count);
  table->setColumnCount(9);
  for (int i = 0; i < table->rowCount(); ++i) {
    for (int j = 0; j < table->columnCount(); ++j) {
      auto item = new QTableWidgetItem();
      item->setFlags(item->flags() ^ Qt::ItemIsEditable);
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
    // set background color
    for (int i = 0; i < msg->sigs.size(); ++i) {
      const auto &sig = msg->sigs[i];
      int start = sig.is_little_endian ? sig.start_bit : bigEndianBitIndex(sig.start_bit);
      for (int j = start; j <= start + sig.size - 1; ++j) {
        table->item(j / 8, j % 8)->setBackground(QColor(getColor(i)));
      }
    }
  }

  setFixedHeight(table->rowHeight(0) * table->rowCount() + 25);
  setData(dat);
}

void BinaryView::setData(const QByteArray &binary) {
  std::string s;
  for (int j = 0; j < binary.size(); ++j) {
    s += std::bitset<8>(binary[j]).to_string();
  }

  setUpdatesEnabled(false);
  char hex[3] = {'\0'};
  for (int i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      table->item(i, j)->setText(QChar(s[i * 8 + j]));
    }
    sprintf(&hex[0], "%02X", (unsigned char)binary[i]);
    table->item(i, 8)->setText(hex);
  }
  setUpdatesEnabled(true);
}

// MessagesView

MessagesView::MessagesView(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QLabel *title = new QLabel("TIME         BYTES");
  main_layout->addWidget(title);

  message_layout = new QVBoxLayout();
  main_layout->addLayout(message_layout);
  main_layout->addStretch();
}

void MessagesView::setMessages(const std::list<CanData> &list) {
  int j = 0;
  for (const auto &can_data : list) {
    QLabel *label;
    if (j >= messages.size()) {
      label = new QLabel();
      message_layout->addWidget(label);
      messages.push_back(label);
    } else {
      label = messages[j];
      label->setVisible(true);
    }
    label->setText(QString("%1         %2").arg(can_data.ts, 0, 'f', 3).arg(can_data.hex_dat));
    ++j;
  }
  for (; j < messages.size(); ++j) {
    messages[j]->setVisible(false);
  }
}

// EditMessageDialog

EditMessageDialog::EditMessageDialog(const QString &id, QWidget *parent) : id(id), QDialog(parent) {
  setWindowTitle(tr("Edit message"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addWidget(new QLabel(tr("ID: (%1)").arg(id)));

  auto msg = const_cast<Msg *>(parser->getMsg(id));
  QHBoxLayout *h_layout = new QHBoxLayout();
  h_layout->addWidget(new QLabel(tr("Name")));
  h_layout->addStretch();
  name_edit = new QLineEdit(this);
  name_edit->setText(msg ? msg->name.c_str() : "untitled");
  h_layout->addWidget(name_edit);
  main_layout->addLayout(h_layout);

  h_layout = new QHBoxLayout();
  h_layout->addWidget(new QLabel(tr("Size")));
  h_layout->addStretch();
  size_spin = new QSpinBox(this);
  size_spin->setValue(msg ? msg->size : parser->can_msgs[id].back().dat.size());
  h_layout->addWidget(size_spin);
  main_layout->addLayout(h_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, this, &EditMessageDialog::save);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void EditMessageDialog::save() {
  if (size_spin->value() <= 0 || name_edit->text().isEmpty()) return;

  if (auto msg = const_cast<Msg *>(parser->getMsg(id))) {
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
}
