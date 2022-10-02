#include "tools/cabana/detailwidget.h"

#include <QDebug>
#include <QHeaderView>
#include <QVBoxLayout>
#include <bitset>

#include "selfdrive/ui/qt/widgets/scrollview.h"

DetailWidget::DetailWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QLabel *title = new QLabel(tr("SELECTED MESSAGE:"), this);
  main_layout->addWidget(title);

  QHBoxLayout *name_layout = new QHBoxLayout();
  name_label = new QLabel(this);
  name_label->setStyleSheet("font-weight:bold;");
  name_layout->addWidget(name_label);
  name_layout->addStretch();
  QPushButton *edit_btn = new QPushButton(tr("Edit"), this);
  name_layout->addWidget(edit_btn);

  main_layout->addLayout(name_layout);

  binary_view = new BinaryView(this);
  main_layout->addWidget(binary_view);

  QWidget *container = new QWidget(this);
  QVBoxLayout *container_layout = new QVBoxLayout(container);
  // container->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  signal_edit_layout = new QVBoxLayout();
  signal_edit_layout->setSpacing(2);
  container_layout->addLayout(signal_edit_layout);

  messages_view = new MessagesView(this);
  container_layout->addWidget(messages_view);

  QScrollArea *scroll = new QScrollArea(this);
  scroll->setWidget(container);
  scroll->setWidgetResizable(true);
  scroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  // scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  main_layout->addWidget(scroll);
  // main_layout->addStretch(0);
  setFixedWidth(600);

  connect(parser, &Parser::updated, this, &DetailWidget::updateState);
}

void DetailWidget::updateState() {
  auto &list = parser->items[address];
  if (!list.empty()) {
    binary_view->setData(list.back().dat);
    messages_view->setMessages(list);
  }
}

void DetailWidget::setItem(uint32_t addr) {
  if (address == addr) return;

  address = addr;

  QString name = tr("untitled");

  for (auto edit : signal_edit) {
    delete edit;
  }
  signal_edit.clear();
  auto it = parser->msg_map.find(addr);
  if (it != parser->msg_map.end()) {
    for (auto &s : it->second->sigs) {
      SignalEdit *edit = new SignalEdit(this);
      edit->setSig(address, s);
      signal_edit_layout->addWidget(edit);
      signal_edit.push_back(edit);
    }
    name = it->second->name.c_str();
  }
  name_label->setText(name);
}

SignalEdit::SignalEdit(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);

  // title
  QHBoxLayout *title_layout = new QHBoxLayout();
  QLabel *icon = new QLabel(">");
  icon->setStyleSheet("font-weight:bold");
  title_layout->addWidget(icon);
  title = new ElidedLabel(this);
  title->setStyleSheet("font-weight:bold");
  connect(title, &ElidedLabel::clicked, [=]() {
    edit_container->isVisible() ? edit_container->hide() : edit_container->show();
    icon->setText(edit_container->isVisible() ? "▼" : ">");
  });
  title_layout->addWidget(title);
  title_layout->addStretch();
  QPushButton *show_plot = new QPushButton(tr("Show Pilot"));
  QObject::connect(show_plot, &QPushButton::clicked, [=]() {
    emit parser->showPlot(address_, name_);
  });
  title_layout->addWidget(show_plot);

  main_layout->addLayout(title_layout);

  edit_container = new QWidget(this);
  QVBoxLayout *v_layout = new QVBoxLayout(edit_container);

  QHBoxLayout *h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Name")));
  name = new QLineEdit();
  h->addWidget(name);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Size")));
  size = new QSpinBox();
  h->addWidget(size);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Most significant bit")));
  msb = new QSpinBox();
  h->addWidget(msb);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Endianness")));
  endianness = new QComboBox();
  endianness->addItems({"Little", "Big"});
  h->addWidget(endianness);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("sign")));
  sign = new QComboBox();
  sign->addItems({"Signed", "Unsigned"});
  h->addWidget(sign);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Factor")));
  factor = new QSpinBox();
  h->addWidget(factor);
  v_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Offset")));
  offset = new QSpinBox();
  h->addWidget(offset);
  v_layout->addLayout(h);

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

  h = new QHBoxLayout();
  remove_btn = new QPushButton(tr("Remove Signal"));
  h->addWidget(remove_btn);
  h->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save"));
  h->addWidget(save_btn);
  v_layout->addLayout(h);

  main_layout->addWidget(edit_container);

  edit_container->setVisible(false);
}

void SignalEdit::setSig(uint32_t address, const Signal &sig) {
  address_ = address;
  name_ = sig.name.c_str();
  title->setText(sig.name.c_str());
  name->setText(sig.name.c_str());
  size->setValue(sig.size);
  offset->setValue(sig.offset);
  factor->setValue(sig.factor);
  msb->setValue(sig.msb);
  sign->setCurrentIndex(sig.is_signed ? 0 : 1);
  endianness->setCurrentIndex(sig.is_little_endian ? 0 : 1);
}

BinaryView::BinaryView(QWidget *parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  table = new QTableWidget(this);
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->hide();
  table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  // table->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  main_layout->addWidget(table);
  table->setColumnCount(9);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
}

void BinaryView::setData(const QByteArray &binary) {
  auto getTableItem = [=](int row, int col) -> QTableWidgetItem * {
    auto item = table->item(row, col);
    if (!item) {
      item = new QTableWidgetItem();
      item->setTextAlignment(Qt::AlignCenter);
      if (col == 8) {
        QFont font;
        font.setBold(true);
        item->setFont(font);
      }
      table->setItem(row, col, item);
    }
    return item;
  };

  std::string s;
  for (int j = 0; j < binary.size(); ++j) {
    s += std::bitset<8>(binary[j]).to_string();
  }
  int old_row_count = table->rowCount();
  table->setRowCount(binary.size());

  char hex[3] = {'\0'};
  for (int i = 0; i < binary.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      getTableItem(i, j)->setText(QChar(s[i * 8 + j]));
    }
    sprintf(&hex[0], "%02X", (unsigned char)binary[i]);
    auto item = getTableItem(i, 8);
    item->setText(hex);
  }

  if (old_row_count != binary.size()) {
    int height = 0;
    for (int i = 0; i < table->rowCount(); i++) {
      height += table->rowHeight(i);
    }
    setFixedHeight(height + 25);
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
