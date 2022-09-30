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

  name_label = new QLabel(this);
  main_layout->addWidget(name_label);

  binary_view = new BinaryView(this);
  main_layout->addWidget(binary_view);

  QWidget *container = new QWidget(this);
  // container->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  signal_edit_layout = new QVBoxLayout(container);
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
  auto it = parser->msg_map.find(address);
  QString name = tr("untitled");
  if (it != parser->msg_map.end()) {
    name = it->second->name.c_str();
    binary_view->setData(parser->items[address].data.dat);
  }
  name_label->setText(name);
}

void DetailWidget::showEvent(QShowEvent *event) {
}

void DetailWidget::hideEvent(QHideEvent *event) {
}

void DetailWidget::setItem(uint32_t addr) {
  address = addr;
  for (auto edit : signal_edit) {
    delete edit;
  }
  signal_edit.clear();
  auto it = parser->msg_map.find(addr);
  if (it != parser->msg_map.end()) {
    for (auto &sig : it->second->sigs) {
      SignalEdit *edit = new SignalEdit(this);
      edit->setSig(sig);
      signal_edit_layout->addWidget(edit);
      signal_edit.push_back(edit);
    }
  }
}

SignalEdit::SignalEdit(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Name")));
  name = new QLineEdit();
  h->addWidget(name);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Size")));
  size = new QSpinBox();
  h->addWidget(size);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Most significant bit")));
  significant_bit = new QSpinBox();
  h->addWidget(significant_bit);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Endianness")));
  endianness = new QComboBox();
  h->addWidget(endianness);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("sign")));
  sign = new QComboBox();
  h->addWidget(sign);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Factor")));
  factor = new QSpinBox();
  h->addWidget(factor);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Offset")));
  offset = new QSpinBox();
  h->addWidget(offset);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Unit")));
  unit = new QLineEdit();
  h->addWidget(unit);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Comment")));
  comment = new QLineEdit();
  h->addWidget(comment);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Minimum value")));
  min_val = new QSpinBox();
  h->addWidget(min_val);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Maximum value")));
  max_val = new QSpinBox();
  h->addWidget(max_val);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  h->addWidget(new QLabel(tr("Value descriptions")));
  val_desc = new QLineEdit();
  h->addWidget(val_desc);
  main_layout->addLayout(h);

  h = new QHBoxLayout();
  remove_btn = new QPushButton(tr("Remove Signal"));
  h->addWidget(remove_btn);
  h->addStretch();
  main_layout->addLayout(h);
}

void SignalEdit::setSig(const Signal &sig) {
  name->setText(sig.name.c_str());
  size->setValue(sig.size);
  offset->setValue(sig.offset);
  factor->setValue(sig.factor);
}

BinaryView::BinaryView(QWidget *parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  table = new QTableWidget(this);
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  table->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
  main_layout->addWidget(table);
  table->setColumnCount(8);
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
}

void BinaryView::setData(const std::string &binary) {
  std::string s;
  for (int j = 0; j < binary.size(); ++j) {
    s += std::bitset<8>(binary[j]).to_string();
  }
  table->setRowCount(s.length() / 8);
  for (int i = 0; i < s.length() / 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      table->setItem(i, j, new QTableWidgetItem(QString("%1").arg(s[i * 8 + j])));
    }
  }
}
