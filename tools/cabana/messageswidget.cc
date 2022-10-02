#include "tools/cabana/messageswidget.h"

#include <QComboBox>
#include <QDebug>
#include <QHeaderView>
#include <QVBoxLayout>
#include <bitset>

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QComboBox *combo = new QComboBox(this);
  auto dbc_names = get_dbc_names();
  for (const auto &name : dbc_names) {
    combo->addItem(QString::fromStdString(name));
  }
  connect(combo, &QComboBox::currentTextChanged, [=](const QString &dbc) {
    parser->openDBC(dbc);
  });

  // For test purpose
  combo->setCurrentText("toyota_nodsu_pt_generated");

  main_layout->addWidget(combo);

  filter = new QLineEdit(this);
  filter->setPlaceholderText(tr("filter messages"));
  main_layout->addWidget(filter);

  table_widget = new QTableWidget(this);
  table_widget->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  table_widget->setColumnCount(4);
  table_widget->setColumnWidth(0, 250);
  table_widget->setColumnWidth(1, 80);
  table_widget->setColumnWidth(2, 80);
  table_widget->setHorizontalHeaderLabels({tr("Name"), tr("ID"), tr("Count"), tr("Bytes")});
  table_widget->horizontalHeader()->setStretchLastSection(true);

  QObject::connect(table_widget, &QTableWidget::itemSelectionChanged, [=]() {
    auto address = table_widget->selectedItems()[0]->data(Qt::UserRole);
    emit addressChanged(address.toUInt());
  });

  main_layout->addWidget(table_widget);

  // main_layout->addStretch();
  connect(parser, &Parser::updated, this, &MessagesWidget::updateState);
}

void MessagesWidget::updateState() {
  auto getTableItem = [=](int row, int col) -> QTableWidgetItem * {
    auto item = table_widget->item(row, col);
    if (!item) {
      item = new QTableWidgetItem();
      table_widget->setItem(row, col, item);
    }
    return item;
  };

  table_widget->setRowCount(parser->items.size());
  int i = 0;
  const QString filter_str = filter->text();
  for (auto &[address, list] : parser->items) {
    if (list.empty()) continue;

    QString name = tr("untitled");
    auto it = parser->msg_map.find(address);
    if (it != parser->msg_map.end()) {
      name = it->second->name.c_str();
    }
    if (!filter_str.isEmpty() && !name.contains(filter_str)) continue;

    auto item = getTableItem(i, 0);
    item->setText(name);
    item->setData(Qt::UserRole, address);
    getTableItem(i, 1)->setText(list.back().id);
    getTableItem(i, 2)->setText(QString("%1").arg(parser->counters[address]));
    getTableItem(i, 3)->setText(list.back().hex_dat);
    ++i;
  }
  table_widget->setRowCount(i);
}
