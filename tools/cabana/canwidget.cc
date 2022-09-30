#include "tools/cabana/canwidget.h"

#include <QComboBox>
#include <QDebug>
#include <QHeaderView>
#include <QVBoxLayout>
#include <bitset>

CanWidget::CanWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QComboBox *combo = new QComboBox(this);
  auto dbc_names = get_dbc_names();
  for (const auto &name : dbc_names) {
    combo->addItem(QString::fromStdString(name));
  }

  main_layout->addWidget(combo);
  parser->openDBC("toyota_nodsu_pt_generated");

  table_widget = new QTableWidget(this);
  table_widget->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
  QObject::connect(table_widget, &QTableWidget::itemSelectionChanged, [=]() {
    auto address = table_widget->selectedItems()[0]->data(Qt::UserRole);
    emit addressChanged(address.toUInt());
  });
  table_widget->setColumnCount(4);
  table_widget->setColumnWidth(2, 20);
  table_widget->setHorizontalHeaderLabels({tr("Name"), tr("ID"), tr("Count"), tr("Bytes")});
  main_layout->addWidget(table_widget);

  // main_layout->addStretch();
  connect(parser, &Parser::updated, this, &CanWidget::updateState);
}

void CanWidget::updateState() {
  table_widget->setRowCount(parser->items.size());
  int i = 0;
  for (auto &[address, data] : parser->items) {
    QString name = tr("untitled");
    auto it = parser->msg_map.find(address);
    if (it != parser->msg_map.end()) {
      name = it->second->name.c_str();
    }
    auto item = new QTableWidgetItem(name);
    item->setData(Qt::UserRole, address);
    table_widget->setItem(i, 0, item);
    table_widget->setItem(i, 2, new QTableWidgetItem(QString("%1").arg(data.cnt)));
    std::string s;
    for (int j = 0; j < data.data.dat.size(); ++j) {
      s += std::bitset<8>(data.data.dat[j]).to_string();
    }
    table_widget->setItem(i, 3, new QTableWidgetItem(s.c_str()));
    ++i;
  }
}
