#include "tools/cabana/messageswidget.h"

#include <QComboBox>
#include <QDebug>
#include <QHeaderView>
#include <QPushButton>
#include <QVBoxLayout>
#include <bitset>

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *dbc_file_layout = new QHBoxLayout();
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
  dbc_file_layout->addWidget(combo);

  dbc_file_layout->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save DBC"), this);
  QObject::connect(save_btn, &QPushButton::clicked, [=]() {
    // TODO: save DBC to file
  });
  dbc_file_layout->addWidget(save_btn);

  main_layout->addLayout(dbc_file_layout);

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
    auto id = table_widget->selectedItems()[0]->data(Qt::UserRole);
    emit msgChanged(id.toString());
  });
  main_layout->addWidget(table_widget);

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

  table_widget->setRowCount(parser->can_msgs.size());
  int i = 0;
  const QString filter_str = filter->text().toLower();
  for (const auto &[id, list] : parser->can_msgs) {
    assert(!list.empty());

    QString name;
    if (auto msg = parser->getMsg(list.back().address)) {
      name = msg->name.c_str();
    } else {
      name = tr("untitled");
    }
    if (!filter_str.isEmpty() && !name.toLower().contains(filter_str)) {
      table_widget->hideRow(i++);
      continue;
    }

    auto item = getTableItem(i, 0);
    item->setText(name);
    item->setData(Qt::UserRole, id);
    getTableItem(i, 1)->setText(id);
    getTableItem(i, 2)->setText(QString("%1").arg(parser->counters[id]));
    getTableItem(i, 3)->setText(list.back().hex_dat);
    table_widget->showRow(i);
    i++;
  }
}
