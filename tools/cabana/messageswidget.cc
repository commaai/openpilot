#include "tools/cabana/messageswidget.h"

#include <QComboBox>
#include <QHeaderView>
#include <QPushButton>
#include <QVBoxLayout>

MessagesWidget::MessagesWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *dbc_file_layout = new QHBoxLayout();
  QComboBox *combo = new QComboBox(this);
  auto dbc_names = get_dbc_names();
  for (const auto &name : dbc_names) {
    combo->addItem(QString::fromStdString(name));
  }
  dbc_file_layout->addWidget(combo);

  dbc_file_layout->addStretch();
  QPushButton *save_btn = new QPushButton(tr("Save DBC"), this);
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
  main_layout->addWidget(table_widget);

  QObject::connect(parser, &Parser::updated, this, &MessagesWidget::updateState);
  QObject::connect(save_btn, &QPushButton::clicked, [=]() {
    // TODO: save DBC to file
  });
  QObject::connect(combo, &QComboBox::currentTextChanged, [=](const QString &dbc) {
    parser->openDBC(dbc);
  });
  QObject::connect(table_widget, &QTableWidget::itemSelectionChanged, [=]() {
    const CanData *c = &(parser->can_msgs[table_widget->selectedItems()[1]->text()]);
    parser->setCurrentMsg(c->id);
    emit msgChanged(c);
  });

  // For test purpose
  combo->setCurrentText("toyota_nodsu_pt_generated");
}

void MessagesWidget::updateState() {
  auto getTableItem = [=](int row, int col) -> QTableWidgetItem * {
    auto item = table_widget->item(row, col);
    if (!item) {
      item = new QTableWidgetItem();
      item->setFlags(item->flags() ^ Qt::ItemIsEditable);
      table_widget->setItem(row, col, item);
    }
    return item;
  };

  table_widget->setRowCount(parser->can_msgs.size());
  int i = 0;
  QString name, untitled = tr("untitled");
  const QString filter_str = filter->text();
  for (const auto &[_, c] : parser->can_msgs) {
    if (auto msg = parser->getMsg(c.address)) {
      name = msg->name.c_str();
    } else {
      name = untitled;
    }
    if (!filter_str.isEmpty() && !name.contains(filter_str, Qt::CaseInsensitive)) {
      table_widget->hideRow(i++);
      continue;
    }

    getTableItem(i, 0)->setText(name);
    getTableItem(i, 1)->setText(c.id);
    getTableItem(i, 2)->setText(QString::number(parser->counters[c.id]));
    getTableItem(i, 3)->setText(c.hex_dat);
    table_widget->showRow(i);
    i++;
  }
  if (table_widget->currentRow() == -1) {
    table_widget->selectRow(0);
  }
}
