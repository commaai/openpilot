#include "tools/cabana/canwidget.h"

#include <QComboBox>
#include <QVBoxLayout>

#include "opendbc/can/common_dbc.h"

CanWidget::CanWidget(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QComboBox *combo = new QComboBox(this);
  auto dbc_names = get_dbc_names();
  for (const auto &name : dbc_names) {
    combo->addItem(QString::fromStdString(name));
  }

  main_layout->addWidget(combo);
  main_layout->addStretch();
}

HexWidget::HexWidget(QWidget *parent) : QWidget(parent) {
}

