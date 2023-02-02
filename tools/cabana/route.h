#pragma once

#include <QDialogButtonBox>
#include <QLineEdit>
#include <QDialog>

class OpenRouteDialog : public QDialog {
  Q_OBJECT

public:
  OpenRouteDialog(QWidget *parent);
  void loadRoute();

  QLineEdit *route_edit;
  QDialogButtonBox *btn_box;
  bool success = true;
};
