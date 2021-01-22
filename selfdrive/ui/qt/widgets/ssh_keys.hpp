#pragma once

#include <QWidget>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QPushButton>
#include <QTimer>

#include "widgets/input_field.hpp"

class SSH : public QWidget {
  Q_OBJECT

public:
  explicit SSH(QWidget* parent = 0);

private:
  InputField* inputField;
  QStackedLayout* slayout;
  

signals:
  void closeSSHSettings();
  void openKeyboard();
  void closeKeyboard();

};

