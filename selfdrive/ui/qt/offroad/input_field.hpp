#pragma once

#include <QWidget>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QStackedLayout>
#include <QLabel>

#include "keyboard.hpp"

class InputField : public QWidget {
  Q_OBJECT

public:
  explicit InputField(QWidget* parent = 0);
  QLabel *label;
  
private:
  QLineEdit *line;
  Keyboard *k;
  QVBoxLayout *l;

public slots:
  void emitEmpty();
  void getText(QString s);

signals:
  void emitText(QString s);
};
