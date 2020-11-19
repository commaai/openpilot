#pragma once

#include <QWidget>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QStackedLayout>

#include "keyboard.hpp"

class InputField : public QWidget {
  Q_OBJECT

public:
  explicit InputField(QWidget* parent = 0);
  
private:
  Keyboard *k;
  QLineEdit *line;
  QVBoxLayout *l;

public slots:
  void getText(QString s);

signals:
  void emitText(QString s);
};
