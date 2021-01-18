#pragma once

#include <QLabel>
#include <QString>
#include <QWidget>
#include <QLineEdit>
#include <QGridLayout>

#include "keyboard.hpp"

class InputField : public QWidget {
  Q_OBJECT

public:
  explicit InputField(QWidget* parent = 0);
  void setPromptText(QString text);

private:
  QLineEdit *line;
  Keyboard *k;
  QLabel *label;
  QGridLayout *layout;

public slots:
  void getText(QString s);
  void emitEmpty();

signals:
  void cancel();
  void emitText(QString s);
};
