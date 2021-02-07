#pragma once

#include <QLabel>
#include <QString>
#include <QWidget>
#include <QDialog>
#include <QLineEdit>
#include <QVBoxLayout>

#include "keyboard.hpp"

class InputDialog : public QDialog {
  Q_OBJECT

public:
  explicit InputDialog(QString prompt_text, QWidget* parent = 0);
  static QString getText(QString prompt);
  QString text();

private:
  QLineEdit *line;
  Keyboard *k;
  QLabel *label;
  QVBoxLayout *layout;

public slots:
  int exec() override;

private slots:
  void handleInput(QString s);
};
