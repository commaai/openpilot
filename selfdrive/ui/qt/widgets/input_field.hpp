#pragma once

#include <QLabel>
#include <QString>
#include <QWidget>
#include <QDialog>
#include <QLineEdit>
#include <QGridLayout>
#include <QVBoxLayout>

#include "keyboard.hpp"

class InputField : public QWidget {
  Q_OBJECT

public:
  explicit InputField(QWidget* parent = 0, int minTextLength = 0);
  void setPromptText(QString text);
  int minTextLength;

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

class InputDialog : public QDialog {
  Q_OBJECT

public:
  explicit InputDialog(QWidget* parent = 0, QString prompt_text = "");

private:
  QLineEdit *line;
  Keyboard *k;
  QLabel *label;
  QVBoxLayout *layout;

public slots:
  void getText(QString s);
  void emitEmpty();

signals:
  void cancel();
  void emitText(QString s);
};
