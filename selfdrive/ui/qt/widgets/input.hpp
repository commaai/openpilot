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
  static QString getText(QString prompt, int minLength = -1);
  QString text();
  void setMessage(QString message, bool clearInputField=true);
  void setMinLength(int length);
  void show();

private:
  int minLength;
  QLineEdit *line;
  Keyboard *k;
  QLabel *label;
  QVBoxLayout *layout;

public slots:
  int exec() override;

private slots:
  void handleInput(QString s);

signals:
  void cancel();
  void emitText(QString text);
};


class ConfirmationDialog : public QDialog {
  Q_OBJECT

public:
  explicit ConfirmationDialog(QString prompt_text, QString confirm_text = "Ok",
                              QString cancel_text = "Cancel", QWidget* parent = 0);
  static bool confirm(QString prompt_text);

private:
  QLabel *prompt;
  QVBoxLayout *layout;

public slots:
  int exec() override;
};
