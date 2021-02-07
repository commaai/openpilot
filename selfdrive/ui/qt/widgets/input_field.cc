#include <QPushButton>

#include "input_field.hpp"
#include "qt_window.hpp"

InputField::InputField(QWidget *parent, int minTextLength): QWidget(parent), minTextLength(minTextLength) {
  layout = new QGridLayout();
  layout->setSpacing(30);

  label = new QLabel(this);
  label->setStyleSheet(R"(font-size: 70px; font-weight: 500;)");
  layout->addWidget(label, 0, 0,Qt::AlignLeft);
  layout->setColumnStretch(0, 1);

  QPushButton* cancel = new QPushButton("Cancel");
  cancel->setFixedSize(300, 150);
  cancel->setStyleSheet(R"(padding: 0;)");
  layout->addWidget(cancel, 0, 1, Qt::AlignRight);
  QObject::connect(cancel, SIGNAL(released()), this, SLOT(emitEmpty()));

  // text box
  line = new QLineEdit();
  line->setStyleSheet(R"(
    color: white;
    background-color: #444444;
    font-size: 80px;
    font-weight: 500;
    padding: 10px;
  )");
  layout->addWidget(line, 1, 0, 1, -1);

  k = new Keyboard(this);
  QObject::connect(k, SIGNAL(emitButton(QString)), this, SLOT(getText(QString)));
  layout->addWidget(k, 2, 0, 1, -1);

  setLayout(layout);
}

void InputField::setPromptText(QString text) {
  label->setText(text);
}

void InputField::emitEmpty() {
  line->setText("");
  emit cancel();
}

void InputField::getText(QString s) {
  if (!QString::compare(s,"⌫")) {
    line->backspace();
  }

  if (!QString::compare(s,"⏎")) {
    if(line->text().length()<minTextLength){
      setPromptText("Need at least "+QString::number(minTextLength)+" characters!");
      return;
    }
    emitText(line->text());
    line->setText("");
  }

  QVector<QString> control_buttons {"⇧", "↑", "ABC", "⏎", "#+=", "⌫", "123"};
  for(QString c : control_buttons) {
    if (!QString::compare(s, c)) {
      return;
    }
  }

  line->insert(s.left(1));
}




InputDialog::InputDialog(QString prompt_text, QWidget *parent): QDialog(parent) {
  layout = new QVBoxLayout();
  layout->setContentsMargins(50, 50, 50, 50);
  layout->setSpacing(20);

  // build header
  QHBoxLayout *header_layout = new QHBoxLayout();

  label = new QLabel(prompt_text, this);
  label->setStyleSheet(R"(font-size: 75px; font-weight: 500;)");
  header_layout->addWidget(label, 1, Qt::AlignLeft);

  QPushButton* cancel_btn = new QPushButton("Cancel");
  cancel_btn->setStyleSheet(R"(
    padding: 30px;
    padding-right: 45px;
    padding-left: 45px;
    border: 10px solid #444444;
    font-size: 45px;
    background-color: #444444;
  )");
  header_layout->addWidget(cancel_btn, 0, Qt::AlignRight);
  QObject::connect(cancel_btn, SIGNAL(released()), this, SLOT(reject()));

  layout->addLayout(header_layout);

  // text box
  layout->addSpacing(20);
  line = new QLineEdit();
  line->setStyleSheet(R"(
    border: none;
    background-color: #444444;
    font-size: 80px;
    font-weight: 500;
    padding: 10px;
  )");
  layout->addWidget(line, 1, Qt::AlignTop);

  k = new Keyboard(this);
  QObject::connect(k, SIGNAL(emitButton(QString)), this, SLOT(handleInput(QString)));
  layout->addWidget(k, 2, Qt::AlignBottom);

  setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
    }
  )");

  setLayout(layout);
}

QString InputDialog::getText(const QString prompt) {
  InputDialog d = InputDialog(prompt);
  const int ret = d.exec();
  if (ret) {
    return d.text();
  } else {
    return QString();
  }
}

QString InputDialog::text() {
  return line->text();
}

int InputDialog::exec() {
  setMainWindow(this);
  return QDialog::exec();
}

void InputDialog::handleInput(QString s) {
  if (!QString::compare(s,"⌫")) {
    line->backspace();
  }

  if (!QString::compare(s,"⏎")) {
    done(QDialog::Accepted);
  }

  QVector<QString> control_buttons {"⇧", "↑", "ABC", "⏎", "#+=", "⌫", "123"};
  for(QString c : control_buttons) {
    if (!QString::compare(s, c)) {
      return;
    }
  }

  line->insert(s.left(1));
}

