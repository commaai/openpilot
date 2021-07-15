#include "selfdrive/ui/qt/widgets/input.h"

#include <QPushButton>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/hardware/hw.h"

QDialogBase::QDialogBase(QWidget *parent) : QDialog(parent) {
  Q_ASSERT(parent != nullptr);
  parent->installEventFilter(this);
}

bool QDialogBase::eventFilter(QObject *o, QEvent *e) {
  if (o == parent() && e->type() == QEvent::Hide) {
    reject();
  }
  return QDialog::eventFilter(o, e);
}

InputDialog::InputDialog(const QString &title, QWidget *parent,
                         const QString &subtitle) : QDialogBase(parent) {
  main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 55, 50, 50);
  main_layout->setSpacing(0);

  // build header
  QHBoxLayout *header_layout = new QHBoxLayout();

  QVBoxLayout *vlayout = new QVBoxLayout;
  header_layout->addLayout(vlayout);
  label = new QLabel(title, this);
  label->setStyleSheet("font-size: 90px; font-weight: bold;");
  vlayout->addWidget(label, 1, Qt::AlignTop | Qt::AlignLeft);

  if (!subtitle.isEmpty()) {
    sublabel = new QLabel(subtitle, this);
    sublabel->setStyleSheet("font-size: 55px; font-weight: light; color: #BDBDBD;");
    vlayout->addWidget(sublabel, 1, Qt::AlignTop | Qt::AlignLeft);
  }

  QPushButton* cancel_btn = new QPushButton("Cancel");
  cancel_btn->setFixedSize(386, 125);
  cancel_btn->setStyleSheet(R"(
    font-size: 48px;
    border-radius: 10px;
    color: #E4E4E4;
    background-color: #444444;
  )");
  header_layout->addWidget(cancel_btn, 0, Qt::AlignRight);
  QObject::connect(cancel_btn, &QPushButton::released, this, &InputDialog::reject);
  QObject::connect(cancel_btn, &QPushButton::released, this, &InputDialog::cancel);

  main_layout->addLayout(header_layout);

  // text box
  main_layout->addStretch();
  line = new QLineEdit();
  line->setStyleSheet(R"(
    font-size: 80px;
    font-weight: light;
    margin-left: 50px;
    margin-right: 50px;
    border: none;
    border-radius: 0;
    border-bottom: 3px solid #BDBDBD;
  )");
  main_layout->addWidget(line, 0, Qt::AlignBottom);

  main_layout->addSpacing(25);
  k = new Keyboard(this);
  QObject::connect(k, &Keyboard::emitButton, this, &InputDialog::handleInput);
  main_layout->addWidget(k, 2, Qt::AlignBottom);

  setStyleSheet(R"(
    * {
      outline: none;
      color: white;
      font-family: Inter;
      background-color: black;
    }
  )");

}

QString InputDialog::getText(const QString &prompt, QWidget *parent, const QString &subtitle,
                             int minLength, const QString &defaultText) {
  InputDialog d = InputDialog(prompt, parent, subtitle);
  d.line->setText(defaultText);
  d.setMinLength(minLength);
  const int ret = d.exec();
  return ret ? d.text() : QString();
}

QString InputDialog::text() {
  return line->text();
}

int InputDialog::exec() {
  setMainWindow(this);
  return QDialog::exec();
}

void InputDialog::show() {
  setMainWindow(this);
}

void InputDialog::handleInput(const QString &s) {
  if (!QString::compare(s,"⌫")) {
    line->backspace();
  } else if (!QString::compare(s,"⏎")) {
    if (line->text().length() >= minLength) {
      done(QDialog::Accepted);
      emitText(line->text());
    } else {
      setMessage("Need at least "+QString::number(minLength)+" characters!", false);
    }
  } else {
    line->insert(s.left(1));
  }
}

void InputDialog::setMessage(const QString &message, bool clearInputField) {
  label->setText(message);
  if (clearInputField) {
    line->setText("");
  }
}

void InputDialog::setMinLength(int length) {
  minLength = length;
}

ConfirmationDialog::ConfirmationDialog(const QString &prompt_text, const QString &confirm_text, const QString &cancel_text,
                                       QWidget *parent) : QDialogBase(parent) {
  setWindowFlags(Qt::Popup);
  main_layout = new QVBoxLayout(this);
  main_layout->setMargin(25);

  prompt = new QLabel(prompt_text, this);
  prompt->setWordWrap(true);
  prompt->setAlignment(Qt::AlignHCenter);
  prompt->setStyleSheet(R"(font-size: 55px; font-weight: 400;)");
  main_layout->addWidget(prompt, 1, Qt::AlignTop | Qt::AlignHCenter);

  // cancel + confirm buttons
  QHBoxLayout *btn_layout = new QHBoxLayout();
  btn_layout->setSpacing(20);
  btn_layout->addStretch(1);
  main_layout->addLayout(btn_layout);

  if (cancel_text.length()) {
    QPushButton* cancel_btn = new QPushButton(cancel_text);
    btn_layout->addWidget(cancel_btn, 0, Qt::AlignRight);
    QObject::connect(cancel_btn, &QPushButton::released, this, &ConfirmationDialog::reject);
  }

  if (confirm_text.length()) {
    QPushButton* confirm_btn = new QPushButton(confirm_text);
    btn_layout->addWidget(confirm_btn, 0, Qt::AlignRight);
    QObject::connect(confirm_btn, &QPushButton::released, this, &ConfirmationDialog::accept);
  }

  setFixedSize(900, 350);
  setStyleSheet(R"(
    * {
      color: black;
      background-color: white;
    }
    QPushButton {
      font-size: 40px;
      padding: 30px;
      padding-right: 45px;
      padding-left: 45px;
      border-radius: 7px;
      background-color: #44444400;
    }
  )");
}

bool ConfirmationDialog::alert(const QString &prompt_text, QWidget *parent) {
  ConfirmationDialog d = ConfirmationDialog(prompt_text, "Ok", "", parent);
  return d.exec();
}

bool ConfirmationDialog::confirm(const QString &prompt_text, QWidget *parent) {
  ConfirmationDialog d = ConfirmationDialog(prompt_text, "Ok", "Cancel", parent);
  return d.exec();
}

int ConfirmationDialog::exec() {
   // TODO: make this work without fullscreen
  if (Hardware::TICI()) {
    setMainWindow(this);
  }
  return QDialog::exec();
}
