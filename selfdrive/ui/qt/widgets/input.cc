#include "selfdrive/ui/qt/widgets/input.h"

#include <QPushButton>
#include <QButtonGroup>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"


DialogBase::DialogBase(QWidget *parent) : QDialog(parent) {
  Q_ASSERT(parent != nullptr);
  parent->installEventFilter(this);

  setStyleSheet(R"(
    * {
      outline: none;
      color: white;
      font-family: Inter;
    }
    DialogBase {
      background-color: black;
    }
    QPushButton {
      height: 160;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      color: white;
      background-color: #333333;
    }
    QPushButton:pressed {
      background-color: #444444;
    }
  )");
}

bool DialogBase::eventFilter(QObject *o, QEvent *e) {
  if (o == parent() && e->type() == QEvent::Hide) {
    reject();
  }
  return QDialog::eventFilter(o, e);
}

int DialogBase::exec() {
  setMainWindow(this);
  return QDialog::exec();
}

InputDialog::InputDialog(const QString &title, QWidget *parent, const QString &subtitle, bool secret) : DialogBase(parent) {
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

  QPushButton* cancel_btn = new QPushButton(tr("Cancel"));
  cancel_btn->setFixedSize(386, 125);
  cancel_btn->setStyleSheet(R"(
    QPushButton {
      font-size: 48px;
      border-radius: 10px;
      color: #E4E4E4;
      background-color: #333333;
    }
    QPushButton:pressed {
      background-color: #444444;
    }
  )");
  header_layout->addWidget(cancel_btn, 0, Qt::AlignRight);
  QObject::connect(cancel_btn, &QPushButton::clicked, this, &InputDialog::reject);
  QObject::connect(cancel_btn, &QPushButton::clicked, this, &InputDialog::cancel);

  main_layout->addLayout(header_layout);

  // text box
  main_layout->addStretch(2);

  QWidget *textbox_widget = new QWidget;
  textbox_widget->setObjectName("textbox");
  QHBoxLayout *textbox_layout = new QHBoxLayout(textbox_widget);
  textbox_layout->setContentsMargins(50, 0, 50, 0);

  textbox_widget->setStyleSheet(R"(
    #textbox {
      margin-left: 50px;
      margin-right: 50px;
      border-radius: 0;
      border-bottom: 3px solid #BDBDBD;
    }
    * {
      border: none;
      font-size: 80px;
      font-weight: light;
      background-color: transparent;
    }
  )");

  line = new QLineEdit();
  line->setStyleSheet("lineedit-password-character: 8226; lineedit-password-mask-delay: 1500;");
  textbox_layout->addWidget(line, 1);

  if (secret) {
    eye_btn = new QPushButton();
    eye_btn->setCheckable(true);
    eye_btn->setFixedSize(150, 120);
    QObject::connect(eye_btn, &QPushButton::toggled, [=](bool checked) {
      if (checked) {
        eye_btn->setIcon(QIcon(ASSET_PATH + "img_eye_closed.svg"));
        eye_btn->setIconSize(QSize(81, 54));
        line->setEchoMode(QLineEdit::Password);
      } else {
        eye_btn->setIcon(QIcon(ASSET_PATH + "img_eye_open.svg"));
        eye_btn->setIconSize(QSize(81, 44));
        line->setEchoMode(QLineEdit::Normal);
      }
    });
    eye_btn->toggle();
    eye_btn->setChecked(false);
    textbox_layout->addWidget(eye_btn);
  }

  main_layout->addWidget(textbox_widget, 0, Qt::AlignBottom);
  main_layout->addSpacing(25);

  k = new Keyboard(this);
  QObject::connect(k, &Keyboard::emitEnter, this, &InputDialog::handleEnter);
  QObject::connect(k, &Keyboard::emitBackspace, this, [=]() {
    line->backspace();
  });
  QObject::connect(k, &Keyboard::emitKey, this, [=](const QString &key) {
    line->insert(key.left(1));
  });

  main_layout->addWidget(k, 2, Qt::AlignBottom);
}

QString InputDialog::getText(const QString &prompt, QWidget *parent, const QString &subtitle,
                             bool secret, int minLength, const QString &defaultText) {
  InputDialog d(prompt, parent, subtitle, secret);
  d.line->setText(defaultText);
  d.setMinLength(minLength);
  const int ret = d.exec();
  return ret ? d.text() : QString();
}

QString InputDialog::text() {
  return line->text();
}

void InputDialog::show() {
  setMainWindow(this);
}

void InputDialog::handleEnter() {
  if (line->text().length() >= minLength) {
    done(QDialog::Accepted);
    emitText(line->text());
  } else {
    setMessage(tr("Need at least %n character(s)!", "", minLength), false);
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

// ConfirmationDialog

ConfirmationDialog::ConfirmationDialog(const QString &prompt_text, const QString &confirm_text, const QString &cancel_text,
                                       const bool rich, QWidget *parent) : DialogBase(parent) {
  QFrame *container = new QFrame(this);
  container->setStyleSheet(R"(
    QFrame { background-color: #1B1B1B; color: #C9C9C9; }
    #confirm_btn { background-color: #465BEA; }
    #confirm_btn:pressed { background-color: #3049F4; }
  )");
  QVBoxLayout *main_layout = new QVBoxLayout(container);
  main_layout->setContentsMargins(32, rich ? 32 : 120, 32, 32);

  QLabel *prompt = new QLabel(prompt_text, this);
  prompt->setWordWrap(true);
  prompt->setAlignment(rich ? Qt::AlignLeft : Qt::AlignHCenter);
  prompt->setStyleSheet((rich ? "font-size: 42px; font-weight: light;" : "font-size: 70px; font-weight: bold;") + QString(" margin: 45px;"));
  main_layout->addWidget(rich ? (QWidget*)new ScrollView(prompt, this) : (QWidget*)prompt, 1, Qt::AlignTop);

  // cancel + confirm buttons
  QHBoxLayout *btn_layout = new QHBoxLayout();
  btn_layout->setSpacing(30);
  main_layout->addLayout(btn_layout);

  if (cancel_text.length()) {
    QPushButton* cancel_btn = new QPushButton(cancel_text);
    btn_layout->addWidget(cancel_btn);
    QObject::connect(cancel_btn, &QPushButton::clicked, this, &ConfirmationDialog::reject);
  }

  if (confirm_text.length()) {
    QPushButton* confirm_btn = new QPushButton(confirm_text);
    confirm_btn->setObjectName("confirm_btn");
    btn_layout->addWidget(confirm_btn);
    QObject::connect(confirm_btn, &QPushButton::clicked, this, &ConfirmationDialog::accept);
  }

  QVBoxLayout *outer_layout = new QVBoxLayout(this);
  int margin = rich ? 100 : 200;
  outer_layout->setContentsMargins(margin, margin, margin, margin);
  outer_layout->addWidget(container);
}

bool ConfirmationDialog::alert(const QString &prompt_text, QWidget *parent) {
  ConfirmationDialog d(prompt_text, tr("Ok"), "", false, parent);
  return d.exec();
}

bool ConfirmationDialog::confirm(const QString &prompt_text, const QString &confirm_text, QWidget *parent) {
  ConfirmationDialog d(prompt_text, confirm_text, tr("Cancel"), false, parent);
  return d.exec();
}

bool ConfirmationDialog::rich(const QString &prompt_text, QWidget *parent) {
  ConfirmationDialog d(prompt_text, tr("Ok"), "", true, parent);
  return d.exec();
}

// MultiOptionDialog

MultiOptionDialog::MultiOptionDialog(const QString &prompt_text, const QStringList &l, const QString &current, QWidget *parent) : DialogBase(parent) {
  QFrame *container = new QFrame(this);
  container->setStyleSheet(R"(
    QFrame { background-color: #1B1B1B; }
    #confirm_btn[enabled="false"] { background-color: #2B2B2B; }
    #confirm_btn:enabled { background-color: #465BEA; }
    #confirm_btn:enabled:pressed { background-color: #3049F4; }
  )");

  QVBoxLayout *main_layout = new QVBoxLayout(container);
  main_layout->setContentsMargins(55, 50, 55, 50);

  QLabel *title = new QLabel(prompt_text, this);
  title->setStyleSheet("font-size: 70px; font-weight: 500;");
  main_layout->addWidget(title, 0, Qt::AlignLeft | Qt::AlignTop);
  main_layout->addSpacing(25);

  QWidget *listWidget = new QWidget(this);
  QVBoxLayout *listLayout = new QVBoxLayout(listWidget);
  listLayout->setSpacing(20);
  listWidget->setStyleSheet(R"(
    QPushButton {
      height: 135;
      padding: 0px 50px;
      text-align: left;
      font-size: 55px;
      font-weight: 300;
      border-radius: 10px;
      background-color: #4F4F4F;
    }
    QPushButton:checked { background-color: #465BEA; }
  )");

  QButtonGroup *group = new QButtonGroup(listWidget);
  group->setExclusive(true);

  QPushButton *confirm_btn = new QPushButton(tr("Select"));
  confirm_btn->setObjectName("confirm_btn");
  confirm_btn->setEnabled(false);

  for (const QString &s : l) {
    QPushButton *selectionLabel = new QPushButton(s);
    selectionLabel->setCheckable(true);
    selectionLabel->setChecked(s == current);
    QObject::connect(selectionLabel, &QPushButton::toggled, [=](bool checked) {
      if (checked) selection = s;
      if (selection != current) {
        confirm_btn->setEnabled(true);
      } else {
        confirm_btn->setEnabled(false);
      }
    });

    group->addButton(selectionLabel);
    listLayout->addWidget(selectionLabel);
  }
  // add stretch to keep buttons spaced correctly
  listLayout->addStretch(1);

  ScrollView *scroll_view = new ScrollView(listWidget, this);
  scroll_view->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

  main_layout->addWidget(scroll_view);
  main_layout->addSpacing(35);

  // cancel + confirm buttons
  QHBoxLayout *blayout = new QHBoxLayout;
  main_layout->addLayout(blayout);
  blayout->setSpacing(50);

  QPushButton *cancel_btn = new QPushButton(tr("Cancel"));
  QObject::connect(cancel_btn, &QPushButton::clicked, this, &ConfirmationDialog::reject);
  QObject::connect(confirm_btn, &QPushButton::clicked, this, &ConfirmationDialog::accept);
  blayout->addWidget(cancel_btn);
  blayout->addWidget(confirm_btn);

  QVBoxLayout *outer_layout = new QVBoxLayout(this);
  outer_layout->setContentsMargins(50, 50, 50, 50);
  outer_layout->addWidget(container);
}

QString MultiOptionDialog::getSelection(const QString &prompt_text, const QStringList &l, const QString &current, QWidget *parent) {
  MultiOptionDialog d(prompt_text, l, current, parent);
  if (d.exec()) {
    return d.selection;
  }
  return "";
}
