#include "selfdrive/ui/qt/widgets/dialog.h"

#include <QPushButton>
#include <QButtonGroup>

#include "selfdrive/ui/qt/util.h"
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
  ConfirmationDialog d = ConfirmationDialog(prompt_text, tr("Ok"), "", false, parent);
  return d.exec();
}

bool ConfirmationDialog::confirm(const QString &prompt_text, const QString &confirm_text, QWidget *parent) {
  ConfirmationDialog d = ConfirmationDialog(prompt_text, confirm_text, tr("Cancel"), false, parent);
  return d.exec();
}

bool ConfirmationDialog::rich(const QString &prompt_text, QWidget *parent) {
  ConfirmationDialog d = ConfirmationDialog(prompt_text, tr("Ok"), "", true, parent);
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
  MultiOptionDialog d = MultiOptionDialog(prompt_text, l, current, parent);
  if (d.exec()) {
    return d.selection;
  }
  return "";
}
