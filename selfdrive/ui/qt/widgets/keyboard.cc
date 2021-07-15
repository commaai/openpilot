#include "selfdrive/ui/qt/widgets/keyboard.h"

#include <QButtonGroup>
#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>

const int DEFAULT_STRETCH = 1;
const int SPACEBAR_STRETCH = 3;

const QString BACKSPACE_KEY = "⌫";
const QString ENTER_KEY = "⏎";

const QStringList CONTROL_BUTTONS = {"↑", "↓", "ABC", "#+=", "123"};

KeyboardLayout::KeyboardLayout(QWidget* parent, const std::vector<QVector<QString>>& layout) : QWidget(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setMargin(0);
  main_layout->setSpacing(20);

  QButtonGroup* btn_group = new QButtonGroup(this);
  QObject::connect(btn_group, SIGNAL(buttonClicked(QAbstractButton*)), parent, SLOT(handleButton(QAbstractButton*)));

  for (const auto &s : layout) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->setSpacing(15);

    if (main_layout->count() == 1) {
      hlayout->addSpacing(90);
    }

    for (const QString &p : s) {
      QPushButton* btn = new QPushButton(p);
      if (p == BACKSPACE_KEY) {
        btn->setAutoRepeat(true);
      } else if (p == ENTER_KEY) {
        btn->setStyleSheet("background-color: #465BEA;");
      }
      btn->setFixedHeight(135);
      btn_group->addButton(btn);
      hlayout->addWidget(btn, p == QString("  ") ? SPACEBAR_STRETCH : DEFAULT_STRETCH);
    }

    if (main_layout->count() == 1) {
      hlayout->addSpacing(90);
    }

    main_layout->addLayout(hlayout);
  }

  setStyleSheet(R"(
    QPushButton {
      font-size: 75px;
      margin: 0px;
      padding: 0px;
      border-radius: 10px;
      color: #dddddd;
      background-color: #444444;
    }
    QPushButton:pressed {
      background-color: #333333;
    }
  )");
}

Keyboard::Keyboard(QWidget *parent) : QFrame(parent) {
  main_layout = new QStackedLayout(this);
  main_layout->setMargin(0);

  // lowercase
  std::vector<QVector<QString>> lowercase = {
    {"q","w","e","r","t","y","u","i","o","p"},
    {"a","s","d","f","g","h","j","k","l"},
    {"↑","z","x","c","v","b","n","m",BACKSPACE_KEY},
    {"123","  ",".",ENTER_KEY},
  };
  main_layout->addWidget(new KeyboardLayout(this, lowercase));

  // uppercase
  std::vector<QVector<QString>> uppercase = {
    {"Q","W","E","R","T","Y","U","I","O","P"},
    {"A","S","D","F","G","H","J","K","L"},
    {"↓","Z","X","C","V","B","N","M",BACKSPACE_KEY},
    {"123","  ",".",ENTER_KEY},
  };
  main_layout->addWidget(new KeyboardLayout(this, uppercase));

  // numbers + specials
  std::vector<QVector<QString>> numbers = {
    {"1","2","3","4","5","6","7","8","9","0"},
    {"-","/",":",";","(",")","$","&&","@","\""},
    {"#+=",".",",","?","!","`",BACKSPACE_KEY},
    {"ABC","  ",".",ENTER_KEY},
  };
  main_layout->addWidget(new KeyboardLayout(this, numbers));

  // extra specials
  std::vector<QVector<QString>> specials = {
    {"[","]","{","}","#","%","^","*","+","="},
    {"_","\\","|","~","<",">","€","£","¥","•"},
    {"123",".",",","?","!","`",BACKSPACE_KEY},
    {"ABC","  ",".",ENTER_KEY},
  };
  main_layout->addWidget(new KeyboardLayout(this, specials));

  main_layout->setCurrentIndex(0);
}

void Keyboard::handleButton(QAbstractButton* btn) {
  const QString key = btn->text();
  if (!QString::compare(key, "↓") || !QString::compare(key, "ABC")) {
    main_layout->setCurrentIndex(0);
  }
  if (!QString::compare(key, "↑")) {
    main_layout->setCurrentIndex(1);
  }
  if (!QString::compare(key, "123")) {
    main_layout->setCurrentIndex(2);
  }
  if (!QString::compare(key, "#+=")) {
    main_layout->setCurrentIndex(3);
  }
  if (!QString::compare(key, ENTER_KEY)) {
    main_layout->setCurrentIndex(0);
  }
  if ("A" <= key && key <= "Z") {
    main_layout->setCurrentIndex(0);
  }

  // TODO: break up into separate signals
  if (!CONTROL_BUTTONS.contains(key)) {
    emit emitButton(key);
  }
}
