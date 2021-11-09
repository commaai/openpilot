#include "selfdrive/ui/qt/widgets/keyboard.h"

#include <vector>

#include <QButtonGroup>
#include <QHBoxLayout>
#include <QMap>
#include <QTouchEvent>
#include <QVBoxLayout>

const QString BACKSPACE_KEY = "⌫";
const QString ENTER_KEY = "→";

const QMap<QString, int> KEY_STRETCH = {{"  ", 5}, {ENTER_KEY, 2}};

const QStringList CONTROL_BUTTONS = {"↑", "↓", "ABC", "#+=", "123", BACKSPACE_KEY, ENTER_KEY};

const float key_spacing_vertical = 20;
const float key_spacing_horizontal = 15;

KeyButton::KeyButton(const QString &text, QWidget *parent) : QPushButton(text, parent) {
  setAttribute(Qt::WA_AcceptTouchEvents);
  setFocusPolicy(Qt::NoFocus);
}

bool KeyButton::event(QEvent *event) {
  if (event->type() == QEvent::TouchBegin || event->type() == QEvent::TouchEnd) {
    QTouchEvent *touchEvent = static_cast<QTouchEvent *>(event);
    if (!touchEvent->touchPoints().empty()) {
      const QEvent::Type mouseType = event->type() == QEvent::TouchBegin ? QEvent::MouseButtonPress : QEvent::MouseButtonRelease;
      QMouseEvent mouseEvent(mouseType, touchEvent->touchPoints().front().pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
      QPushButton::event(&mouseEvent);
      event->accept();
      parentWidget()->update();
      return true;
    }
  }
  return QPushButton::event(event);
}

KeyboardLayout::KeyboardLayout(QWidget* parent, const std::vector<QVector<QString>>& layout) : QWidget(parent) {
  QVBoxLayout* main_layout = new QVBoxLayout(this);
  main_layout->setMargin(0);
  main_layout->setSpacing(0);

  QButtonGroup* btn_group = new QButtonGroup(this);
  QObject::connect(btn_group, SIGNAL(buttonClicked(QAbstractButton*)), parent, SLOT(handleButton(QAbstractButton*)));

  for (const auto &s : layout) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->setSpacing(0);

    if (main_layout->count() == 1) {
      hlayout->addSpacing(90);
    }

    for (const QString &p : s) {
      KeyButton* btn = new KeyButton(p);
      if (p == BACKSPACE_KEY) {
        btn->setAutoRepeat(true);
      } else if (p == ENTER_KEY) {
        btn->setStyleSheet("background-color: #465BEA;");
      }
      btn->setFixedHeight(135 + key_spacing_vertical);
      btn_group->addButton(btn);
      hlayout->addWidget(btn, KEY_STRETCH.value(p, 1));
    }

    if (main_layout->count() == 1) {
      hlayout->addSpacing(90);
    }

    main_layout->addLayout(hlayout);
  }

  setStyleSheet(QString(R"(
    QPushButton {
      font-size: 75px;
      margin-left: %1px;
      margin-right: %1px;
      margin-top: %2px;
      margin-bottom: %2px;
      padding: 0px;
      border-radius: 10px;
      color: #dddddd;
      background-color: #444444;
    }
    QPushButton:pressed {
      background-color: #333333;
    }
  )").arg(key_spacing_vertical / 2).arg(key_spacing_horizontal / 2));
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
  const QString &key = btn->text();
  if (CONTROL_BUTTONS.contains(key)) {
    if (key == "↓" || key == "ABC") {
      main_layout->setCurrentIndex(0);
    } else if (key == "↑") {
      main_layout->setCurrentIndex(1);
    } else if (key == "123") {
      main_layout->setCurrentIndex(2);
    } else if (key == "#+=") {
      main_layout->setCurrentIndex(3);
    } else if (key == ENTER_KEY) {
      main_layout->setCurrentIndex(0);
      emit emitEnter();
    } else if (key == BACKSPACE_KEY) {
      emit emitBackspace();
    }
  } else {
    if ("A" <= key && key <= "Z") {
      main_layout->setCurrentIndex(0);
    }
    emit emitKey(key);
  }
}
