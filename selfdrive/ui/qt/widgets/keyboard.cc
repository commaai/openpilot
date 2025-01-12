#include "selfdrive/ui/qt/widgets/keyboard.h"

#include <vector>
#include <QDateTime>

#include <QButtonGroup>
#include <QHBoxLayout>
#include <QMap>
#include <QTouchEvent>
#include <QVBoxLayout>

const QString BACKSPACE_KEY = "⌫";
const QString ENTER_KEY = "→";
const QString CAPS_KEY = "⇧";
const QString CAPS_LOCK_KEY = "⇪";

const QMap<QString, int> KEY_STRETCH = {{"  ", 3}, {ENTER_KEY, 2}};

const QStringList CONTROL_BUTTONS = {CAPS_KEY, CAPS_LOCK_KEY, "ABC", "#+=", "123", BACKSPACE_KEY, ENTER_KEY};

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
        btn->setStyleSheet(R"(
          QPushButton {
            background-color: #465BEA;
          }
          QPushButton:pressed {
            background-color: #444444;
          }
        )");
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
    {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"},
    {"a", "s", "d", "f", "g", "h", "j", "k", "l"},
    {CAPS_KEY, "z", "x", "c", "v", "b", "n", "m", BACKSPACE_KEY},
    {"123", "/", "-", "  ", ".", ENTER_KEY},
  };
  main_layout->addWidget(new KeyboardLayout(this, lowercase));

  // uppercase
  std::vector<QVector<QString>> uppercase = {
    {"Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"},
    {"A", "S", "D", "F", "G", "H", "J", "K", "L"},
    {CAPS_KEY, "Z", "X", "C", "V", "B", "N", "M", BACKSPACE_KEY},
    {"123", "/", "-", "  ", ".", ENTER_KEY},
  };
  main_layout->addWidget(new KeyboardLayout(this, uppercase));

  // numbers + specials
  std::vector<QVector<QString>> numbers = {
    {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"},
    {"-", "/", ":", ";", "(", ")", "$", "&&", "@", "\""},
    {"#+=", ".", ",", "?", "!", "`", BACKSPACE_KEY},
    {"ABC", "  ", ".", ENTER_KEY},
  };
  main_layout->addWidget(new KeyboardLayout(this, numbers));

  // extra specials
  std::vector<QVector<QString>> specials = {
    {"[", "]", "{", "}", "#", "%", "^", "*", "+", "="},
    {"_", "\\", "|", "~", "<", ">", "€", "£", "¥", "•"},
    {"123", ".", ",", "?", "!", "'", BACKSPACE_KEY},
    {"ABC", "  ", ".", ENTER_KEY},
  };
  main_layout->addWidget(new KeyboardLayout(this, specials));

  main_layout->setCurrentIndex(0);
}

void Keyboard::handleCapsPress() {
  qint64 current_time = QDateTime::currentMSecsSinceEpoch();
  bool is_double_tap = (current_time - last_caps_press) <= DOUBLE_TAP_INTERVAL_MS;
  last_caps_press = current_time;

  if (caps_locked) {
    // If caps locked, a single tap disables it and returns to lowercase
    caps_locked = false;
    main_layout->setCurrentIndex(0);
  } else if (is_double_tap) {
    // Double tap when not caps locked enables caps lock
    caps_locked = true;
    main_layout->setCurrentIndex(1);
  } else {
    // Single tap when not caps locked just toggles case
    main_layout->setCurrentIndex(main_layout->currentIndex() == 0 ? 1 : 0);
  }

  // Update caps button
  QWidget* current_layout = main_layout->currentWidget();
  if (current_layout) {
    QList<KeyButton*> buttons = current_layout->findChildren<KeyButton*>();
    for (KeyButton* btn : buttons) {
      if (btn->text() == CAPS_KEY || btn->text() == CAPS_LOCK_KEY) {
        btn->setText(caps_locked ? CAPS_LOCK_KEY : CAPS_KEY);
        btn->setStyleSheet(caps_locked || main_layout->currentIndex() == 1 ?
                          "background-color: #465BEA;" : "");
      }
    }
  }
}

void Keyboard::handleButton(QAbstractButton* btn) {
  const QString &key = btn->text();
  if (CONTROL_BUTTONS.contains(key)) {
    if (key == "ABC") {
      main_layout->setCurrentIndex(0);  // Always go to lowercase
      caps_locked = false;  // Reset caps lock state
    } else if (key == CAPS_KEY || key == CAPS_LOCK_KEY) {
      handleCapsPress();
    } else if (key == "123") {
      main_layout->setCurrentIndex(2);
    } else if (key == "#+=") {
      main_layout->setCurrentIndex(3);
    } else if (key == ENTER_KEY) {
      emit emitEnter();
    } else if (key == BACKSPACE_KEY) {
      emit emitBackspace();
    }
  } else {
    // Only switch to lowercase after typing an uppercase letter if not caps locked
    if ("A" <= key && key <= "Z" && !caps_locked) {
      main_layout->setCurrentIndex(0);
    }
    emit emitKey(key);
  }
}
