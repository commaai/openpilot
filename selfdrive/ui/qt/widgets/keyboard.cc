#include <QDebug>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QButtonGroup>

#include "keyboard.hpp"

const int DEFAULT_STRETCH = 1;
const int SPACEBAR_STRETCH = 3;


template <size_t rows, size_t cols>
static QWidget *keyboardWidget(QWidget *parent, QString (&layout)[rows][cols]) {
  QWidget *widget = new QWidget(parent);
  QVBoxLayout* vlayout = new QVBoxLayout;
  vlayout->setMargin(0);
  vlayout->setSpacing(35);

  QButtonGroup* btn_group = new QButtonGroup(widget);
  QObject::connect(btn_group, SIGNAL(buttonClicked(QAbstractButton*)), parent, SLOT(handleButton(QAbstractButton*)));

  for (int i = 0; i < rows; ++i) {
    QHBoxLayout *hlayout = new QHBoxLayout;
    hlayout->setSpacing(25);

    if (vlayout->count() == 1) {
      hlayout->addSpacing(90);
    }

    for (int j = 0; j < cols; ++j) {
      const QString &c  = layout[i][j];
      if (c.isEmpty()) break;

      QPushButton* btn = new QPushButton(c);
      btn->setFixedHeight(135);
      btn_group->addButton(btn);
      hlayout->addWidget(btn, c == "  " ? SPACEBAR_STRETCH : DEFAULT_STRETCH);
    }

    if (vlayout->count() == 1) {
      hlayout->addSpacing(90);
    }

    vlayout->addLayout(hlayout);
  }

  widget->setStyleSheet(R"(
    * {
      outline: none;
    }
    QPushButton {
      font-size: 65px;
      margin: 0px;
      padding: 0px;
      border-radius: 30px;
      color: #dddddd;
      background-color: #444444;
    }
    QPushButton:pressed {
      background-color: #000000;
    }
  )");
  widget->setLayout(vlayout);
  return widget;
}

// lowercase
static QString lowercase[][10] = {
    {"q","w","e","r","t","y","u","i","o","p"},
    {"a","s","d","f","g","h","j","k","l"},
    {"⇧","z","x","c","v","b","n","m","⌫"},
    {"123","  ","⏎"},
};

// uppercase
static QString uppercase[][10] = {
    {"Q","W","E","R","T","Y","U","I","O","P"},
    {"A","S","D","F","G","H","J","K","L"},
    {"↑","Z","X","C","V","B","N","M","⌫"},
    {"123","  ","⏎"},
};

// numbers + specials
static QString numbers[][10] = {
    {"1","2","3","4","5","6","7","8","9","0"},
    {"-","/",":",";","(",")","$","&&","@","\""},
    {"#+=",".",",","?","!","`","⌫"},
    {"ABC","  ","⏎"},
};

// extra specials
static QString specials[][10] = {
    {"[","]","{","}","#","%","^","*","+","="},
    {"_","\\","|","~","<",">","€","£","¥","•"},
    {"123",".",",","?","!","`","⌫"},
    {"ABC","  ","⏎"},
};

Keyboard::Keyboard(QWidget *parent) : QFrame(parent) {
  main_layout = new QStackedLayout;
  main_layout->setMargin(0);

  main_layout->addWidget(keyboardWidget(this, lowercase));
  main_layout->addWidget(keyboardWidget(this, uppercase));
  main_layout->addWidget(keyboardWidget(this, numbers));
  main_layout->addWidget(keyboardWidget(this, specials));

  setLayout(main_layout);
  main_layout->setCurrentIndex(0);
}

void Keyboard::handleButton(QAbstractButton* m_button) {
  QString id = m_button->text();
  if (!QString::compare(m_button->text(), "↑") || !QString::compare(m_button->text(), "ABC")) {
    main_layout->setCurrentIndex(0);
  }
  if (!QString::compare(m_button->text(), "⇧")) {
    main_layout->setCurrentIndex(1);
  }
  if (!QString::compare(m_button->text(), "123")) {
    main_layout->setCurrentIndex(2);
  }
  if (!QString::compare(m_button->text(), "#+=")) {
    main_layout->setCurrentIndex(3);
  }
  if (!QString::compare(m_button->text(), "⏎")) {
    main_layout->setCurrentIndex(0);
  }
  if ("A" <= id && id <= "Z") {
    main_layout->setCurrentIndex(0);
  }
  emit emitButton(m_button->text());
}
