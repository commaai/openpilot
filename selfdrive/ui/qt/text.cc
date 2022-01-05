#include <QApplication>
#include <QScrollBar>
#include <QVBoxLayout>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/text.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"


Text::Text(char *argv[], QApplication &a, QWidget *parent) : Wakeable(), QWidget(parent) {
  QGridLayout *main_layout = new QGridLayout(this);
  main_layout->setMargin(50);

  label = new QLabel(argv[1]);
  label->setWordWrap(true);
  label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
  scroll = new ScrollView(label);
  scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  main_layout->addWidget(scroll, 0, 0, Qt::AlignTop);

  // Scroll to the bottom
  QObject::connect(scroll->verticalScrollBar(), &QAbstractSlider::rangeChanged, [=]() {
    scroll->verticalScrollBar()->setValue(scroll->verticalScrollBar()->maximum());
  });

  btn = new QPushButton();
#ifdef __aarch64__
  btn->setText("Reboot");
  QObject::connect(btn, &QPushButton::clicked, [=]() {
    Hardware::reboot();
  });
#else
  btn->setText("Exit");
  QObject::connect(btn, &QPushButton::clicked, &a, &QApplication::quit);
#endif
  main_layout->addWidget(btn, 0, 0, Qt::AlignRight | Qt::AlignBottom);

  setStyleSheet(R"(
    * {
      outline: none;
      color: white;
      background-color: black;
      font-size: 60px;
    }
    QPushButton {
      padding: 50px;
      padding-right: 100px;
      padding-left: 100px;
      border: 2px solid white;
      border-radius: 20px;
      margin-right: 40px;
    }
  )");

  // Connect text signal directly to UI state signal for awake boolean
  QObject::connect(this, &Text::displayPowerChanged, uiState(), &UIState::displayPowerChanged);

  // Connect to UI state messages for brightness and wakefulness
  QObject::connect(uiState(), &UIState::uiUpdate, this, &Text::update);

  setAwake(true);
  resetInteractiveTimeout();
};

void Text::update(const UIState &s) {
  Wakeable::update(s);
}

int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  Text text(argv, a);
  setMainWindow(&text);
  return a.exec();
}
