#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QWidget>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  QWidget window;
  setMainWindow(&window);

  QGridLayout *main_layout = new QGridLayout(&window);
  main_layout->setMargin(50);

  QLabel *label = new QLabel(argv[1]);
  label->setWordWrap(true);
  label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
  ScrollView *scroll = new ScrollView(label);
  scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  main_layout->addWidget(scroll, 0, 0, Qt::AlignTop);

  // Scroll to the bottom
  QObject::connect(scroll->verticalScrollBar(), &QAbstractSlider::rangeChanged, [=]() {
    scroll->verticalScrollBar()->setValue(scroll->verticalScrollBar()->maximum());
  });

  QHBoxLayout *button_layout = new QHBoxLayout();
  if (!Hardware::PC()) {
#ifdef QCOM
    QPushButton *wifiBtn = new QPushButton("Wi-Fi Settings");
    QObject::connect(wifiBtn, &ButtonControl::clicked, [=]() { 
      HardwareEon::launch_wifi(); 
    });
    button_layout->addWidget(wifiBtn);
#endif
    QPushButton *btn = new QPushButton("Reboot");
    QObject::connect(btn, &QPushButton::clicked, [=]() {
      Hardware::reboot();
    });
    button_layout->addWidget(btn);
  } else {
    QPushButton *btn = new QPushButton("Exit");
    QObject::connect(btn, &QPushButton::clicked, &a, &QApplication::quit);
    button_layout->addWidget(btn);
  }
  main_layout->addLayout(button_layout, 0, 0, Qt::AlignRight | Qt::AlignBottom);

  window.setStyleSheet(R"(
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

  return a.exec();
}
