#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QWidget>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget window;
  setMainWindow(&window);

  Hardware::set_display_power(true);
  Hardware::set_brightness(65);

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

  QPushButton *btn = new QPushButton();
#ifdef __aarch64__
  btn->setText("Reboot");
  QObject::connect(btn, &QPushButton::released, [=]() {
    Hardware::reboot();
  });
#else
  btn->setText("Exit");
  QObject::connect(btn, &QPushButton::released, &a, &QApplication::quit);
#endif
  main_layout->addWidget(btn, 0, 0, Qt::AlignRight | Qt::AlignBottom);

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
