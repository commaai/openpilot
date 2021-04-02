#include <QLabel>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#include "qt_window.hpp"
#include "selfdrive/hardware/hw.h"
#include "widgets/scrollview.hpp"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget window;
  setMainWindow(&window);

  QGridLayout *layout = new QGridLayout;
  layout->setMargin(100);

  // TODO: make this scroll
  QLabel *lab = new QLabel(argv[1]);
  lab->setWordWrap(true);
  ScrollView *scroll = new ScrollView(lab);
  scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  layout->addWidget(scroll, 0, 0, Qt::AlignTop);

  QPushButton *btn = new QPushButton();
#ifdef __aarch64__
  btn->setText("Reboot");
  QObject::connect(btn, &QPushButton::released, [=]() {
    Hardware::reboot();
  });
#else
  btn->setText("Exit");
  QObject::connect(btn, SIGNAL(released()), &a, SLOT(quit()));
#endif
  layout->addWidget(btn, 0, 0, Qt::AlignRight | Qt::AlignBottom);

  window.setLayout(layout);
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
      margin-right: 20px;
    }
  )");

  return a.exec();
}
