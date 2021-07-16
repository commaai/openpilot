#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QHBoxLayout>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  QWidget window;
  setMainWindow(&window);

  window.setAttribute(Qt::WA_AcceptTouchEvents);

  QHBoxLayout *main_layout = new QHBoxLayout(&window);

  QPushButton *btn1 = new QPushButton("Btn 1");
  btn1->setAttribute(Qt::WA_AcceptTouchEvents);
  main_layout->addWidget(btn1);

  QPushButton *btn2 = new QPushButton("Btn 2");
  btn2->setAttribute(Qt::WA_AcceptTouchEvents);
  main_layout->addWidget(btn2);

  QPushButton *btn3 = new QPushButton("Btn 3");
  btn3->setAttribute(Qt::WA_AcceptTouchEvents);
  main_layout->addWidget(btn3);

  QPushButton *btn4 = new QPushButton("Btn 4");
  btn4->setAttribute(Qt::WA_AcceptTouchEvents);
  main_layout->addWidget(btn4);

  window.setStyleSheet(R"(
    * {
      outline: none;
      color: white;
      background-color: black;
    }
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

//  QList<QAbstractSlider *> sliders = window.findChildren<QAbstractSlider *>();
//     foreach (QAbstractSlider *slider, sliders)

  return a.exec();
}
