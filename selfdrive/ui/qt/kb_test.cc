#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QHBoxLayout>
#include <QMainWindow>
#include <QDebug>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"
#include "selfdrive/ui/qt/kb_test.h"

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);
//  installEventFilter(this);

  QPushButton *btn1 = new QPushButton("Btn 1");
  btn1->setAttribute(Qt::WA_AcceptTouchEvents);
//  btn1->installEventFilter(this);
  main_layout->addWidget(btn1);

  QPushButton *btn2 = new QPushButton("Btn 2");
  btn2->setAttribute(Qt::WA_AcceptTouchEvents);
//  btn2->installEventFilter(this);
  main_layout->addWidget(btn2);

  QPushButton *btn3 = new QPushButton("Btn 3");
  btn3->setAttribute(Qt::WA_AcceptTouchEvents);
//  btn3->installEventFilter(this);
  main_layout->addWidget(btn3);

  QPushButton *btn4 = new QPushButton("Btn 4");
  btn4->setAttribute(Qt::WA_AcceptTouchEvents);
//  btn4->installEventFilter(this);
  main_layout->addWidget(btn4);

  setStyleSheet(R"(
    * {
      outline: none;
      color: white;
      background-color: black;
    }
    QPushButton {
      height: 135px;
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

bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
  qDebug() << "EVENT:" << event;
//  qDebug() << "EVENT type:" << event->type();
//  if (event->type() == QEvent::TouchBegin) {
  if (event->type() == QEvent::TouchBegin) {
    QTouchEvent *touchEvent = static_cast<QTouchEvent *>(event);
    QPointF pos;
    for (const auto tp : touchEvent->touchPoints()) {
      pos.setX(tp.startPos().x());
      pos.setY(tp.startPos().y());
      break;
    }
    Qt::MouseButtons buttons;
    Qt::KeyboardModifiers modifiers;
    qDebug() << "Sent mouse event with pos:" << pos;
    QMouseEvent eventNew(QEvent::MouseButtonPress, pos, Qt::LeftButton, buttons, modifiers);  // TODO fix screen pos
//    QApplication::sendEvent(obj, &eventNew);
    return QWidget::eventFilter(obj, event);
//    return true;
//  } else if (event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseMove || event->type() == QEvent::HoverMove) {
//    event->ignore();
//    return true;
  } else {
    return QWidget::eventFilter(obj, event);
  }
}


int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);
  MainWindow window;
  setMainWindow(&window);
  a.installEventFilter(&window);

  window.setAttribute(Qt::WA_AcceptTouchEvents);
  window.setAttribute(Qt::AA_SynthesizeMouseForUnhandledTouchEvents, false);


//  QList<QAbstractSlider *> sliders = window.findChildren<QAbstractSlider *>();
//     foreach (QAbstractSlider *slider, sliders)

  return a.exec();
}
