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

QTouchButton::QTouchButton(const QString &text, QWidget *parent) : QPushButton(text, parent) {

}

void QTouchButton::focusOutEvent(QFocusEvent *e) {
  if (e->reason() != Qt::MouseFocusReason) {
    QPushButton::focusOutEvent(e);
  }
}

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);

  QTouchButton *btn1 = new QTouchButton("Btn 1");
//  btn1->setAttribute(Qt::WA_AcceptTouchEvents);
  btn1->setObjectName("btn1");
  main_layout->addWidget(btn1);

  QTouchButton *btn2 = new QTouchButton("Btn 2");
//  btn2->setAttribute(Qt::WA_AcceptTouchEvents);
  btn2->setObjectName("btn2");
  main_layout->addWidget(btn2);

  QTouchButton *btn3 = new QTouchButton("Btn 3");
//  btn3->setAttribute(Qt::WA_AcceptTouchEvents);
  btn3->setObjectName("btn3");
  main_layout->addWidget(btn3);

  QTouchButton *btn4 = new QTouchButton("Btn 4");
//  btn4->setAttribute(Qt::WA_AcceptTouchEvents);
  btn4->setObjectName("btn4");
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

  // synthesize mouse events from touch events
  if (event->type() == QEvent::TouchBegin || event->type() == QEvent::TouchUpdate || event->type() == QEvent::TouchEnd || event->type() == QEvent::TouchCancel) {
    const QTouchEvent *touchEvent = static_cast<QTouchEvent *>(event);
    for (const QTouchEvent::TouchPoint &touchPoint : touchEvent->touchPoints()) {
      const QPointF &localPos(touchPoint.lastPos());
      const QPointF &screenPos(touchPoint.screenPos());
      Qt::MouseButton button = Qt::NoButton;
      Qt::MouseButtons buttons = Qt::LeftButton;

      if (touchPoint.state() == Qt::TouchPointPressed) {
        button = Qt::LeftButton;
        QMouseEvent mouseEvent(QEvent::MouseButtonPress, localPos, screenPos, button, buttons, Qt::NoModifier);
        QApplication::sendEvent(obj, &mouseEvent);
      } else if (touchPoint.state() == Qt::TouchPointReleased) {
        button = Qt::LeftButton;
        buttons = Qt::NoButton;
        QMouseEvent mouseEvent(QEvent::MouseButtonRelease, localPos, screenPos, button, buttons, Qt::NoModifier);
        QApplication::sendEvent(obj, &mouseEvent);
      } else if (touchPoint.state() == Qt::TouchPointMoved) {
        QMouseEvent mouseEvent(QEvent::MouseMove, localPos, screenPos, button, buttons, Qt::NoModifier);
        QApplication::sendEvent(obj, &mouseEvent);
      }
    }
    event->accept();
    return true;
  }
  return false;
}


int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);

  MainWindow window;
  a.installEventFilter(&window);
  setMainWindow(&window);

  a.setAttribute(Qt::AA_SynthesizeMouseForUnhandledTouchEvents, false);
  window.setAttribute(Qt::WA_AcceptTouchEvents);

  return a.exec();
}
