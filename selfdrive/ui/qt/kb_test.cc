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

//QTouchButton::QTouchButton(const QString &text, QWidget *parent) : QPushButton(text, parent) {
//
//}

//void QTouchButton::focusOutEvent(QFocusEvent *e) {
//  if (e->reason() != Qt::MouseFocusReason) {
//    QPushButton::focusOutEvent(e);
//  }
//}

//void QTouchButton::event(QEvent *event) {
//  if (ev->type() == QEvent::TouchBegin || ev->type() == QEvent::TouchEnd || ev->type() == QEvent::TouchUpdate) {
//    qDebug() << "TOUCH EVENT:" << ev;
//  }
//}

//void QTouchButton::touchEvent(QTouchEvent *ev) {
//  switch (ev->type()) {
//    case QEvent::TouchBegin:
//      qDebug() << "Touch begin for btn" << objectName();
//      break;
//    case QEvent::TouchEnd:
//      qDebug() << "Touch end for btn" << objectName();
//      break;
//    case QEvent::TouchUpdate:
//      qDebug() << "Touch update for btn" << objectName();
//      break;
//    default:
//      break;
//  }
//}

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *main_layout = new QHBoxLayout(this);

  for (int i = 0; i < 7; i++) {
    QPushButton *btn = new QPushButton(QString("Btn %1").arg(i));
    btn->setAttribute(Qt::WA_AcceptTouchEvents);
    btn->setObjectName(QString("btn%1").arg(i));
    btn->installEventFilter(this);
    btn->setFocusPolicy(Qt::NoFocus);
    QObject::connect(btn, &QPushButton::clicked, this, [=]() { qDebug() << QString("Btn %1 clicked!").arg(i); });
    main_layout->addWidget(btn);
  }

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
      QEvent::Type mouseEventType = QEvent::MouseMove;
      Qt::MouseButton button = Qt::NoButton;
      Qt::MouseButtons buttons = Qt::LeftButton;

      if (touchPoint.state() == Qt::TouchPointPressed) {
        mouseEventType = QEvent::MouseButtonPress;
        button = Qt::LeftButton;
      } else if (touchPoint.state() == Qt::TouchPointReleased) {
        mouseEventType = QEvent::MouseButtonRelease;
        button = Qt::LeftButton;
        buttons = Qt::NoButton;
      }
      QMouseEvent mouseEvent(mouseEventType, localPos, screenPos, button, buttons, Qt::NoModifier);
      QApplication::sendEvent(obj, &mouseEvent);
    }
    event->accept();
    return true;
  }
  return QWidget::eventFilter(obj, event);
}


int main(int argc, char *argv[]) {
  initApp();
  QApplication a(argc, argv);

  MainWindow window;
//  a.installEventFilter(&window);
  setMainWindow(&window);

  a.setAttribute(Qt::AA_SynthesizeMouseForUnhandledTouchEvents, false);
//  window.setAttribute(Qt::WA_AcceptTouchEvents);

  return a.exec();
}
