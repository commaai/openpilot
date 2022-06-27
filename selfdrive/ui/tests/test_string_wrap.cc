#include <QEventLoop>
#include <QMap>
#include <QThread>

#include "catch2/catch.hpp"

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"
#include <QDebug>
#include <QTimer>
#include <QObject>
#include <QWidget>
#include <QPushButton>
#include <QTranslator>

const QString TEST_TEXT = "SOURCE_TEXT_WRAPPED";
QRegExp RE_NUM("\\d*");

class TestMainWindow : public MainWindow {
public:
  explicit TestMainWindow();
  ~TestMainWindow();

//  void testWidget() {
//    for (auto w : QObject::findChildren<QPushButton*>()) {
//      qDebug() << "Hi";
//    }
//  }
};

QStringList getParentWidgets(QWidget* widget){
  QStringList parentWidgets;
  while (widget->parentWidget() != Q_NULLPTR) {
    widget = widget->parentWidget();
    parentWidgets.append(widget->metaObject()->className());
  }
  return parentWidgets;
}

template <typename T>
void checkTextWidgetType(MainWindow &w) {
  for (auto widget : w.findChildren<T>()) {
    const QString text = widget->text();
    bool isEmpty = text.isEmpty();
    bool isNumber = RE_NUM.exactMatch(text);
    bool wrapped = text.contains(TEST_TEXT);
    // dynamic if source string wrapped, but UI adds text
    bool dynamic = wrapped && text != TEST_TEXT;
    QString parentWidgets = getParentWidgets(widget).join(",");
    if (!isEmpty && !isNumber && !wrapped) {
      FAIL(("\"" + text + "\" must be wrapped. Parent widgets: " + parentWidgets).toStdString());
    }
    if (dynamic) {
      WARN(("\"" + text + "\" is dynamic and needs a custom retranslate function. Parent widgets: " + parentWidgets).toStdString());
    }
  }
}

TEST_CASE("test UI string wrapping") {
  MainWindow w;
  checkTextWidgetType<QPushButton*>(w);
  checkTextWidgetType<QLabel*>(w);
}
