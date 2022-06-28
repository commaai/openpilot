#include "catch2/catch.hpp"
#include "selfdrive/ui/qt/window.h"
#include <QDebug>

const QString TEST_TEXT = "(WRAPPED_SOURCE_TEXT)";
QRegExp RE_NUM("\\d*");

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
    qWarning() << text;
    bool isNumber = RE_NUM.exactMatch(text);
    bool wrapped = text.contains(TEST_TEXT);
    QString parentWidgets = getParentWidgets(widget).join("->");
    if (!text.isEmpty() && !isNumber && !wrapped) {
      FAIL(("\"" + text + "\" must be wrapped. Parent widgets: " + parentWidgets).toStdString());
    }
    // dynamic if source string wrapped, but UI adds text
    if (wrapped && text != TEST_TEXT) {
      WARN(("\"" + text + "\" is dynamic and needs a custom retranslate function. Parent widgets: " + parentWidgets).toStdString());
    }
  }
}

TEST_CASE("ui_all_strings_marked_tr") {
  qWarning() << "TestCase";
  MainWindow w;
  qWarning() << "MainWindow";
  checkTextWidgetType<QPushButton*>(w);
  qWarning() << "QPushButton";
  checkTextWidgetType<QLabel*>(w);
  qWarning() << "QLabel";
}
