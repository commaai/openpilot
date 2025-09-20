#include "catch2/catch.hpp"

#include "common/params.h"
#include "selfdrive/ui/qt/window.h"

const QString TEST_TEXT = "(WRAPPED_SOURCE_TEXT)";  // what each string should be translated to
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
void checkWidgetTrWrap(MainWindow &w) {
  for (auto widget : w.findChildren<T>()) {
  const QString text = widget->text();
    bool isNumber = RE_NUM.exactMatch(text);
    bool wrapped = text.contains(TEST_TEXT);
    QString parentWidgets = getParentWidgets(widget).join("->");

    if (!text.isEmpty() && !isNumber && !wrapped) {
      FAIL(("\"" + text + "\" must be wrapped. Parent widgets: " + parentWidgets).toStdString());
    }

    // warn if source string wrapped, but UI adds text
    // TODO: add way to ignore this
    if (wrapped && text != TEST_TEXT) {
      WARN(("\"" + text + "\" is dynamic and needs a custom retranslate function. Parent widgets: " + parentWidgets).toStdString());
    }
  }
}

// Tests all strings in the UI are wrapped with tr()
TEST_CASE("UI: test all strings wrapped") {
  Params().remove("LanguageSetting");
  Params().remove("HardwareSerial");
  Params().remove("DongleId");
  qputenv("TICI", "1");

  MainWindow w;
  checkWidgetTrWrap<QPushButton*>(w);
  checkWidgetTrWrap<QLabel*>(w);
}
