#pragma once

#include <array>
#include <utility>

#include <QApplication>
#include <QColor>
#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QStaticText>
#include <QStringBuilder>
#include <QStyledItemDelegate>
#include <QToolButton>
#include <QValidator>

#include "tools/cabana/core/observable.h"
#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/utils/strings.h"
#include "tools/cabana/utils/util.h"

// needed by QVariant::fromValue() in the Qt views; goes away with QVariant
Q_DECLARE_METATYPE(MessageId)
Q_DECLARE_METATYPE(ValueDescription)

inline QColor toQColor(const CabanaColor &color) {
  return QColor(color.r, color.g, color.b, color.a);
}

class LogSlider : public QSlider {
  Q_OBJECT

public:
  LogSlider(double factor, Qt::Orientation orientation, QWidget *parent = nullptr) : scale(factor), QSlider(orientation, parent) {}

  void setRange(double min, double max) {
    scale.setRange(min, max);
    QSlider::setRange(min, max);
    setValue(QSlider::value());
  }
  int value() const { return scale.value(QSlider::value(), minimum(), maximum()); }
  void setValue(int v) { QSlider::setValue(scale.position(v, minimum(), maximum())); }

private:
  LogScale scale;
};

enum {
  ColorsRole = Qt::UserRole + 1,
  BytesRole = Qt::UserRole + 2
};

class MessageBytesDelegate : public QStyledItemDelegate {
  Q_OBJECT
public:
  MessageBytesDelegate(QObject *parent, bool multiple_lines = false);
  void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  bool multipleLines() const { return multiple_lines; }
  void setMultipleLines(bool v) { multiple_lines = v; }
  QSize sizeForBytes(int n) const;

private:
  std::array<QStaticText, 256> hex_text_table;
  QFontMetrics font_metrics;
  QFont fixed_font;
  QSize byte_size = {};
  bool multiple_lines = false;
  int h_margin, v_margin;
};

// QValidator wrappers around the std::string validators in util.h
#define CABANA_VALIDATOR(Name)                                          \
  class Name : public QValidator {                                      \
    Q_OBJECT                                                            \
  public:                                                               \
    Name(QObject *parent = nullptr) : QValidator(parent) {}             \
    QValidator::State validate(QString &input, int &pos) const override; \
  };
CABANA_VALIDATOR(NameValidator)
CABANA_VALIDATOR(NodeValidator)
CABANA_VALIDATOR(NonWhitespaceValidator)
CABANA_VALIDATOR(IpAddressValidator)
CABANA_VALIDATOR(DoubleValidator)
#undef CABANA_VALIDATOR

namespace utils {

QPixmap icon(const QString &id);
bool isDarkTheme();
void setTheme(int theme);
inline void drawStaticText(QPainter *p, const QRect &r, const QStaticText &text) {
  auto size = (r.size() - text.size()) / 2;
  p->drawStaticText(r.left() + size.width(), r.top() + size.height(), text);
}
inline auto qbytes(const std::vector<uint8_t> &dat) {
  return decltype(QString().toUtf8())((const char *)dat.data(), (int)dat.size());
}

}

class ToolButton : public QToolButton {
  Q_OBJECT
public:
  ToolButton(const QString &icon, const QString &tooltip = {}, QWidget *parent = nullptr) : QToolButton(parent) {
    setIcon(icon);
    setToolTip(tooltip);
    setAutoRaise(true);
    const int metric = QApplication::style()->pixelMetric(QStyle::PM_SmallIconSize);
    setIconSize({metric, metric});
    theme = settings.theme;
    settings_connection_ = settings.changed.connect([this]() { updateIcon(); });
  }
  void setIcon(const QString &icon) {
    icon_str = icon;
    QToolButton::setIcon(utils::icon(icon_str));
  }

private:
  void updateIcon() { if (std::exchange(theme, settings.theme) != theme) setIcon(icon_str); }
  Connection settings_connection_;
  QString icon_str;
  int theme;
};

class TabBar : public QTabBar {
  Q_OBJECT

public:
  TabBar(QWidget *parent) : QTabBar(parent) {}
  int addTab(const QString &text);

private:
  void closeTabClicked();
};

void initApp(int argc, char *argv[], bool disable_hidpi = true);
QPixmap bootstrapPixmap(const QString &id);
