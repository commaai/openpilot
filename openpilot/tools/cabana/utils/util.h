#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
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

#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/settings.h"

inline QColor toQColor(const CabanaColor &color) {
  return QColor(color.r, color.g, color.b, color.a);
}

class LogSlider : public QSlider {
  Q_OBJECT

public:
  LogSlider(double factor, Qt::Orientation orientation, QWidget *parent = nullptr) : factor(factor), QSlider(orientation, parent) {}

  void setRange(double min, double max) {
    log_min = factor * std::log10(min);
    log_max = factor * std::log10(max);
    QSlider::setRange(min, max);
    setValue(QSlider::value());
  }
  int value() const {
    double v = log_min + (log_max - log_min) * ((QSlider::value() - minimum()) / double(maximum() - minimum()));
    return std::lround(std::pow(10, v / factor));
  }
  void setValue(int v) {
    double log_v = std::clamp(factor * std::log10(v), log_min, log_max);
    v = minimum() + (maximum() - minimum()) * ((log_v - log_min) / (log_max - log_min));
    QSlider::setValue(v);
  }

private:
  double factor, log_min = 0, log_max = 1;
};

enum {
  ColorsRole = Qt::UserRole + 1,
  BytesRole = Qt::UserRole + 2
};

class SegmentTree {
public:
  SegmentTree() = default;
  void build(const std::vector<QPointF> &arr);
  inline std::pair<double, double> minmax(int left, int right) const { return get_minmax(1, 0, size - 1, left, right); }

private:
  std::pair<double, double> get_minmax(int n, int left, int right, int range_left, int range_right) const;
  void build_tree(const std::vector<QPointF> &arr, int n, int left, int right);
  std::vector<std::pair<double, double>> tree;
  int size = 0;
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

// Accepts a single identifier: one or more [A-Za-z0-9_], spaces rewritten to '_'.
class NameValidator : public QValidator {
  Q_OBJECT
public:
  NameValidator(QObject *parent=nullptr);
  QValidator::State validate(QString &input, int &pos) const override;
};

// Accepts comma-separated identifiers: \w+(,\w+)*
class NodeValidator : public QValidator {
  Q_OBJECT
public:
  NodeValidator(QObject *parent=nullptr);
  QValidator::State validate(QString &input, int &pos) const override;
};

// Accepts one or more non-whitespace characters (\S+).
class NonWhitespaceValidator : public QValidator {
  Q_OBJECT
public:
  NonWhitespaceValidator(QObject *parent=nullptr);
  QValidator::State validate(QString &input, int &pos) const override;
};

// Accepts a dotted IPv4 address (0-255 per octet).
class IpAddressValidator : public QValidator {
  Q_OBJECT
public:
  IpAddressValidator(QObject *parent=nullptr);
  QValidator::State validate(QString &input, int &pos) const override;
};

// C-locale floating-point validator (matches QString::toDouble).
class DoubleValidator : public QValidator {
  Q_OBJECT
public:
  DoubleValidator(QObject *parent = nullptr);
  QValidator::State validate(QString &input, int &pos) const override;
};

namespace utils {

QPixmap icon(const QString &id);
std::string homePath();
std::filesystem::path configPath();
bool getClipboardText(std::string *text);  // false if no clipboard tool is available
bool setClipboardText(const std::string &text);
bool isDarkTheme();
void setTheme(int theme);
QString formatSeconds(double sec, bool include_milliseconds = false, bool absolute_time = false);
inline void drawStaticText(QPainter *p, const QRect &r, const QStaticText &text) {
  auto size = (r.size() - text.size()) / 2;
  p->drawStaticText(r.left() + size.width(), r.top() + size.height(), text);
}
inline QString toHex(const std::vector<uint8_t> &dat, char separator = '\0') {
  static const char digits[] = "0123456789ABCDEF";
  QString hex;
  hex.reserve(dat.size() * (separator ? 3 : 2));
  for (size_t i = 0; i < dat.size(); ++i) {
    if (separator && i) hex += QLatin1Char(separator);
    hex += QLatin1Char(digits[dat[i] >> 4]);
    hex += QLatin1Char(digits[dat[i] & 0xf]);
  }
  return hex;
}

// boundary conversions for the remaining Qt byte-array based state APIs
template <typename T>
std::vector<uint8_t> toBytes(const T &dat) { return {dat.begin(), dat.end()}; }
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
    connect(&settings, &Settings::changed, this, &ToolButton::updateIcon);
  }
  void setIcon(const QString &icon) {
    icon_str = icon;
    QToolButton::setIcon(utils::icon(icon_str));
  }

private:
  void updateIcon() { if (std::exchange(theme, settings.theme) != theme) setIcon(icon_str); }
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

// Watches SIGINT/SIGTERM via a self-pipe and a dedicated waiter thread
// (no Qt notifiers/timers). Exit is marshaled onto the GUI thread.
class UnixSignalHandler {
public:
  UnixSignalHandler();
  ~UnixSignalHandler();
  static void signalHandler(int s);

private:
  inline static int sig_fd[2] = {};
  std::atomic<bool> shutting_down{false};
  std::thread waiter;
};

int num_decimals(double num);
QString signalToolTip(const cabana::Signal *sig);
inline QString toHexString(int value) { return QString("0x%1").arg(QString::number(value, 16).toUpper(), 2, '0'); }
void initApp(int argc, char *argv[], bool disable_hidpi = true);
QPixmap bootstrapPixmap(const QString &id);
