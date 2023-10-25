#pragma once

#include <array>
#include <cmath>
#include <deque>
#include <vector>
#include <utility>

#include <QApplication>
#include <QByteArray>
#include <QDateTime>
#include <QDoubleValidator>
#include <QFont>
#include <QPainter>
#include <QRegExpValidator>
#include <QSocketNotifier>
#include <QStaticText>
#include <QStringBuilder>
#include <QStyledItemDelegate>
#include <QToolButton>
#include <QVector>

#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/settings.h"

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
  int widthForBytes(int n) const;

private:
  std::array<QStaticText, 256> hex_text_table;
  QFont fixed_font;
  QSize byte_size = {};
  bool multiple_lines = false;
};

inline QString toHex(const QByteArray &dat) { return dat.toHex(' ').toUpper(); }
QString toHex(uint8_t byte);

class NameValidator : public QRegExpValidator {
  Q_OBJECT
public:
  NameValidator(QObject *parent=nullptr);
  QValidator::State validate(QString &input, int &pos) const override;
};

class DoubleValidator : public QDoubleValidator {
  Q_OBJECT
public:
  DoubleValidator(QObject *parent = nullptr);
};

namespace utils {
QPixmap icon(const QString &id);
void setTheme(int theme);
QString formatSeconds(double sec, bool include_milliseconds = false, bool absolute_time = false);
inline void drawStaticText(QPainter *p, const QRect &r, const QStaticText &text) {
  auto size = (r.size() - text.size()) / 2;
  p->drawStaticText(r.left() + size.width(), r.top() + size.height(), text);
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

class UnixSignalHandler : public QObject {
  Q_OBJECT

public:
  UnixSignalHandler(QObject *parent = nullptr);
  ~UnixSignalHandler();
  static void signalHandler(int s);

public slots:
  void handleSigTerm();

private:
  inline static int sig_fd[2] = {};
  QSocketNotifier *sn;
};

class MonotonicBuffer {
public:
  MonotonicBuffer(size_t initial_size) : next_buffer_size(initial_size) {}
  ~MonotonicBuffer();
  void *allocate(size_t bytes, size_t alignment = 16ul);
  void deallocate(void *p) {}

private:
  void *current_buf = nullptr;
  size_t next_buffer_size = 0;
  size_t available = 0;
  std::deque<void *> buffers;
  static constexpr float growth_factor = 1.5;
};

int num_decimals(double num);
QString signalToolTip(const cabana::Signal *sig);
