#pragma once

#include <array>
#include <cmath>

#include <QByteArray>
#include <QColor>
#include <QFont>
#include <QRegExpValidator>
#include <QStringBuilder>
#include <QStyledItemDelegate>
#include <QToolButton>
#include <QVector>

#include "tools/cabana/dbc.h"

class ChangeTracker {
public:
  void compute(const QByteArray &dat, double ts, uint32_t freq);
  void clear();

  QVector<double> last_change_t;
  QVector<QColor> colors;
  QVector<std::array<uint32_t, 8>> bit_change_counts;

private:
  const int periodic_threshold = 10;
  const int start_alpha = 128;
  const float fade_time = 2.0;
  QByteArray prev_dat;
};

class LogSlider : public QSlider {
  Q_OBJECT

public:
  LogSlider(double factor, Qt::Orientation orientation, QWidget *parent = nullptr) : factor(factor), QSlider(orientation, parent) {};

  void setRange(double min, double max) { QSlider::setRange(logScale(min), logScale(max)); }
  int value() const { return invLogScale(QSlider::value()); }
  void setValue(int value) { QSlider::setValue(logScale(value)); }

private:
  double factor;
  int logScale(int value) const { return factor * std::log10(value); }
  int invLogScale(int value) const { return std::pow(10, value / factor); }
};


enum {
  ColorsRole = Qt::UserRole + 1,
  BytesRole = Qt::UserRole + 2
};

class SegmentTree {
public:
  SegmentTree() = default;
  void build(const QVector<QPointF> &arr);
  inline std::pair<double, double> minmax(int left, int right) const { return get_minmax(1, 0, size - 1, left, right); }

private:
  std::pair<double, double> get_minmax(int n, int left, int right, int range_left, int range_right) const;
  void build_tree(const QVector<QPointF> &arr, int n, int left, int right);
  std::vector<std::pair<double ,double>> tree;
  int size = 0;
};

class MessageBytesDelegate : public QStyledItemDelegate {
  Q_OBJECT
public:
  MessageBytesDelegate(QObject *parent);
  void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  QFont fixed_font;
  int byte_width;
};

inline QString toHex(const QByteArray &dat) { return dat.toHex(' ').toUpper(); }
QString toHex(uint8_t byte);
QColor getColor(const cabana::Signal *sig);

class NameValidator : public QRegExpValidator {
  Q_OBJECT

public:
  NameValidator(QObject *parent=nullptr);
  QValidator::State validate(QString &input, int &pos) const override;
};

namespace utils {
QPixmap icon(const QString &id);
}

QToolButton *toolButton(const QString &icon, const QString &tooltip);
int num_decimals(double num);
