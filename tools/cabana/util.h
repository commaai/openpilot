#pragma once

#include <QByteArray>
#include <QColor>
#include <QFont>
#include <QRegExpValidator>
#include <QStyledItemDelegate>
#include <QVector>

#include "opendbc/can/common_dbc.h"

class ChangeTracker {
public:
  void compute(const QByteArray &dat, double ts, uint32_t freq);
  static QList<QVariant> toVariantList(const QVector<QColor> &colors);
  void clear();

  QVector<double> last_change_t;
  QVector<QColor> colors;

private:
  const int periodic_threshold = 10;
  const int start_alpha = 128;
  const float fade_time = 2.0;
  QByteArray prev_dat;
};

class MessageBytesDelegate : public QStyledItemDelegate {
  Q_OBJECT
public:
  MessageBytesDelegate(QObject *parent);
  void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
  QFont fixed_font;
};

inline QString toHex(const QByteArray &dat) { return dat.toHex(' ').toUpper(); }
inline char toHex(uint value) { return "0123456789ABCDEF"[value & 0xF]; }
QColor getColor(const Signal *sig);

class NameValidator : public QRegExpValidator {
  Q_OBJECT

public:
  NameValidator(QObject *parent=nullptr);
  QValidator::State validate(QString &input, int &pos) const override;
};

namespace utils {
QPixmap icon(const QString &id);
}
