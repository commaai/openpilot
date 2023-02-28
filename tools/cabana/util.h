#pragma once

#include <array>

#include <QByteArray>
#include <QColor>
#include <QFont>
#include <QRegExpValidator>
#include <QStringBuilder>
#include <QStyledItemDelegate>
#include <QToolButton>
#include <QVector>

#include "tools/cabana/dbcmanager.h"
using namespace dbcmanager;

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

enum {
  ColorsRole = Qt::UserRole + 1,
  BytesRole = Qt::UserRole + 2
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
QColor getColor(const dbcmanager::Signal *sig);

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
