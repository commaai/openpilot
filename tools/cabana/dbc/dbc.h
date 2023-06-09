#pragma once

#include <QColor>
#include <QList>
#include <QMetaType>
#include <QString>
#include <limits>

#include "opendbc/can/common_dbc.h"

const QString UNTITLED = "untitled";

struct MessageId {
  uint8_t source = 0;
  uint32_t address = 0;

  QString toString() const {
    return QString("%1:%2").arg(source).arg(address, 1, 16);
  }

  bool operator==(const MessageId &other) const {
    return source == other.source && address == other.address;
  }

  bool operator!=(const MessageId &other) const {
    return !(*this == other);
  }

  bool operator<(const MessageId &other) const {
    return std::pair{source, address} < std::pair{other.source, other.address};
  }

  bool operator>(const MessageId &other) const {
    return std::pair{source, address} > std::pair{other.source, other.address};
  }
};

uint qHash(const MessageId &item);
Q_DECLARE_METATYPE(MessageId);

template <>
struct std::hash<MessageId> {
  std::size_t operator()(const MessageId &k) const noexcept { return qHash(k); }
};

typedef QList<std::pair<double, QString>> ValueDescription;

namespace cabana {

class Signal {
public:
  Signal() = default;
  Signal(const Signal &other) = default;
  void update();
  QString formatValue(double value) const;

  QString name;
  int start_bit, msb, lsb, size;
  double factor, offset;
  bool is_signed;
  bool is_little_endian;
  double min, max;
  QString unit;
  QString comment;
  ValueDescription val_desc;
  int precision = 0;
  QColor color;
};

class Msg {
public:
  Msg() = default;
  Msg(const Msg &other) { *this = other; }
  ~Msg();
  cabana::Signal *addSignal(const cabana::Signal &sig);
  cabana::Signal *updateSignal(const QString &sig_name, const cabana::Signal &sig);
  void removeSignal(const QString &sig_name);
  Msg &operator=(const Msg &other);
  int indexOf(const cabana::Signal *sig) const;
  cabana::Signal *sig(const QString &sig_name) const;
  QString newSignalName();
  inline const std::vector<cabana::Signal *> &getSignals() const { return sigs; }

  uint32_t address;
  QString name;
  uint32_t size;
  QString comment;
  std::vector<cabana::Signal *> sigs;

  QList<uint8_t> mask;
  void update();
};

bool operator==(const cabana::Signal &l, const cabana::Signal &r);
inline bool operator!=(const cabana::Signal &l, const cabana::Signal &r) { return !(l == r); }

}  // namespace cabana

// Helper functions
double get_raw_value(const uint8_t *data, size_t data_size, const cabana::Signal &sig);
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);
void updateSigSizeParamsFromRange(cabana::Signal &s, int start_bit, int size);
std::pair<int, int> getSignalRange(const cabana::Signal *s);
inline std::vector<std::string> allDBCNames() { return get_dbc_names(); }
inline QString doubleToString(double value) { return QString::number(value, 'g', std::numeric_limits<double>::digits10); }
