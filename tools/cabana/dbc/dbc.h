#pragma once

#include <limits>
#include <QList>
#include <QMetaType>
#include <QString>

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

typedef QList<std::pair<QString, QString>> ValueDescription;

namespace cabana {
  struct Signal {
    QString name;
    int start_bit, msb, lsb, size;
    bool is_signed;
    double factor, offset;
    bool is_little_endian;
    double min, max;
    QString unit;
    QString comment;
    ValueDescription val_desc;
    int precision = 0;
    void updatePrecision();
    QString formatValue(double value) const;
  };

  struct Msg {
    uint32_t address;
    QString name;
    uint32_t size;
    QString comment;
    QList<cabana::Signal> sigs;

    QList<uint8_t> mask;
    void updateMask();

    std::vector<const cabana::Signal*> getSignals() const;
    const cabana::Signal *sig(const QString &sig_name) const {
        auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s.name == sig_name; });
        return it != sigs.end() ? &(*it) : nullptr;
    }
  };

  bool operator==(const cabana::Signal &l, const cabana::Signal &r);
  inline bool operator!=(const cabana::Signal &l, const cabana::Signal &r) { return !(l == r); }
}

// Helper functions
double get_raw_value(const uint8_t *data, size_t data_size, const cabana::Signal &sig);
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);
void updateSigSizeParamsFromRange(cabana::Signal &s, int start_bit, int size);
std::pair<int, int> getSignalRange(const cabana::Signal *s);
inline std::vector<std::string> allDBCNames() { return get_dbc_names(); }
inline QString doubleToString(double value) { return QString::number(value, 'g', std::numeric_limits<double>::digits10); }

