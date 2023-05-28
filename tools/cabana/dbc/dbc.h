#pragma once

#include <limits>
#include <QColor>
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

typedef QList<std::pair<double, QString>> ValueDescription;

namespace cabana {
  struct Signal {
    enum class Type {
      Normal = 0,
      Multiplexed,
      Multiplexor
    };

    Type type = Type::Normal;
    QString name;
    int start_bit, msb, lsb, size;
    double factor, offset;
    bool is_signed;
    bool is_little_endian;
    double min, max;
    QString unit;
    QString comment;
    ValueDescription val_desc;

    // Multiplexed
    int multiplex_value = 0;
    Signal *multiplexor = nullptr;

    int precision = 0;
    QColor color;

    void update();
    QString formatValue(double value) const;
    bool getValue(const uint8_t *data, size_t data_size, double *val) const;
    bool operator==(const cabana::Signal &other) const;
    inline bool operator!=(const cabana::Signal &other) { return !(*this == other); }
  };

  struct Msg {
    uint32_t address;
    QString name;
    uint32_t size;
    QString comment;
    QList<cabana::Signal> sigs;

    QList<uint8_t> mask;
    cabana::Signal *multiplexor = nullptr;

    void update();
    std::vector<const cabana::Signal*> getSignals() const;
    const cabana::Signal *sig(const QString &sig_name) const {
      auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s.name == sig_name; });
      return it != sigs.end() ? &(*it) : nullptr;
    }
  };
}

// Helper functions
double get_raw_value(const uint8_t *data, size_t data_size, const cabana::Signal &sig);
int bigEndianStartBitsIndex(int start_bit);
int bigEndianBitIndex(int index);
void updateSigSizeParamsFromRange(cabana::Signal &s, int start_bit, int size);
std::pair<int, int> getSignalRange(const cabana::Signal *s);
inline std::vector<std::string> allDBCNames() { return get_dbc_names(); }
inline QString doubleToString(double value) { return QString::number(value, 'g', std::numeric_limits<double>::digits10); }
