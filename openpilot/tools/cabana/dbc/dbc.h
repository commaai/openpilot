#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <QColor>
#include <QMetaType>

const std::string UNTITLED = "untitled";
const std::string DEFAULT_NODE_NAME = "XXX";
constexpr int CAN_MAX_DATA_BYTES = 64;

struct MessageId {
  uint8_t source = 0;
  uint32_t address = 0;

  std::string toString() const {
    char buf[64];
    snprintf(buf, sizeof(buf), "%u:%X", source, address);
    return buf;
  }

  inline static MessageId fromString(const std::string &str) {
    auto pos = str.find(':');
    if (pos == std::string::npos) return {};
    return MessageId{.source = uint8_t(std::stoul(str.substr(0, pos))),
                     .address = uint32_t(std::stoul(str.substr(pos + 1), nullptr, 16))};
  }

  bool operator==(const MessageId &other) const {
    return source == other.source && address == other.address;
  }

  bool operator!=(const MessageId &other) const {
    return !(*this == other);
  }

  bool operator<(const MessageId &other) const {
    return std::tie(source, address) < std::tie(other.source, other.address);
  }

  bool operator>(const MessageId &other) const {
    return std::tie(source, address) > std::tie(other.source, other.address);
  }
};

Q_DECLARE_METATYPE(MessageId);

template <>
struct std::hash<MessageId> {
  std::size_t operator()(const MessageId &k) const noexcept {
    return std::hash<uint8_t>{}(k.source) ^ (std::hash<uint32_t>{}(k.address) << 1);
  }
};

typedef std::vector<std::pair<double, std::string>> ValueDescription;
Q_DECLARE_METATYPE(ValueDescription);

namespace cabana {

class Signal {
public:
  Signal() = default;
  Signal(const Signal &other) = default;
  void update();
  bool getValue(const uint8_t *data, size_t data_size, double *val) const;
  std::string formatValue(double value, bool with_unit = true) const;
  bool operator==(const cabana::Signal &other) const;
  inline bool operator!=(const cabana::Signal &other) const { return !(*this == other); }

  enum class Type {
    Normal = 0,
    Multiplexed,
    Multiplexor
  };

  Type type = Type::Normal;
  std::string name;
  int start_bit, msb, lsb, size;
  double factor = 1.0;
  double offset = 0;
  bool is_signed;
  bool is_little_endian;
  double min, max;
  std::string unit;
  std::string comment;
  std::string receiver_name;
  ValueDescription val_desc;
  int precision = 0;
  QColor color;

  // Multiplexed
  int multiplex_value = 0;
  Signal *multiplexor = nullptr;
};

class Msg {
public:
  Msg() = default;
  Msg(const Msg &other) { *this = other; }
  ~Msg();
  cabana::Signal *addSignal(const cabana::Signal &sig);
  cabana::Signal *updateSignal(const std::string &sig_name, const cabana::Signal &sig);
  void removeSignal(const std::string &sig_name);
  Msg &operator=(const Msg &other);
  int indexOf(const cabana::Signal *sig) const;
  cabana::Signal *sig(const std::string &sig_name) const;
  std::string newSignalName();
  void update();
  inline const std::vector<cabana::Signal *> &getSignals() const { return sigs; }

  uint32_t address;
  std::string name;
  uint32_t size;
  std::string comment;
  std::string transmitter;
  std::vector<cabana::Signal *> sigs;

  std::vector<uint8_t> mask;
  cabana::Signal *multiplexor = nullptr;
};

}  // namespace cabana

// Helper functions
double get_raw_value(const uint8_t *data, size_t data_size, const cabana::Signal &sig);
void updateMsbLsb(cabana::Signal &s);
inline int flipBitPos(int start_bit) { return 8 * (start_bit / 8) + 7 - start_bit % 8; }
inline std::string doubleToString(double value) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.*g", std::numeric_limits<double>::digits10, value);
  return buf;
}
