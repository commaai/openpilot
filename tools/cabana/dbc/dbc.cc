#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/util.h"

uint qHash(const MessageId &item) {
  return qHash(item.source) ^ qHash(item.address);
}

QVector<const cabana::Signal *> cabana::Msg::getSignals() const {
  QVector<const Signal *> ret;
  ret.reserve(sigs.size());
  for (auto &sig : sigs) ret.push_back(&sig);
  std::sort(ret.begin(), ret.end(), [](auto l, auto r) {
    return std::tuple(r->type, l->multiplex_value, l->start_bit, l->name) <
           std::tuple(l->type, r->multiplex_value, r->start_bit, r->name);
  });
  return ret;
}

void cabana::Msg::update() {
  mask = QVector<uint8_t>(size, 0x00).toList();
  multiplexor = nullptr;

  for (auto &sig : sigs) {
    if (sig.type == cabana::Signal::Type::Multiplexor) {
      multiplexor = &sig;
    }
    sig.update();

    int i = sig.msb / 8;
    int bits = sig.size;
    while (i >= 0 && i < size && bits > 0) {
      int lsb = (int)(sig.lsb / 8) == i ? sig.lsb : i * 8;
      int msb = (int)(sig.msb / 8) == i ? sig.msb : (i + 1) * 8 - 1;

      int sz = msb - lsb + 1;
      int shift = (lsb - (i * 8));

      mask[i] |= ((1ULL << sz) - 1) << shift;

      bits -= size;
      i = sig.is_little_endian ? i - 1 : i + 1;
    }
  }
  for (auto &sig : sigs) {
    sig.multiplexor = sig.type == cabana::Signal::Type::Multiplexed ? multiplexor : nullptr;
    if (!sig.multiplexor) {
      sig.multiplex_value = 0;
    }
  }
}

// cabana::Signal

void cabana::Signal::update() {
  float h = 19 * (float)lsb / 64.0;
  h = fmod(h, 1.0);
  size_t hash = qHash(name);
  float s = 0.25 + 0.25 * (float)(hash & 0xff) / 255.0;
  float v = 0.75 + 0.25 * (float)((hash >> 8) & 0xff) / 255.0;

  color = QColor::fromHsvF(h, s, v);
  precision = std::max(num_decimals(factor), num_decimals(offset));
}

QString cabana::Signal::formatValue(double value) const {
  // Show enum string
  for (auto &[val, desc] : val_desc) {
    if (std::abs(value - val) < 1e-6) {
      return desc;
    }
  }

  QString val_str = QString::number(value, 'f', precision);
  if (!unit.isEmpty()) {
    val_str += " " + unit;
  }
  return val_str;
}

bool cabana::Signal::getValue(const uint8_t *data, size_t data_size, double *val) const {
  if (multiplexor && get_raw_value(data, data_size, *multiplexor) != multiplex_value) {
    return false;
  }
  *val = get_raw_value(data, data_size, *this);
  return true;
}

bool cabana::Signal::operator==(const cabana::Signal &other) const {
  return name == other.name && size == other.size &&
         start_bit == other.start_bit &&
         msb == other.msb && lsb == other.lsb &&
         is_signed == other.is_signed && is_little_endian == other.is_little_endian &&
         factor == other.factor && offset == other.offset &&
         min == other.min && max == other.max && comment == other.comment && unit == other.unit && val_desc == other.val_desc &&
         multiplex_value == other.multiplex_value && type == other.type;
}

// helper functions

static QVector<int> BIG_ENDIAN_START_BITS = []() {
  QVector<int> ret;
  for (int i = 0; i < 64; i++)
    for (int j = 7; j >= 0; j--)
      ret.push_back(j + i * 8);
  return ret;
}();

double get_raw_value(const uint8_t *data, size_t data_size, const cabana::Signal &sig) {
  int64_t val = 0;

  int i = sig.msb / 8;
  int bits = sig.size;
  while (i >= 0 && i < data_size && bits > 0) {
    int lsb = (int)(sig.lsb / 8) == i ? sig.lsb : i * 8;
    int msb = (int)(sig.msb / 8) == i ? sig.msb : (i + 1) * 8 - 1;
    int size = msb - lsb + 1;

    uint64_t d = (data[i] >> (lsb - (i * 8))) & ((1ULL << size) - 1);
    val |= d << (bits - size);

    bits -= size;
    i = sig.is_little_endian ? i - 1 : i + 1;
  }
  if (sig.is_signed) {
    val -= ((val >> (sig.size - 1)) & 0x1) ? (1ULL << sig.size) : 0;
  }
  return val * sig.factor + sig.offset;
}

int bigEndianStartBitsIndex(int start_bit) { return BIG_ENDIAN_START_BITS[start_bit]; }
int bigEndianBitIndex(int index) { return BIG_ENDIAN_START_BITS.indexOf(index); }

void updateSigSizeParamsFromRange(cabana::Signal &s, int start_bit, int size) {
  s.start_bit = s.is_little_endian ? start_bit : bigEndianBitIndex(start_bit);
  s.size = size;
  if (s.is_little_endian) {
    s.lsb = s.start_bit;
    s.msb = s.start_bit + s.size - 1;
  } else {
    s.lsb = bigEndianStartBitsIndex(bigEndianBitIndex(s.start_bit) + s.size - 1);
    s.msb = s.start_bit;
  }
}

std::pair<int, int> getSignalRange(const cabana::Signal *s) {
  int from = s->is_little_endian ? s->start_bit : bigEndianBitIndex(s->start_bit);
  int to = from + s->size - 1;
  return {from, to};
}
