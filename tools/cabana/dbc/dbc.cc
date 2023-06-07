#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/util.h"

uint qHash(const MessageId &item) {
  return qHash(item.source) ^ qHash(item.address);
}

std::vector<const cabana::Signal*> cabana::Msg::getSignals() const {
  std::vector<const Signal*> ret;
  ret.reserve(sigs.size());
  for (auto &sig : sigs) ret.push_back(&sig);
  std::sort(ret.begin(), ret.end(), [](auto l, auto r) {
    if (l->start_bit != r->start_bit) {
      return l->start_bit < r->start_bit;
    }
    // For VECTOR__INDEPENDENT_SIG_MSG, many signals have same start bit
    return l->name < r->name;
  });
  return ret;
}

void cabana::Msg::updateMask() {
  mask = QVector<uint8_t>(size, 0x00).toList();
  for (auto &sig : sigs) {
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
}

void cabana::Signal::updatePrecision() {
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

bool cabana::operator==(const cabana::Signal &l, const cabana::Signal &r) {
  return l.name == r.name && l.size == r.size &&
         l.start_bit == r.start_bit &&
         l.msb == r.msb && l.lsb == r.lsb &&
         l.is_signed == r.is_signed && l.is_little_endian == r.is_little_endian &&
         l.factor == r.factor && l.offset == r.offset &&
         l.min == r.min && l.max == r.max && l.comment == r.comment && l.unit == r.unit && l.val_desc == r.val_desc;
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
