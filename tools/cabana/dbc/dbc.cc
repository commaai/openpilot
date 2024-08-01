#include "tools/cabana/dbc/dbc.h"

#include <algorithm>

#include "tools/cabana/utils/util.h"

uint qHash(const MessageId &item) {
  return qHash(item.source) ^ qHash(item.address);
}

// cabana::Msg

cabana::Msg::~Msg() {
  for (auto s : sigs) {
    delete s;
  }
}

cabana::Signal *cabana::Msg::addSignal(const cabana::Signal &sig) {
  auto s = sigs.emplace_back(new cabana::Signal(sig));
  update();
  return s;
}

cabana::Signal *cabana::Msg::updateSignal(const QString &sig_name, const cabana::Signal &new_sig) {
  auto s = sig(sig_name);
  if (s) {
    *s = new_sig;
    update();
  }
  return s;
}

void cabana::Msg::removeSignal(const QString &sig_name) {
  auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s->name == sig_name; });
  if (it != sigs.end()) {
    delete *it;
    sigs.erase(it);
    update();
  }
}

cabana::Msg &cabana::Msg::operator=(const cabana::Msg &other) {
  address = other.address;
  name = other.name;
  size = other.size;
  comment = other.comment;

  for (auto s : sigs) delete s;
  sigs.clear();
  for (auto s : other.sigs) {
    sigs.push_back(new cabana::Signal(*s));
  }

  update();
  return *this;
}

cabana::Signal *cabana::Msg::sig(const QString &sig_name) const {
  auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s->name == sig_name; });
  return it != sigs.end() ? *it : nullptr;
}

int cabana::Msg::indexOf(const cabana::Signal *sig) const {
  for (int i = 0; i < sigs.size(); ++i) {
    if (sigs[i] == sig) return i;
  }
  return -1;
}

QString cabana::Msg::newSignalName() {
  QString new_name;
  for (int i = 1; /**/; ++i) {
    new_name = QString("NEW_SIGNAL_%1").arg(i);
    if (sig(new_name) == nullptr) break;
  }
  return new_name;
}

void cabana::Msg::update() {
  if (transmitter.isEmpty()) {
    transmitter = DEFAULT_NODE_NAME;
  }
  mask.assign(size, 0x00);
  multiplexor = nullptr;

  // sort signals
  std::sort(sigs.begin(), sigs.end(), [](auto l, auto r) {
    return std::tie(r->type, l->multiplex_value, l->start_bit, l->name) <
           std::tie(l->type, r->multiplex_value, r->start_bit, r->name);
  });

  for (auto sig : sigs) {
    if (sig->type == cabana::Signal::Type::Multiplexor) {
      multiplexor = sig;
    }
    sig->update();

    // update mask
    int i = sig->msb / 8;
    int bits = sig->size;
    while (i >= 0 && i < size && bits > 0) {
      int lsb = (int)(sig->lsb / 8) == i ? sig->lsb : i * 8;
      int msb = (int)(sig->msb / 8) == i ? sig->msb : (i + 1) * 8 - 1;

      int sz = msb - lsb + 1;
      int shift = (lsb - (i * 8));

      mask[i] |= ((1ULL << sz) - 1) << shift;

      bits -= size;
      i = sig->is_little_endian ? i - 1 : i + 1;
    }
  }

  for (auto sig : sigs) {
    sig->multiplexor = sig->type == cabana::Signal::Type::Multiplexed ? multiplexor : nullptr;
    if (!sig->multiplexor) {
      if (sig->type == cabana::Signal::Type::Multiplexed) {
        sig->type = cabana::Signal::Type::Normal;
      }
      sig->multiplex_value = 0;
    }
  }
}

// cabana::Signal

void cabana::Signal::update() {
  updateMsbLsb(*this);
  if (receiver_name.isEmpty()) {
    receiver_name = DEFAULT_NODE_NAME;
  }

  float h = 19 * (float)lsb / 64.0;
  h = fmod(h, 1.0);
  size_t hash = qHash(name);
  float s = 0.25 + 0.25 * (float)(hash & 0xff) / 255.0;
  float v = 0.75 + 0.25 * (float)((hash >> 8) & 0xff) / 255.0;

  color = QColor::fromHsvF(h, s, v);
  precision = std::max(num_decimals(factor), num_decimals(offset));
}

QString cabana::Signal::formatValue(double value, bool with_unit) const {
  // Show enum string
  int64_t raw_value = round((value - offset) / factor);
  for (const auto &[val, desc] : val_desc) {
    if (std::abs(raw_value - val) < 1e-6) {
      return desc;
    }
  }

  QString val_str = QString::number(value, 'f', precision);
  if (with_unit && !unit.isEmpty()) {
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
         multiplex_value == other.multiplex_value && type == other.type && receiver_name == other.receiver_name;
}

// helper functions

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

void updateMsbLsb(cabana::Signal &s) {
  if (s.is_little_endian) {
    s.lsb = s.start_bit;
    s.msb = s.start_bit + s.size - 1;
  } else {
    s.lsb = flipBitPos(flipBitPos(s.start_bit) + s.size - 1);
    s.msb = s.start_bit;
  }
}
