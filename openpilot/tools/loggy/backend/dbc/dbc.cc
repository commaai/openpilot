#include "tools/loggy/backend/dbc/dbc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace loggy {

namespace {

ColorRGBA hsvToRgb(float h, float s, float v) {
  auto clamp01 = [](float x) { return std::max(0.0f, std::min(1.0f, x)); };
  h = clamp01(h);
  s = clamp01(s);
  v = clamp01(v);

  float r = v, g = v, b = v;
  if (s > 0.0f) {
    float hh = h * 6.0f;
    if (hh >= 6.0f) hh = 0.0f;
    int i = static_cast<int>(hh);
    float f = hh - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      default: r = v; g = p; b = q; break;
    }
  }

  auto to255 = [&clamp01](float x) { return static_cast<uint8_t>(std::lround(clamp01(x) * 255.0f)); };
  return ColorRGBA{to255(r), to255(g), to255(b), 255};
}

int num_decimals(double num) {
  // Mirrors QString::number(num) (format 'g', precision 6): count every
  // character after the first '.', including any exponent suffix.
  char buf[64];
  snprintf(buf, sizeof(buf), "%.6g", num);
  std::string str(buf);
  auto dot_pos = str.find('.');
  return dot_pos == std::string::npos ? 0 : static_cast<int>(str.size() - dot_pos - 1);
}

}  // namespace

// Msg

Msg::~Msg() {
  for (auto s : sigs) {
    delete s;
  }
}

Signal *Msg::addSignal(const Signal &sig) {
  auto s = sigs.emplace_back(new Signal(sig));
  update();
  return s;
}

Signal *Msg::updateSignal(const std::string &sig_name, const Signal &new_sig) {
  auto s = sig(sig_name);
  if (s) {
    *s = new_sig;
    update();
  }
  return s;
}

void Msg::removeSignal(const std::string &sig_name) {
  auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s->name == sig_name; });
  if (it != sigs.end()) {
    delete *it;
    sigs.erase(it);
    update();
  }
}

Msg &Msg::operator=(const Msg &other) {
  address = other.address;
  name = other.name;
  size = other.size;
  comment = other.comment;
  transmitter = other.transmitter;

  for (auto s : sigs) delete s;
  sigs.clear();
  for (auto s : other.sigs) {
    sigs.push_back(new Signal(*s));
  }

  update();
  return *this;
}

Signal *Msg::sig(const std::string &sig_name) const {
  auto it = std::find_if(sigs.begin(), sigs.end(), [&](auto &s) { return s->name == sig_name; });
  return it != sigs.end() ? *it : nullptr;
}

int Msg::indexOf(const Signal *sig) const {
  for (int i = 0; i < sigs.size(); ++i) {
    if (sigs[i] == sig) return i;
  }
  return -1;
}

std::string Msg::newSignalName() {
  std::string new_name;
  for (int i = 1; /**/; ++i) {
    new_name = "NEW_SIGNAL_" + std::to_string(i);
    if (sig(new_name) == nullptr) break;
  }
  return new_name;
}

void Msg::update() {
  if (transmitter.empty()) {
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
    if (sig->type == Signal::Type::Multiplexor) {
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

      bits -= sz;
      i = sig->is_little_endian ? i - 1 : i + 1;
    }
  }

  for (auto sig : sigs) {
    sig->multiplexor = sig->type == Signal::Type::Multiplexed ? multiplexor : nullptr;
    if (!sig->multiplexor) {
      if (sig->type == Signal::Type::Multiplexed) {
        sig->type = Signal::Type::Normal;
      }
      sig->multiplex_value = 0;
    }
  }
}

// Signal

void Signal::update() {
  updateMsbLsb(*this);
  if (receiver_name.empty()) {
    receiver_name = DEFAULT_NODE_NAME;
  }

  float h = 19 * (float)lsb / 64.0;
  h = fmod(h, 1.0);
  size_t hash = std::hash<std::string>{}(name);
  float s = 0.25 + 0.25 * (float)(hash & 0xff) / 255.0;
  float v = 0.75 + 0.25 * (float)((hash >> 8) & 0xff) / 255.0;

  color = hsvToRgb(h, s, v);
  precision = std::max(num_decimals(factor), num_decimals(offset));
}

std::string Signal::formatValue(double value, bool with_unit) const {
  // Show enum string
  int64_t raw_value = round((value - offset) / factor);
  for (const auto &[val, desc] : val_desc) {
    if (std::abs(raw_value - val) < 1e-6) {
      return desc;
    }
  }

  char buf[64];
  snprintf(buf, sizeof(buf), "%.*f", precision, value);
  std::string val_str(buf);
  if (with_unit && !unit.empty()) {
    val_str += " " + unit;
  }
  return val_str;
}

bool Signal::getValue(const uint8_t *data, size_t data_size, double *val) const {
  if (multiplexor && get_raw_value(data, data_size, *multiplexor) != multiplex_value) {
    return false;
  }
  *val = get_raw_value(data, data_size, *this);
  return true;
}

bool Signal::operator==(const Signal &other) const {
  return name == other.name && size == other.size &&
         start_bit == other.start_bit &&
         msb == other.msb && lsb == other.lsb &&
         is_signed == other.is_signed && is_little_endian == other.is_little_endian &&
         factor == other.factor && offset == other.offset &&
         min == other.min && max == other.max && comment == other.comment && unit == other.unit && val_desc == other.val_desc &&
         multiplex_value == other.multiplex_value && type == other.type && receiver_name == other.receiver_name;
}

// helper functions

double get_raw_value(const uint8_t *data, size_t data_size, const Signal &sig) {
  const int msb_byte = sig.msb / 8;
  if (msb_byte >= (int)data_size) return 0;

  const int lsb_byte = sig.lsb / 8;
  uint64_t val = 0;

  // Fast path: signal fits in a single byte
  if (msb_byte == lsb_byte) {
    val = (data[msb_byte] >> (sig.lsb & 7)) & ((1ULL << sig.size) - 1);
  } else {
    // Multi-byte case: signal spans across multiple bytes
    int bits = sig.size;
    int i = msb_byte;
    const int step = sig.is_little_endian ? -1 : 1;
    while (i >= 0 && i < (int)data_size && bits > 0) {
      const int msb = (i == msb_byte) ? sig.msb & 7 : 7;
      const int lsb = (i == lsb_byte) ? sig.lsb & 7 : 0;
      const int nbits = msb - lsb + 1;
      val = (val << nbits) | ((data[i] >> lsb) & ((1ULL << nbits) - 1));
      bits -= nbits;
      i += step;
    }
  }

  // Sign extension (if needed)
  if (sig.is_signed && (val & (1ULL << (sig.size - 1)))) {
    val |= ~((1ULL << sig.size) - 1);
  }

  return static_cast<int64_t>(val) * sig.factor + sig.offset;
}

void updateMsbLsb(Signal &s) {
  if (s.is_little_endian) {
    s.lsb = s.start_bit;
    s.msb = s.start_bit + s.size - 1;
  } else {
    s.lsb = flipBitPos(flipBitPos(s.start_bit) + s.size - 1);
    s.msb = s.start_bit;
  }
}

}  // namespace loggy
