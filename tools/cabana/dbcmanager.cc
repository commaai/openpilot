#include "tools/cabana/dbcmanager.h"

#include <sstream>
#include <QVector>

DBCManager::DBCManager(QObject *parent) : QObject(parent) {}

DBCManager::~DBCManager() {}

void DBCManager::open(const QString &dbc_file_name) {
  dbc_name = dbc_file_name;
  dbc = const_cast<DBC *>(dbc_lookup(dbc_name.toStdString()));
  msg_map.clear();
  for (auto &msg : dbc->msgs) {
    msg_map[msg.address] = &msg;
  }
  emit DBCFileChanged();
}

void DBCManager::open(const QString &name, const QString &content) {
  this->dbc_name = name;
  std::istringstream stream(content.toStdString());
  dbc = const_cast<DBC *>(dbc_parse_from_stream(name.toStdString(), stream));
  msg_map.clear();
  for (auto &msg : dbc->msgs) {
    msg_map[msg.address] = &msg;
  }
  emit DBCFileChanged();
}

void save(const QString &dbc_file_name) {
  // TODO: save DBC to file
}

void DBCManager::updateMsg(const QString &id, const QString &name, uint32_t size) {
  auto m = const_cast<Msg *>(msg(id));
  if (m) {
    m->name = name.toStdString();
    m->size = size;
  } else {
    uint32_t address = addressFromId(id);
    dbc->msgs.push_back({.address = address, .name = name.toStdString(), .size = size});
    msg_map[address] = &dbc->msgs.back();
  }
  emit msgUpdated(id);
}

void DBCManager::addSignal(const QString &id, const Signal &sig) {
  if (Msg *m = const_cast<Msg *>(msg(id))) {
    m->sigs.push_back(sig);
    emit signalAdded(&m->sigs.back());
  }
}

void DBCManager::updateSignal(const QString &id, const QString &sig_name, const Signal &sig) {
  if (Msg *m = const_cast<Msg *>(msg(id))) {
    auto it = std::find_if(m->sigs.begin(), m->sigs.end(), [=](auto &sig) { return sig_name == sig.name.c_str(); });
    if (it != m->sigs.end()) {
      *it = sig;
      emit signalUpdated(&(*it));
    }
  }
}

void DBCManager::removeSignal(const QString &id, const QString &sig_name) {
  if (Msg *m = const_cast<Msg *>(msg(id))) {
    auto it = std::find_if(m->sigs.begin(), m->sigs.end(), [=](auto &sig) { return sig_name == sig.name.c_str(); });
    if (it != m->sigs.end()) {
      emit signalRemoved(&(*it));
      m->sigs.erase(it);
    }
  }
}

uint32_t DBCManager::addressFromId(const QString &id) {
  return id.mid(id.indexOf(':') + 1).toUInt(nullptr, 16);
}

DBCManager *dbc() {
  static DBCManager dbc_manager(nullptr);
  return &dbc_manager;
}

// helper functions

static QVector<int> BIG_ENDIAN_START_BITS = []() {
  QVector<int> ret;
  for (int i = 0; i < 64; i++)
    for (int j = 7; j >= 0; j--)
      ret.push_back(j + i * 8);
  return ret;
}();

int bigEndianStartBitsIndex(int start_bit) {
  return BIG_ENDIAN_START_BITS[start_bit];
}

int bigEndianBitIndex(int index) {
  return BIG_ENDIAN_START_BITS.indexOf(index);
}

double get_raw_value(uint8_t *data, size_t data_size, const Signal &sig) {
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
  double value = val * sig.factor + sig.offset;
  return value;
}

void updateSigSizeParamsFromRange(Signal &s, int from, int to) {
  s.start_bit = s.is_little_endian ? from : bigEndianBitIndex(from);
  s.size = to - from + 1;
  if (s.is_little_endian) {
    s.lsb = s.start_bit;
    s.msb = s.start_bit + s.size - 1;
  } else {
    s.lsb = bigEndianStartBitsIndex(bigEndianBitIndex(s.start_bit) + s.size - 1);
    s.msb = s.start_bit;
  }
}

std::pair<int, int> getSignalRange(const Signal *s) {
  int from = s->is_little_endian ? s->start_bit : bigEndianBitIndex(s->start_bit);
  int to = from + s->size - 1;
  return {from, to};
}
