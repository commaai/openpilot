#include "tools/cabana/dbcmanager.h"

#include <limits>
#include <sstream>
#include <QVector>

DBCManager::DBCManager(QObject *parent) : QObject(parent) {}

DBCManager::~DBCManager() {}

void DBCManager::open(const QString &dbc_file_name) {
  dbc = const_cast<DBC *>(dbc_lookup(dbc_file_name.toStdString()));
  updateMsgMap();
  emit DBCFileChanged();
}

void DBCManager::open(const QString &name, const QString &content) {
  std::istringstream stream(content.toStdString());
  dbc = const_cast<DBC *>(dbc_parse_from_stream(name.toStdString(), stream));
  updateMsgMap();
  emit DBCFileChanged();
}

void DBCManager::updateMsgMap() {
  msg_map.clear();
  for (auto &msg : dbc->msgs)
    msg_map[msg.address] = &msg;
}

QString DBCManager::generateDBC() {
  if (!dbc) return {};

  QString dbc_string;
  for (auto &m : dbc->msgs) {
    dbc_string += QString("BO_ %1 %2: %3 XXX\n").arg(m.address).arg(m.name.c_str()).arg(m.size);
    for (auto &sig : m.sigs) {
      dbc_string += QString(" SG_ %1 : %2|%3@%4%5 (%6,%7) [0|0] \"\" XXX\n")
                        .arg(sig.name.c_str())
                        .arg(sig.start_bit)
                        .arg(sig.size)
                        .arg(sig.is_little_endian ? '1' : '0')
                        .arg(sig.is_signed ? '-' : '+')
                        .arg(sig.factor, 0, 'g', std::numeric_limits<double>::digits10)
                        .arg(sig.offset, 0, 'g', std::numeric_limits<double>::digits10);
    }
    dbc_string += "\n";
  }
  return dbc_string;
}

void DBCManager::updateMsg(const QString &id, const QString &name, uint32_t size) {
  auto [bus, address] = parseId(id);
  if (auto m = const_cast<Msg *>(msg(address))) {
    m->name = name.toStdString();
    m->size = size;
  } else {
    m = &dbc->msgs.emplace_back(Msg{.address = address, .name = name.toStdString(), .size = size});
    msg_map[address] = m;
  }
  emit msgUpdated(address);
}

void DBCManager::removeMsg(const QString &id) {
  uint32_t address = parseId(id).second;
  auto it = std::find_if(dbc->msgs.begin(), dbc->msgs.end(), [address](auto &m) { return m.address == address; });
  if (it != dbc->msgs.end()) {
    dbc->msgs.erase(it);
    updateMsgMap();
    emit msgRemoved(address);
  }
}

void DBCManager::addSignal(const QString &id, const Signal &sig) {
  if (Msg *m = const_cast<Msg *>(msg(id))) {
    emit signalAdded(&m->sigs.emplace_back(sig));
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

std::pair<uint8_t, uint32_t> DBCManager::parseId(const QString &id) {
  const auto list = id.split(':');
  return {list[0].toInt(), list[1].toUInt(nullptr, 16)};
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
