#include "tools/cabana/dbcmanager.h"

#include <limits>
#include <sstream>
#include <QVector>

DBCManager::DBCManager(QObject *parent) : QObject(parent) {}

DBCManager::~DBCManager() {}

void DBCManager::open(const QString &dbc_file_name) {
  dbc = const_cast<DBC *>(dbc_lookup(dbc_file_name.toStdString()));
  initMsgMap();
}

void DBCManager::open(const QString &name, const QString &content) {
  std::istringstream stream(content.toStdString());
  dbc = const_cast<DBC *>(dbc_parse_from_stream(name.toStdString(), stream));
  initMsgMap();
}

void DBCManager::initMsgMap() {
  msgs.clear();
  for (auto &msg : dbc->msgs) {
    auto &m = msgs[msg.address];
    m.name = msg.name.c_str();
    m.size = msg.size;
    for (auto &s : msg.sigs)
      m.sigs[QString::fromStdString(s.name)] = s;
  }
  emit DBCFileChanged();
}

QString DBCManager::generateDBC() {
  QString dbc_string;
  for (auto &[address, m] : msgs) {
    dbc_string += QString("BO_ %1 %2: %3 XXX\n").arg(address).arg(m.name).arg(m.size);
    for (auto &[name, sig] : m.sigs) {
      dbc_string += QString(" SG_ %1 : %2|%3@%4%5 (%6,%7) [0|0] \"\" XXX\n")
                        .arg(name)
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
  auto [_, address] = parseId(id);
  auto &m = msgs[address];
  m.name = name;
  m.size = size;
  emit msgUpdated(address);
}

void DBCManager::removeMsg(const QString &id) {
  uint32_t address = parseId(id).second;
  msgs.erase(address);
  emit msgRemoved(address);
}

void DBCManager::addSignal(const QString &id, const Signal &sig) {
  if (auto m = const_cast<DBCMsg *>(msg(id))) {
    auto &s = m->sigs[sig.name.c_str()];
    s = sig;
    emit signalAdded(parseId(id).second, &s);
  }
}

void DBCManager::updateSignal(const QString &id, const QString &sig_name, const Signal &sig) {
  if (auto m = const_cast<DBCMsg *>(msg(id))) {
    // change key name
    QString new_name = QString::fromStdString(sig.name);
    auto node = m->sigs.extract(sig_name);
    node.key() = new_name;
    auto it = m->sigs.insert(std::move(node));
    auto &s = m->sigs[new_name];
    s = sig;
    emit signalUpdated(&s);
  }
}

void DBCManager::removeSignal(const QString &id, const QString &sig_name) {
  if (auto m = const_cast<DBCMsg *>(msg(id))) {
    auto it = m->sigs.find(sig_name);
    if (it != m->sigs.end()) {
      emit signalRemoved(&(it->second));
      m->sigs.erase(it);
    }
  }
}

std::pair<uint8_t, uint32_t> DBCManager::parseId(const QString &id) {
  const auto list = id.split(':');
  if (list.size() != 2) return {0, 0};
  return {list[0].toInt(), list[1].toUInt(nullptr, 16)};
}

DBCManager *dbc() {
  static DBCManager dbc_manager(nullptr);
  return &dbc_manager;
}

// DBCMsg

std::vector<const Signal*> DBCMsg::getSignals() const {
  std::vector<const Signal*> ret;
  ret.reserve(sigs.size());
  for (auto &[_, sig] : sigs) ret.push_back(&sig);
  std::sort(ret.begin(), ret.end(), [](auto l, auto r) { return l->start_bit < r->start_bit; });
  return ret;
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

void updateSigSizeParamsFromRange(Signal &s, int start_bit, int size) {
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

std::pair<int, int> getSignalRange(const Signal *s) {
  int from = s->is_little_endian ? s->start_bit : bigEndianBitIndex(s->start_bit);
  int to = from + s->size - 1;
  return {from, to};
}

bool operator==(const Signal &l, const Signal &r) {
  return l.name == r.name && l.size == r.size &&
         l.start_bit == r.start_bit &&
         l.msb == r.msb && l.lsb == r.lsb &&
         l.is_signed == r.is_signed && l.is_little_endian == r.is_little_endian &&
         l.factor == r.factor && l.offset == r.offset;
}
