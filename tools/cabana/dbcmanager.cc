#include "tools/cabana/dbcmanager.h"

#include <QDebug>
#include <QFile>
#include <QVector>
#include <limits>
#include <regex>
#include <sstream>

namespace {

std::regex bo_regexp(R"(^BO_ (\w+) (\w+) *: (\w+) (\w+))");
std::regex sg_regexp(R"(^SG_ (\w+) : (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
std::regex sgm_regexp(R"(^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
std::regex sg_comment_regexp(R"(^CM_ SG_ *(\w+) *(\w+) *\"(.*)\";)");
std::regex val_regexp(R"(VAL_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+\".+?\"[^;]*))");

inline bool startswith(const std::string &str, const char *prefix) { return str.find(prefix, 0) == 0; }
inline std::string &trim(std::string &s, const char *t = " \t\n\r\f\v") {
  s.erase(s.find_last_not_of(t) + 1);
  return s.erase(0, s.find_first_not_of(t));
}

}  // namespace

DBCManager::DBCManager(QObject *parent) : QObject(parent) {}

DBCManager::~DBCManager() {}

void DBCManager::open(const QString &dbc_file_name) {
  QString opendbc_file_path = QString("%1/%2.dbc").arg(OPENDBC_FILE_PATH, dbc_file_name);
  QFile file(opendbc_file_path);
  if (file.open(QIODevice::ReadOnly)) {
    open(dbc_file_name, file.readAll());
  }
}

void DBCManager::open(const QString &name, const QString &content) {
  std::istringstream stream(content.toStdString());
  dbc = const_cast<DBC *>(dbc_parse_from_stream(name.toStdString(), stream));
  msgs.clear();
  for (auto &msg : dbc->msgs) {
    auto &m = msgs[msg.address];
    m.name = msg.name.c_str();
    m.size = msg.size;
    for (auto &s : msg.sigs) {
      auto &sig = m.sigs[QString::fromStdString(s.name)];
      sig = s;
      m.sig_extra_info[&sig] = {};
    }
  }
  parseExtraInfo(content.toStdString());
  emit DBCFileChanged();
}

void DBCManager::parseExtraInfo(const std::string &content) {
  auto getExtraInfo = [&](uint32_t address, const std::string &sig_name) -> SignalExtraInfo * {
    if (auto m = (DBCMsg *)msg(address)) {
      if (auto it = m->sigs.find(sig_name.c_str()); it != m->sigs.end()) {
        return &m->sig_extra_info[&(it->second)];
      }
    }
    return nullptr;
  };

  std::smatch match;
  std::string line;
  uint32_t address = 0;
  std::istringstream stream(content);
  while (std::getline(stream, line)) {
    line = trim(line);
    if (startswith(line, "BO_ ")) {
      address = (std::regex_match(line, match, bo_regexp)) ? std::stoul(match[1].str()) : 0;
    } else if (startswith(line, "SG_ ")) {
      int offset = 0;
      if (!std::regex_search(line, match, sg_regexp)) {
        if (!std::regex_search(line, match, sgm_regexp)) continue;
        offset = 1;
      }
      std::string sig_name = match[1].str();
      if (auto extra = getExtraInfo(address, sig_name)) {
        extra->min = match[offset + 8].str().c_str();
        extra->max = match[offset + 9].str().c_str();
        extra->unit = match[offset + 10].str().c_str();
      }
    } else if (startswith(line, "VAL_ ")) {
      if (std::regex_search(line, match, val_regexp)) {
        uint32_t msg_address = std::stoul(match[1].str());
        std::string sig_name = match[2].str();
        if (auto extra = getExtraInfo(msg_address, sig_name)) {
          extra->val_desc = QString::fromStdString(match[3].str()).trimmed();
        }
      }
    } else if (startswith(line, "CM_ SG_ ")) {
      if (std::regex_search(line, match, sg_comment_regexp)) {
        uint32_t msg_address = std::stoul(match[1].str());
        std::string sig_name = match[2].str();
        if (auto extra = getExtraInfo(msg_address, sig_name)) {
          extra->comment = match[3].str().c_str();
        }
      }
    }
  }
}

QString DBCManager::generateDBC() {
  QString dbc_string, signal_comment, val_desc;
  for (auto &[address, m] : msgs) {
    dbc_string += QString("BO_ %1 %2: %3 XXX\n").arg(address).arg(m.name).arg(m.size);
    for (auto &[name, sig] : m.sigs) {
      const SignalExtraInfo &extra = m.extraInfo(&sig);
      dbc_string += QString(" SG_ %1 : %2|%3@%4%5 (%6,%7) [%8|%9] \"%10\" XXX\n")
                        .arg(name)
                        .arg(sig.start_bit)
                        .arg(sig.size)
                        .arg(sig.is_little_endian ? '1' : '0')
                        .arg(sig.is_signed ? '-' : '+')
                        .arg(sig.factor, 0, 'g', std::numeric_limits<double>::digits10)
                        .arg(sig.offset, 0, 'g', std::numeric_limits<double>::digits10)
                        .arg(extra.min)
                        .arg(extra.max)
                        .arg(extra.unit);
      if (!extra.comment.isEmpty()) {
        signal_comment += QString("CM_ SG_ %1 %2 \"%3\";\n").arg(address).arg(name).arg(extra.comment);
      }
      if (!extra.val_desc.isEmpty()) {
        val_desc += QString("VAL_ %1 %2 %3;\n").arg(address).arg(name).arg(extra.val_desc);
      }
    }
    dbc_string += "\n";
  }
  return dbc_string + signal_comment + val_desc;
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

void DBCManager::addSignal(const QString &id, const Signal &sig, const SignalExtraInfo &extra) {
  if (auto m = const_cast<DBCMsg *>(msg(id))) {
    auto &s = m->sigs[sig.name.c_str()];
    m->sig_extra_info[&s] = extra;
    s = sig;
    emit signalAdded(parseId(id).second, &s);
  }
}

void DBCManager::updateSignal(const QString &id, const QString &sig_name, const Signal &sig, const SignalExtraInfo &extra) {
  if (auto m = const_cast<DBCMsg *>(msg(id))) {
    // change key name
    QString new_name = QString::fromStdString(sig.name);
    auto node = m->sigs.extract(sig_name);
    node.key() = new_name;
    auto it = m->sigs.insert(std::move(node));
    auto &s = m->sigs[new_name];
    s = sig;

    m->sig_extra_info[&s] = extra;
    emit signalUpdated(&s);
  }
}

void DBCManager::removeSignal(const QString &id, const QString &sig_name) {
  if (auto m = const_cast<DBCMsg *>(msg(id))) {
    auto it = m->sigs.find(sig_name);
    if (it != m->sigs.end()) {
      emit signalRemoved(&(it->second));
      m->sig_extra_info.erase(&(it->second));
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

std::vector<const Signal *> DBCMsg::getSignals() const {
  std::vector<const Signal *> ret;
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

bool operator==(const SignalExtraInfo &l, const SignalExtraInfo &r) {
  return l.min == r.min && l.max == r.max && l.comment == r.comment &&
         l.unit == r.unit && l.val_desc == r.val_desc;
}
