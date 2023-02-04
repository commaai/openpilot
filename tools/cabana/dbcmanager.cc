#include "tools/cabana/dbcmanager.h"

#include <QFile>
#include <QRegExp>
#include <QTextStream>
#include <QVector>
#include <limits>
#include <sstream>

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
  std::string content_str = content.toStdString();
  std::istringstream stream(content_str);
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
  parseExtraInfo(content);
  emit DBCFileChanged();
}

void DBCManager::parseExtraInfo(const QString &content) {
  static QRegExp bo_regexp(R"(^BO_ (\w+) (\w+) *: (\w+) (\w+))");
  static QRegExp sg_regexp(R"(^SG_ (\w+) : (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegExp sgm_regexp(R"(^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegExp sg_comment_regexp(R"(^CM_ SG_ *(\w+) *(\w+) *\"(.*)\";)");
  static QRegExp val_regexp(R"(VAL_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+\".+?\"[^;]*))");
  auto getExtraInfo = [&](uint32_t address, const QString &name) -> SignalExtraInfo * {
    if (auto m = (DBCMsg *)msg(address)) {
      if (auto sig = m->sig(name)) return &(m->sig_extra_info[sig]);
    }
    return nullptr;
  };

  QTextStream stream((QString *)&content);
  uint32_t address;
  while (!stream.atEnd()) {
    QString line = stream.readLine().trimmed();
    if (line.startsWith("BO_ ") && bo_regexp.indexIn(line) != -1) {
      address = bo_regexp.capturedTexts()[1].toUInt();
    } else if (line.startsWith("SG_ ")) {
      QStringList result;
      if (sg_regexp.indexIn(line) != -1) {
        result = sg_regexp.capturedTexts();
      } else if (sgm_regexp.indexIn(line) != -1) {
        result = sgm_regexp.capturedTexts();
        result.removeAt(0);
      }
      if (!result.isEmpty()) {
        if (auto extra = getExtraInfo(address, result[1])) {
          extra->min = result[8];
          extra->max = result[9];
          extra->unit = result[10];
        }
      }
    } else if (line.startsWith("VAL_ ") && val_regexp.indexIn(line) != -1) {
      auto result = val_regexp.capturedTexts();
      if (auto extra = getExtraInfo(result[1].toUInt(), result[2])) {
        extra->val_desc = result[3].trimmed();
      }
    } else if (line.startsWith("CM_ SG_ ") && sg_comment_regexp.indexIn(line) != -1) {
      auto result = sg_comment_regexp.capturedTexts();
      if (auto extra = getExtraInfo(result[1].toUInt(), result[2])) {
        extra->comment = result[3];
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

DBCMsg &DBCMsg::operator=(const DBCMsg &src) {
  name = src.name;
  size = src.size;
  for (auto &[name, s] : src.sigs) {
    sigs[name] = s;
    sig_extra_info[&sigs[name]] = src.extraInfo(&s);
  }
  return *this;
}
