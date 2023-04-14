#include "tools/cabana/dbc/dbcfile.h"

#include <QDebug>

#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>
#include <QVector>
#include <limits>
#include <sstream>


DBCFile::DBCFile(const QString &dbc_file_name, QObject *parent) : QObject(parent) {
  QFile file(dbc_file_name);
  if (file.open(QIODevice::ReadOnly)) {
    name_ = QFileInfo(dbc_file_name).baseName();

    // Remove auto save file extension
    if (dbc_file_name.endsWith(AUTO_SAVE_EXTENSION)) {
      filename = dbc_file_name.left(dbc_file_name.length() - AUTO_SAVE_EXTENSION.length());
    } else {
      filename = dbc_file_name;
    }
    open(file.readAll());
  } else {
    throw std::runtime_error("Failed to open file.");
  }
}

DBCFile::DBCFile(const QString &name, const QString &content, QObject *parent) : QObject(parent), name_(name), filename("") {
  // Open from clipboard
  open(content);
}

void DBCFile::open(const QString &content) {
  std::istringstream stream(content.toStdString());
  auto dbc = const_cast<DBC *>(dbc_parse_from_stream(name_.toStdString(), stream));
  msgs.clear();
  for (auto &msg : dbc->msgs) {
    auto &m = msgs[msg.address];
    m.name = msg.name.c_str();
    m.size = msg.size;
    for (auto &s : msg.sigs) {
      m.sigs.push_back({});
      auto &sig = m.sigs.last();
      sig.name = s.name.c_str();
      sig.start_bit = s.start_bit;
      sig.msb = s.msb;
      sig.lsb = s.lsb;
      sig.size = s.size;
      sig.is_signed = s.is_signed;
      sig.factor = s.factor;
      sig.offset = s.offset;
      sig.is_little_endian = s.is_little_endian;
      sig.updatePrecision();
    }
  }
  parseExtraInfo(content);

  delete dbc;
}

bool DBCFile::save() {
  assert (!filename.isEmpty());
  if (writeContents(filename)) {
    cleanupAutoSaveFile();
    return true;
  } else {
    return false;
  }
}

bool DBCFile::saveAs(const QString &new_filename) {
  filename = new_filename;
  return save();
}

bool DBCFile::autoSave() {
  if (!filename.isEmpty()) {
    return writeContents(filename + AUTO_SAVE_EXTENSION);
  } else {
    return false;
  }
}

void DBCFile::cleanupAutoSaveFile() {
  if (!filename.isEmpty()) {
    QFile::remove(filename + AUTO_SAVE_EXTENSION);
  }
}

bool DBCFile::writeContents(const QString &fn) {
  QFile file(fn);
  if (file.open(QIODevice::WriteOnly)) {
    file.write(generateDBC().toUtf8());
    return true;
  } else {
    return false;
  }
}

cabana::Signal *DBCFile::addSignal(const MessageId &id, const cabana::Signal &sig) {
  if (auto m = const_cast<cabana::Msg *>(msg(id.address))) {
    m->sigs.push_back(sig);
    return &m->sigs.last();
  }

  return nullptr;
}

 cabana::Signal *DBCFile::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  if (auto m = const_cast<cabana::Msg *>(msg(id))) {
    if (auto s = (cabana::Signal *)m->sig(sig_name)) {
      *s = sig;
      return s;
    }
  }

  return nullptr;
}

cabana::Signal *DBCFile::getSignal(const MessageId &id, const QString &sig_name) {
  if (auto m = const_cast<cabana::Msg *>(msg(id))) {
    auto it = std::find_if(m->sigs.begin(), m->sigs.end(), [&](auto &s) { return s.name == sig_name; });
    if (it != m->sigs.end()) {
     return &(*it);
    }
  }
  return nullptr;
 }

void DBCFile::removeSignal(const MessageId &id, const QString &sig_name) {
  if (auto m = const_cast<cabana::Msg *>(msg(id))) {
    auto it = std::find_if(m->sigs.begin(), m->sigs.end(), [&](auto &s) { return s.name == sig_name; });
    if (it != m->sigs.end()) {
      m->sigs.erase(it);
    }
  }
}

void DBCFile::updateMsg(const MessageId &id, const QString &name, uint32_t size) {
  auto &m = msgs[id.address];
  m.name = name;
  m.size = size;
}

void DBCFile::removeMsg(const MessageId &id) {
  msgs.erase(id.address);
}

std::map<uint32_t, cabana::Msg> DBCFile::getMessages() {
  return msgs;
}

const cabana::Msg *DBCFile::msg(const MessageId &id) const {
  return msg(id.address);
}

const cabana::Msg *DBCFile::msg(uint32_t address) const {
  auto it = msgs.find(address);
  return it != msgs.end() ? &it->second : nullptr;
}

const cabana::Msg* DBCFile::msg(const QString &name) {
  for (auto &[_, msg] : msgs) {
    if (msg.name == name) {
      return &msg;
    }
  }

  return nullptr;
}


QStringList DBCFile::signalNames() const {
  // Used for autocompletion
  QStringList ret;
  for (auto const& [_, msg] : msgs) {
    for (auto sig: msg.getSignals()) {
      ret << sig->name;
    }
  }
  ret.sort();
  ret.removeDuplicates();
  return ret;
}

int DBCFile::signalCount(const MessageId &id) const {
  if (msgs.count(id.address) == 0) return 0;
  return msgs.at(id.address).sigs.size();
}

int DBCFile::signalCount() const {
  int total = 0;
  for (auto const& [_, msg] : msgs) {
    total += msg.sigs.size();
  }
  return total;
}

int DBCFile::msgCount() const {
  return msgs.size();
}

QString DBCFile::name() const {
  return name_;
}

bool DBCFile::isEmpty() const {
  return (signalCount() == 0) && name().isEmpty();
}

void DBCFile::parseExtraInfo(const QString &content) {
  static QRegularExpression bo_regexp(R"(^BO_ (\w+) (\w+) *: (\w+) (\w+))");
  static QRegularExpression sg_regexp(R"(^SG_ (\w+) : (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegularExpression sgm_regexp(R"(^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegularExpression sg_comment_regexp(R"(^CM_ SG_ *(\w+) *(\w+) *\"(.*)\";)");
  static QRegularExpression val_regexp(R"(VAL_ (\w+) (\w+) (.*);)");
  auto get_sig = [this](uint32_t address, const QString &name) -> cabana::Signal * {
    auto m = (cabana::Msg *)msg(address);
    return m ? (cabana::Signal *)m->sig(name) : nullptr;
  };

  QTextStream stream((QString *)&content);
  uint32_t address = 0;
  while (!stream.atEnd()) {
    QString line = stream.readLine().trimmed();
    if (line.startsWith("BO_ ")) {
      if (auto match = bo_regexp.match(line); match.hasMatch()) {
        address = match.captured(1).toUInt();
      }
    } else if (line.startsWith("SG_ ")) {
      int offset = 0;
      auto match = sg_regexp.match(line);
      if (!match.hasMatch()) {
        match = sgm_regexp.match(line);
        offset = 1;
      }
      if (match.hasMatch()) {
        if (auto s = get_sig(address, match.captured(1))) {
          s->min = match.captured(8 + offset);
          s->max = match.captured(9 + offset);
          s->unit = match.captured(10 + offset);
        }
      }
    } else if (line.startsWith("VAL_ ")) {
      if (auto match = val_regexp.match(line); match.hasMatch()) {
        if (auto s = get_sig(match.captured(1).toUInt(), match.captured(2))) {
          QStringList desc_list = match.captured(3).trimmed().split('"');
          for (int i = 0; i < desc_list.size(); i += 2) {
            auto val = desc_list[i].trimmed();
            if (!val.isEmpty() && (i + 1) < desc_list.size()) {
              auto desc = desc_list[i+1].trimmed();
              s->val_desc.push_back({val, desc});
            }
          }
        }
      }
    } else if (line.startsWith("CM_ SG_ ")) {
      if (auto match = sg_comment_regexp.match(line); match.hasMatch()) {
        if (auto s = get_sig(match.captured(1).toUInt(), match.captured(2))) {
          s->comment = match.captured(3).trimmed();
        }
      }
    }
  }
}

QString DBCFile::generateDBC() {
  QString dbc_string, signal_comment, val_desc;
  for (auto &[address, m] : msgs) {
    dbc_string += QString("BO_ %1 %2: %3 XXX\n").arg(address).arg(m.name).arg(m.size);
    for (auto sig : m.getSignals()) {
      dbc_string += QString(" SG_ %1 : %2|%3@%4%5 (%6,%7) [%8|%9] \"%10\" XXX\n")
                        .arg(sig->name)
                        .arg(sig->start_bit)
                        .arg(sig->size)
                        .arg(sig->is_little_endian ? '1' : '0')
                        .arg(sig->is_signed ? '-' : '+')
                        .arg(sig->factor, 0, 'g', std::numeric_limits<double>::digits10)
                        .arg(sig->offset, 0, 'g', std::numeric_limits<double>::digits10)
                        .arg(sig->min)
                        .arg(sig->max)
                        .arg(sig->unit);
      if (!sig->comment.isEmpty()) {
        signal_comment += QString("CM_ SG_ %1 %2 \"%3\";\n").arg(address).arg(sig->name).arg(sig->comment);
      }
      if (!sig->val_desc.isEmpty()) {
        QStringList text;
        for (auto &[val, desc] : sig->val_desc) {
          text << QString("%1 \"%2\"").arg(val, desc);
        }
        val_desc += QString("VAL_ %1 %2 %3;\n").arg(address).arg(sig->name).arg(text.join(" "));
      }
    }
    dbc_string += "\n";
  }
  return dbc_string + signal_comment + val_desc;
}
