#include "tools/cabana/dbcmanager.h"
#include <QDebug>

#include <QFile>
#include <QRegularExpression>
#include <QTextStream>
#include <QVector>
#include <limits>
#include <sstream>

bool DBCManager::open(const QString &dbc_file_name, QString *error) {
  QString opendbc_file_path = QString("%1/%2.dbc").arg(OPENDBC_FILE_PATH, dbc_file_name);
  QFile file(opendbc_file_path);
  if (file.open(QIODevice::ReadOnly)) {
    return open(dbc_file_name, file.readAll(), error);
  }
  return false;
}

bool DBCManager::open(const QString &name, const QString &content, QString *error) {
  try {
    std::istringstream stream(content.toStdString());
    auto dbc = const_cast<DBC *>(dbc_parse_from_stream(name.toStdString(), stream));
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
      }
    }
    parseExtraInfo(content);
    name_ = name;
    emit DBCFileChanged();
    delete dbc;
  } catch (std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
  return true;
}

void DBCManager::parseExtraInfo(const QString &content) {
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

QString DBCManager::generateDBC() {
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

void DBCManager::updateMsg(const MessageId &id, const QString &name, uint32_t size) {
  auto &m = msgs[id.address];
  m.name = name;
  m.size = size;
  emit msgUpdated(id.address);
}

void DBCManager::removeMsg(const MessageId &id) {
  msgs.erase(id.address);
  emit msgRemoved(id.address);
}

void DBCManager::addSignal(const MessageId &id, const cabana::Signal &sig) {
  if (auto m = const_cast<cabana::Msg *>(msg(id.address))) {
    m->sigs.push_back(sig);
    auto s = &m->sigs.last();
    emit signalAdded(id.address, s);
  }
}

void DBCManager::updateSignal(const MessageId &id, const QString &sig_name, const cabana::Signal &sig) {
  if (auto m = const_cast<cabana::Msg *>(msg(id))) {
    if (auto s = (cabana::Signal *)m->sig(sig_name)) {
      *s = sig;
      emit signalUpdated(s);
    }
  }
}

void DBCManager::removeSignal(const MessageId &id, const QString &sig_name) {
  if (auto m = const_cast<cabana::Msg *>(msg(id))) {
    auto it = std::find_if(m->sigs.begin(), m->sigs.end(), [&](auto &s) { return s.name == sig_name; });
    if (it != m->sigs.end()) {
      emit signalRemoved(&(*it));
      m->sigs.erase(it);
    }
  }
}

DBCManager *dbc() {
  static DBCManager dbc_manager(nullptr);
  return &dbc_manager;
}

