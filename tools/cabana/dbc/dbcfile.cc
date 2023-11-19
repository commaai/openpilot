#include "tools/cabana/dbc/dbcfile.h"

#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>

DBCFile::DBCFile(const QString &dbc_file_name) {
  QFile file(dbc_file_name);
  if (file.open(QIODevice::ReadOnly)) {
    name_ = QFileInfo(dbc_file_name).baseName();
    filename = dbc_file_name;
    // Remove auto save file extension
    if (dbc_file_name.endsWith(AUTO_SAVE_EXTENSION)) {
      filename.chop(AUTO_SAVE_EXTENSION.length());
    }
    parse(file.readAll());
  } else {
    throw std::runtime_error("Failed to open file.");
  }
}

DBCFile::DBCFile(const QString &name, const QString &content) : name_(name), filename("") {
  // Open from clipboard
  parse(content);
}

bool DBCFile::save() {
  assert(!filename.isEmpty());
  if (writeContents(filename)) {
    cleanupAutoSaveFile();
    return true;
  }
  return false;
}

bool DBCFile::saveAs(const QString &new_filename) {
  filename = new_filename;
  return save();
}

bool DBCFile::autoSave() {
  return !filename.isEmpty() && writeContents(filename + AUTO_SAVE_EXTENSION);
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
  }
  return false;
}

void DBCFile::updateMsg(const MessageId &id, const QString &name, uint32_t size, const QString &node, const QString &comment) {
  auto &m = msgs[id.address];
  m.address = id.address;
  m.name = name;
  m.size = size;
  m.transmitter = node.isEmpty() ? DEFAULT_NODE_NAME : node;
  m.comment = comment;
}

cabana::Msg *DBCFile::msg(uint32_t address) {
  auto it = msgs.find(address);
  return it != msgs.end() ? &it->second : nullptr;
}

cabana::Msg *DBCFile::msg(const QString &name) {
  auto it = std::find_if(msgs.begin(), msgs.end(), [&name](auto &m) { return m.second.name == name; });
  return it != msgs.end() ? &(it->second) : nullptr;
}

int DBCFile::signalCount() {
  return std::accumulate(msgs.cbegin(), msgs.cend(), 0, [](int &n, const auto &m) { return n + m.second.sigs.size(); });
}

void DBCFile::parse(const QString &content) {
  static QRegularExpression bo_regexp(R"(^BO_ (\w+) (\w+) *: (\w+) (\w+))");
  static QRegularExpression sg_regexp(R"(^SG_ (\w+) : (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegularExpression sgm_regexp(R"(^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegularExpression msg_comment_regexp(R"(^CM_ BO_ *(\w+) *\"([^"]*)\"\s*;)");
  static QRegularExpression sg_comment_regexp(R"(^CM_ SG_ *(\w+) *(\w+) *\"([^"]*)\"\s*;)");
  static QRegularExpression val_regexp(R"(VAL_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+\".+?\"[^;]*))");

  int line_num = 0;
  QString line;
  auto dbc_assert = [&line_num, &line, this](bool condition, const QString &msg = "") {
    if (!condition) throw std::runtime_error(QString("[%1:%2]%3: %4").arg(filename).arg(line_num).arg(msg).arg(line).toStdString());
  };
  auto get_sig = [this](uint32_t address, const QString &name) -> cabana::Signal * {
    auto m = (cabana::Msg *)msg(address);
    return m ? (cabana::Signal *)m->sig(name) : nullptr;
  };

  msgs.clear();
  QTextStream stream((QString *)&content);
  cabana::Msg *current_msg = nullptr;
  int multiplexor_cnt = 0;
  while (!stream.atEnd()) {
    ++line_num;
    QString raw_line = stream.readLine();
    line = raw_line.trimmed();
    if (line.startsWith("BO_ ")) {
      multiplexor_cnt = 0;
      auto match = bo_regexp.match(line);
      dbc_assert(match.hasMatch());
      auto address = match.captured(1).toUInt();
      dbc_assert(msgs.count(address) == 0, QString("Duplicate message address: %1").arg(address));
      current_msg = &msgs[address];
      current_msg->address = address;
      current_msg->name = match.captured(2);
      current_msg->size = match.captured(3).toULong();
      current_msg->transmitter = match.captured(4).trimmed();
    } else if (line.startsWith("SG_ ")) {
      int offset = 0;
      auto match = sg_regexp.match(line);
      if (!match.hasMatch()) {
        match = sgm_regexp.match(line);
        offset = 1;
      }
      dbc_assert(match.hasMatch());
      dbc_assert(current_msg, "No Message");
      auto name = match.captured(1);
      dbc_assert(current_msg->sig(name) == nullptr, "Duplicate signal name");
      cabana::Signal s{};
      if (offset == 1) {
        auto indicator = match.captured(2);
        if (indicator == "M") {
          // Only one signal within a single message can be the multiplexer switch.
          dbc_assert(++multiplexor_cnt < 2, "Multiple multiplexor");
          s.type = cabana::Signal::Type::Multiplexor;
        } else {
          s.type = cabana::Signal::Type::Multiplexed;
          s.multiplex_value = indicator.mid(1).toInt();
        }
      }
      s.name = name;
      s.start_bit = match.captured(offset + 2).toInt();
      s.size = match.captured(offset + 3).toInt();
      s.is_little_endian = match.captured(offset + 4).toInt() == 1;
      s.is_signed = match.captured(offset + 5) == "-";
      s.factor = match.captured(offset + 6).toDouble();
      s.offset = match.captured(offset + 7).toDouble();
      s.min = match.captured(8 + offset).toDouble();
      s.max = match.captured(9 + offset).toDouble();
      s.unit = match.captured(10 + offset);
      s.receiver_name = match.captured(11 + offset).trimmed();

      current_msg->sigs.push_back(new cabana::Signal(s));
    } else if (line.startsWith("VAL_ ")) {
      auto match = val_regexp.match(line);
      dbc_assert(match.hasMatch());
      if (auto s = get_sig(match.captured(1).toUInt(), match.captured(2))) {
        QStringList desc_list = match.captured(3).trimmed().split('"');
        for (int i = 0; i < desc_list.size(); i += 2) {
          auto val = desc_list[i].trimmed();
          if (!val.isEmpty() && (i + 1) < desc_list.size()) {
            auto desc = desc_list[i + 1].trimmed();
            s->val_desc.push_back({val.toDouble(), desc});
          }
        }
      }
    } else if (line.startsWith("CM_ BO_")) {
      if (!line.endsWith("\";")) {
        int pos = stream.pos() - raw_line.length() - 1;
        line = content.mid(pos, content.indexOf("\";", pos));
      }
      auto match = msg_comment_regexp.match(line);
      dbc_assert(match.hasMatch());
      if (auto m = (cabana::Msg *)msg(match.captured(1).toUInt())) {
        m->comment = match.captured(2).trimmed();
      }
    } else if (line.startsWith("CM_ SG_ ")) {
      if (!line.endsWith("\";")) {
        int pos = stream.pos() - raw_line.length() - 1;
        line = content.mid(pos, content.indexOf("\";", pos));
      }
      auto match = sg_comment_regexp.match(line);
      dbc_assert(match.hasMatch());
      if (auto s = get_sig(match.captured(1).toUInt(), match.captured(2))) {
        s->comment = match.captured(3).trimmed();
      }
    }
  }

  for (auto &[_, m] : msgs) {
    m.update();
  }
}

QString DBCFile::generateDBC() {
  QString dbc_string, signal_comment, message_comment, val_desc;
  for (const auto &[address, m] : msgs) {
    const QString transmitter = m.transmitter.isEmpty() ? DEFAULT_NODE_NAME : m.transmitter;
    dbc_string += QString("BO_ %1 %2: %3 %4\n").arg(address).arg(m.name).arg(m.size).arg(transmitter);
    if (!m.comment.isEmpty()) {
      message_comment += QString("CM_ BO_ %1 \"%2\";\n").arg(address).arg(m.comment);
    }
    for (auto sig : m.getSignals()) {
      QString multiplexer_indicator;
      if (sig->type == cabana::Signal::Type::Multiplexor) {
        multiplexer_indicator = "M ";
      } else if (sig->type == cabana::Signal::Type::Multiplexed) {
        multiplexer_indicator = QString("m%1 ").arg(sig->multiplex_value);
      }
      dbc_string += QString(" SG_ %1 %2: %3|%4@%5%6 (%7,%8) [%9|%10] \"%11\" %12\n")
                        .arg(sig->name)
                        .arg(multiplexer_indicator)
                        .arg(sig->start_bit)
                        .arg(sig->size)
                        .arg(sig->is_little_endian ? '1' : '0')
                        .arg(sig->is_signed ? '-' : '+')
                        .arg(doubleToString(sig->factor))
                        .arg(doubleToString(sig->offset))
                        .arg(doubleToString(sig->min))
                        .arg(doubleToString(sig->max))
                        .arg(sig->unit)
                        .arg(sig->receiver_name.isEmpty() ? DEFAULT_NODE_NAME : sig->receiver_name);
      if (!sig->comment.isEmpty()) {
        signal_comment += QString("CM_ SG_ %1 %2 \"%3\";\n").arg(address).arg(sig->name).arg(sig->comment);
      }
      if (!sig->val_desc.isEmpty()) {
        QStringList text;
        for (auto &[val, desc] : sig->val_desc) {
          text << QString("%1 \"%2\"").arg(val).arg(desc);
        }
        val_desc += QString("VAL_ %1 %2 %3;\n").arg(address).arg(sig->name).arg(text.join(" "));
      }
    }
    dbc_string += "\n";
  }
  return dbc_string + message_comment + signal_comment + val_desc;
}
