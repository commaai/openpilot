#include "tools/cabana/dbc/dbcfile.h"

#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>

DBCFile::DBCFile(const QString &dbc_file_name) {
  QFile file(dbc_file_name);
  if (file.open(QIODevice::ReadOnly)) {
    name_ = QFileInfo(dbc_file_name).baseName();
    filename = dbc_file_name;
    parse(file.readAll());
  } else {
    throw std::runtime_error("Failed to open file.");
  }
}

DBCFile::DBCFile(const QString &name, const QString &content) : name_(name), filename("") {
  parse(content);
}

bool DBCFile::save() {
  assert(!filename.isEmpty());
  return writeContents(filename);
}

bool DBCFile::saveAs(const QString &new_filename) {
  filename = new_filename;
  return save();
}

bool DBCFile::writeContents(const QString &fn) {
  QFile file(fn);
  if (file.open(QIODevice::WriteOnly)) {
    return file.write(generateDBC().toUtf8()) >= 0;
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

cabana::Signal *DBCFile::signal(uint32_t address, const QString &name) {
  auto m = msg(address);
  return m ? (cabana::Signal *)m->sig(name) : nullptr;
}

void DBCFile::parse(const QString &content) {
  msgs.clear();

  int line_num = 0;
  QString line;
  cabana::Msg *current_msg = nullptr;
  int multiplexor_cnt = 0;
  bool seen_first = false;
  QTextStream stream((QString *)&content);

  while (!stream.atEnd()) {
    ++line_num;
    QString raw_line = stream.readLine();
    line = raw_line.trimmed();

    bool seen = true;
    try {
      if (line.startsWith("BO_ ")) {
        multiplexor_cnt = 0;
        current_msg = parseBO(line);
      } else if (line.startsWith("SG_ ")) {
        parseSG(line, current_msg, multiplexor_cnt);
      } else if (line.startsWith("VAL_ ")) {
        parseVAL(line);
      } else if (line.startsWith("CM_ BO_")) {
        parseCM_BO(line, content, raw_line, stream);
      } else if (line.startsWith("CM_ SG_ ")) {
        parseCM_SG(line, content, raw_line, stream);
      } else {
        seen = false;
      }
    } catch (std::exception &e) {
      throw std::runtime_error(QString("[%1:%2]%3: %4").arg(filename).arg(line_num).arg(e.what()).arg(line).toStdString());
    }

    if (seen) {
      seen_first = true;
    } else if (!seen_first) {
      header += raw_line + "\n";
    }
  }

  for (auto &[_, m] : msgs) {
    m.update();
  }
}

cabana::Msg *DBCFile::parseBO(const QString &line) {
  static QRegularExpression bo_regexp(R"(^BO_ (?<address>\w+) (?<name>\w+) *: (?<size>\w+) (?<transmitter>\w+))");

  QRegularExpressionMatch match = bo_regexp.match(line);
  if (!match.hasMatch())
    throw std::runtime_error("Invalid BO_ line format");

  uint32_t address = match.captured("address").toUInt();
  if (msgs.count(address) > 0)
    throw std::runtime_error(QString("Duplicate message address: %1").arg(address).toStdString());

  // Create a new message object
  cabana::Msg *msg = &msgs[address];
  msg->address = address;
  msg->name = match.captured("name");
  msg->size = match.captured("size").toULong();
  msg->transmitter = match.captured("transmitter").trimmed();
  return msg;
}

void DBCFile::parseCM_BO(const QString &line, const QString &content, const QString &raw_line, const QTextStream &stream) {
  static QRegularExpression msg_comment_regexp(R"(^CM_ BO_ *(?<address>\w+) *\"(?<comment>(?:[^"\\]|\\.)*)\"\s*;)");

  QString parse_line = line;
  if (!parse_line.endsWith("\";")) {
    int pos = stream.pos() - raw_line.length() - 1;
    parse_line = content.mid(pos, content.indexOf("\";", pos));
  }
  auto match = msg_comment_regexp.match(parse_line);
  if (!match.hasMatch())
    throw std::runtime_error("Invalid message comment format");

  if (auto m = (cabana::Msg *)msg(match.captured("address").toUInt()))
    m->comment = match.captured("comment").trimmed().replace("\\\"", "\"");
}

void DBCFile::parseSG(const QString &line, cabana::Msg *current_msg, int &multiplexor_cnt) {
  static QRegularExpression sg_regexp(R"(^SG_ (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");
  static QRegularExpression sgm_regexp(R"(^SG_ (\w+) (\w+) *: (\d+)\|(\d+)@(\d+)([\+|\-]) \(([0-9.+\-eE]+),([0-9.+\-eE]+)\) \[([0-9.+\-eE]+)\|([0-9.+\-eE]+)\] \"(.*)\" (.*))");

  if (!current_msg)
    throw std::runtime_error("No Message");

  int offset = 0;
  auto match = sg_regexp.match(line);
  if (!match.hasMatch()) {
    match = sgm_regexp.match(line);
    offset = 1;
  }
  if (!match.hasMatch())
    throw std::runtime_error("Invalid SG_ line format");

  QString name = match.captured(1);
  if (current_msg->sig(name) != nullptr)
    throw std::runtime_error("Duplicate signal name");

  cabana::Signal s{};
  if (offset == 1) {
    auto indicator = match.captured(2);
    if (indicator == "M") {
      ++multiplexor_cnt;
      // Only one signal within a single message can be the multiplexer switch.
      if (multiplexor_cnt >= 2)
        throw std::runtime_error("Multiple multiplexor");

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
}

void DBCFile::parseCM_SG(const QString &line, const QString &content, const QString &raw_line, const QTextStream &stream) {
  static QRegularExpression sg_comment_regexp(R"(^CM_ SG_ *(\w+) *(\w+) *\"((?:[^"\\]|\\.)*)\"\s*;)");

  QString parse_line = line;
  if (!parse_line.endsWith("\";")) {
    int pos = stream.pos() - raw_line.length() - 1;
    parse_line = content.mid(pos, content.indexOf("\";", pos));
  }
  auto match = sg_comment_regexp.match(parse_line);
  if (!match.hasMatch())
    throw std::runtime_error("Invalid CM_ SG_ line format");

  if (auto s = signal(match.captured(1).toUInt(), match.captured(2))) {
    s->comment = match.captured(3).trimmed().replace("\\\"", "\"");
  }
}

void DBCFile::parseVAL(const QString &line) {
  static QRegularExpression val_regexp(R"(VAL_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+\".+?\"[^;]*))");

  auto match = val_regexp.match(line);
  if (!match.hasMatch())
    throw std::runtime_error("invalid VAL_ line format");

  if (auto s = signal(match.captured(1).toUInt(), match.captured(2))) {
    QStringList desc_list = match.captured(3).trimmed().split('"');
    for (int i = 0; i < desc_list.size(); i += 2) {
      auto val = desc_list[i].trimmed();
      if (!val.isEmpty() && (i + 1) < desc_list.size()) {
        auto desc = desc_list[i + 1].trimmed();
        s->val_desc.push_back({val.toDouble(), desc});
      }
    }
  }
}

QString DBCFile::generateDBC() {
  QString dbc_string, comment, val_desc;
  for (const auto &[address, m] : msgs) {
    const QString transmitter = m.transmitter.isEmpty() ? DEFAULT_NODE_NAME : m.transmitter;
    dbc_string += QString("BO_ %1 %2: %3 %4\n").arg(address).arg(m.name).arg(m.size).arg(transmitter);
    if (!m.comment.isEmpty()) {
      comment += QString("CM_ BO_ %1 \"%2\";\n").arg(address).arg(QString(m.comment).replace("\"", "\\\""));
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
        comment += QString("CM_ SG_ %1 %2 \"%3\";\n").arg(address).arg(sig->name).arg(QString(sig->comment).replace("\"", "\\\""));
      }
      if (!sig->val_desc.empty()) {
        QStringList text;
        for (auto &[val, desc] : sig->val_desc) {
          text << QString("%1 \"%2\"").arg(val).arg(desc);
        }
        val_desc += QString("VAL_ %1 %2 %3;\n").arg(address).arg(sig->name).arg(text.join(" "));
      }
    }
    dbc_string += "\n";
  }
  return header + dbc_string + comment + val_desc;
}
