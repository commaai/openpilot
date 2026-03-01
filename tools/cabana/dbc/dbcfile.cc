#include "tools/cabana/dbc/dbcfile.h"

#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QString>

DBCFile::DBCFile(const std::string &dbc_file_name) {
  QFile file(QString::fromStdString(dbc_file_name));
  if (file.open(QIODevice::ReadOnly)) {
    name_ = QFileInfo(QString::fromStdString(dbc_file_name)).baseName().toStdString();
    filename = dbc_file_name;
    parse(file.readAll());
  } else {
    throw std::runtime_error("Failed to open file.");
  }
}

DBCFile::DBCFile(const std::string &name, const std::string &content) : name_(name), filename("") {
  parse(QString::fromStdString(content));
}

bool DBCFile::save() {
  assert(!filename.empty());
  return writeContents(filename);
}

bool DBCFile::saveAs(const std::string &new_filename) {
  filename = new_filename;
  return save();
}

bool DBCFile::writeContents(const std::string &fn) {
  QFile file(QString::fromStdString(fn));
  if (file.open(QIODevice::WriteOnly)) {
    std::string content = generateDBC();
    return file.write(content.c_str(), content.size()) >= 0;
  }
  return false;
}

void DBCFile::updateMsg(const MessageId &id, const std::string &name, uint32_t size, const std::string &node, const std::string &comment) {
  auto &m = msgs[id.address];
  m.address = id.address;
  m.name = name;
  m.size = size;
  m.transmitter = node.empty() ? DEFAULT_NODE_NAME : node;
  m.comment = comment;
}

cabana::Msg *DBCFile::msg(uint32_t address) {
  auto it = msgs.find(address);
  return it != msgs.end() ? &it->second : nullptr;
}

cabana::Msg *DBCFile::msg(const std::string &name) {
  auto it = std::find_if(msgs.begin(), msgs.end(), [&name](auto &m) { return m.second.name == name; });
  return it != msgs.end() ? &(it->second) : nullptr;
}

cabana::Signal *DBCFile::signal(uint32_t address, const std::string &name) {
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
      throw std::runtime_error(QString("[%1:%2]%3: %4").arg(QString::fromStdString(filename)).arg(line_num).arg(e.what()).arg(line).toStdString());
    }

    if (seen) {
      seen_first = true;
    } else if (!seen_first) {
      header += raw_line.toStdString() + "\n";
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
  msg->name = match.captured("name").toStdString();
  msg->size = match.captured("size").toULong();
  msg->transmitter = match.captured("transmitter").trimmed().toStdString();
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
    m->comment = match.captured("comment").trimmed().replace("\\\"", "\"").toStdString();
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

  std::string name = match.captured(1).toStdString();
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
  s.unit = match.captured(10 + offset).toStdString();
  s.receiver_name = match.captured(11 + offset).trimmed().toStdString();
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

  if (auto s = signal(match.captured(1).toUInt(), match.captured(2).toStdString())) {
    s->comment = match.captured(3).trimmed().replace("\\\"", "\"").toStdString();
  }
}

void DBCFile::parseVAL(const QString &line) {
  static QRegularExpression val_regexp(R"(VAL_ (\w+) (\w+) (\s*[-+]?[0-9]+\s+\".+?\"[^;]*))");

  auto match = val_regexp.match(line);
  if (!match.hasMatch())
    throw std::runtime_error("invalid VAL_ line format");

  if (auto s = signal(match.captured(1).toUInt(), match.captured(2).toStdString())) {
    QStringList desc_list = match.captured(3).trimmed().split('"');
    for (int i = 0; i < desc_list.size(); i += 2) {
      auto val = desc_list[i].trimmed();
      if (!val.isEmpty() && (i + 1) < desc_list.size()) {
        auto desc = desc_list[i + 1].trimmed();
        s->val_desc.push_back({val.toDouble(), desc.toStdString()});
      }
    }
  }
}

std::string DBCFile::generateDBC() {
  std::string dbc_string, comment, val_desc;
  for (const auto &[address, m] : msgs) {
    const std::string &transmitter = m.transmitter.empty() ? DEFAULT_NODE_NAME : m.transmitter;
    dbc_string += "BO_ " + std::to_string(address) + " " + m.name + ": " + std::to_string(m.size) + " " + transmitter + "\n";
    if (!m.comment.empty()) {
      std::string escaped_comment = m.comment;
      // Replace " with \"
      for (size_t pos = 0; (pos = escaped_comment.find('"', pos)) != std::string::npos; pos += 2)
        escaped_comment.replace(pos, 1, "\\\"");
      comment += "CM_ BO_ " + std::to_string(address) + " \"" + escaped_comment + "\";\n";
    }
    for (auto sig : m.getSignals()) {
      std::string multiplexer_indicator;
      if (sig->type == cabana::Signal::Type::Multiplexor) {
        multiplexer_indicator = "M ";
      } else if (sig->type == cabana::Signal::Type::Multiplexed) {
        multiplexer_indicator = "m" + std::to_string(sig->multiplex_value) + " ";
      }
      const std::string &recv = sig->receiver_name.empty() ? DEFAULT_NODE_NAME : sig->receiver_name;
      dbc_string += " SG_ " + sig->name + " " + multiplexer_indicator + ": " +
                    std::to_string(sig->start_bit) + "|" + std::to_string(sig->size) + "@" +
                    std::string(1, sig->is_little_endian ? '1' : '0') +
                    std::string(1, sig->is_signed ? '-' : '+') +
                    " (" + doubleToString(sig->factor) + "," + doubleToString(sig->offset) + ")" +
                    " [" + doubleToString(sig->min) + "|" + doubleToString(sig->max) + "]" +
                    " \"" + sig->unit + "\" " + recv + "\n";
      if (!sig->comment.empty()) {
        std::string escaped_comment = sig->comment;
        for (size_t pos = 0; (pos = escaped_comment.find('"', pos)) != std::string::npos; pos += 2)
          escaped_comment.replace(pos, 1, "\\\"");
        comment += "CM_ SG_ " + std::to_string(address) + " " + sig->name + " \"" + escaped_comment + "\";\n";
      }
      if (!sig->val_desc.empty()) {
        std::string text;
        for (auto &[val, desc] : sig->val_desc) {
          if (!text.empty()) text += " ";
          char val_buf[64];
          snprintf(val_buf, sizeof(val_buf), "%g", val);
          text += std::string(val_buf) + " \"" + desc + "\"";
        }
        val_desc += "VAL_ " + std::to_string(address) + " " + sig->name + " " + text + ";\n";
      }
    }
    dbc_string += "\n";
  }
  return header + dbc_string + comment + val_desc;
}
