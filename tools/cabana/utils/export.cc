#include "tools/cabana/utils/export.h"

#include <QFile>
#include <QTextStream>

#include "tools/cabana/streams/abstractstream.h"

namespace utils {

void exportToCSV(const QString &file_name, std::optional<MessageId> msg_id) {
  QFile file(file_name);
  if (file.open(QIODevice::ReadWrite | QIODevice::Truncate)) {
    QTextStream stream(&file);
    stream << "time,addr,bus,data\n";
    for (auto e : msg_id ? can->events(*msg_id) : can->allEvents()) {
      stream << QString::number(can->toSeconds(e->mono_time), 'f', 3) << ","
             << "0x" << QString::number(e->address, 16) << "," << e->src << ","
             << "0x" << QByteArray::fromRawData((const char *)e->dat, e->size).toHex().toUpper() << "\n";
    }
  }
}

void exportSignalsToCSV(const QString &file_name, const MessageId &msg_id) {
  QFile file(file_name);
  if (auto msg = dbc()->msg(msg_id); msg && msg->sigs.size() && file.open(QIODevice::ReadWrite | QIODevice::Truncate)) {
    QTextStream stream(&file);
    stream << "time,addr,bus";
    for (auto s : msg->sigs)
      stream << "," << s->name;
    stream << "\n";

    for (auto e : can->events(msg_id)) {
      stream << QString::number(can->toSeconds(e->mono_time), 'f', 3) << ","
             << "0x" << QString::number(e->address, 16) << "," << e->src;
      for (auto s : msg->sigs) {
        double value = 0;
        s->getValue(e->dat, e->size, &value);
        stream << "," << QString::number(value, 'f', s->precision);
      }
      stream << "\n";
    }
  }
}

}  // namespace utils
