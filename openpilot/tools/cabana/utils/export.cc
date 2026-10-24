#include "tools/cabana/utils/export.h"

#include <fstream>
#include <iomanip>

#include "tools/cabana/streams/abstractstream.h"

namespace utils {

void exportToCSV(const std::string &file_name, std::optional<MessageId> msg_id) {
  std::ofstream stream(file_name, std::ios::trunc);
  if (stream) {
    stream << "time,addr,bus,data\n";
    for (auto e : msg_id ? can->events(*msg_id) : can->allEvents()) {
      stream << std::fixed << std::setprecision(3) << can->toSeconds(e->mono_time) << ","
             << "0x" << std::hex << e->address << std::dec << "," << static_cast<int>(e->src) << ",0x"
             << std::uppercase << std::hex << std::setfill('0');
      for (int i = 0; i < e->size; ++i) stream << std::setw(2) << static_cast<int>(e->dat[i]);
      stream << std::nouppercase << std::dec << "\n";
    }
  }
}

void exportSignalsToCSV(const std::string &file_name, const MessageId &msg_id) {
  std::ofstream stream(file_name, std::ios::trunc);
  if (auto msg = dbc()->msg(msg_id); msg && !msg->sigs.empty() && stream) {
    stream << "time,addr,bus";
    for (auto s : msg->sigs)
      stream << "," << s->name.c_str();
    stream << "\n";

    for (auto e : can->events(msg_id)) {
      stream << std::fixed << std::setprecision(3) << can->toSeconds(e->mono_time) << ","
             << "0x" << std::hex << e->address << std::dec << "," << static_cast<int>(e->src);
      for (auto s : msg->sigs) {
        double value = 0;
        s->getValue(e->dat, e->size, &value);
        stream << "," << std::fixed << std::setprecision(s->precision) << value;
      }
      stream << "\n";
    }
  }
}

}  // namespace utils
