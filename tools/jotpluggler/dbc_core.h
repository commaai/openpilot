#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dbc {

struct ValueDescriptionEntry {
  double value = 0.0;
  std::string text;
};

struct Signal {
  enum class Type {
    Normal = 0,
    Multiplexed,
    Multiplexor,
  };

  Type type = Type::Normal;
  std::string name;
  int start_bit = 0;
  int msb = 0;
  int lsb = 0;
  int size = 0;
  double factor = 1.0;
  double offset = 0.0;
  double min = 0.0;
  double max = 0.0;
  bool is_signed = false;
  bool is_little_endian = false;
  std::string unit;
  std::string comment;
  std::string receiver_name;
  int multiplex_value = 0;
  int multiplexor_index = -1;
  std::vector<ValueDescriptionEntry> value_descriptions;
};

struct Message {
  uint32_t address = 0;
  std::string name;
  uint32_t size = 0;
  std::string comment;
  std::string transmitter;
  std::vector<Signal> signals;
  int multiplexor_index = -1;

  const std::vector<Signal> &getSignals() const { return signals; }
};

class Database {
public:
  Database() = default;
  explicit Database(const std::filesystem::path &path);
  static Database fromContent(const std::string &content, const std::string &filename = "<dbc>");

  const Message *message(uint32_t address) const;
  const std::unordered_map<uint32_t, Message> &messages() const { return messages_; }
  std::vector<std::string> enumNames(const Signal &signal) const;

private:
  void parse(const std::string &content, const std::string &filename);
  void parseBo(const std::string &line, int line_number, Message **current_message);
  void parseSg(const std::string &line, int line_number, Message *current_message);
  void parseVal(const std::string &line, int line_number);
  void parseCmBo(const std::string &line, int line_number);
  void parseCmSg(const std::string &line, int line_number);
  void finalize();

  std::string filename_;
  std::unordered_map<uint32_t, Message> messages_;
};

void updateMsbLsb(Signal *signal);
double rawSignalValue(const Signal &signal, const uint8_t *data, size_t data_size);
std::optional<double> signalValue(const Signal &signal, const Message &message, const uint8_t *data, size_t data_size);

}  // namespace dbc
