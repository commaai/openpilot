#pragma once

#include <map>
#include <string>
#include <sstream>

class Params {
private:
  std::string params_path;
  const std::string lock_path() const { return params_path + "/.lock"; }
  const std::string key_path() const { return params_path + "/d"; }
  const std::string key_file(std::string key) const { return params_path + "/d/" + key; }
  std::optional<std::string> read_value(const char *key, bool block);
public:
  Params(bool persistent_param = false);
  Params(std::string path);

  bool delete_value(std::string key);
  bool read_all(std::map<std::string, std::string> &params);

  inline std::string get(std::string key, bool block = false) {
    return read_value(key.c_str(), block).value_or("");
  }
  template <class T>
  std::optional<T> get(const char *param_name, bool block = false) {
    if (auto data = read_value(param_name, block)) {
      std::istringstream iss(*data);
      T value{};
      iss >> value;
      if (!iss.fail()) return value;
    }
    return std::nullopt;
  }
  inline bool getBool(const char *param_name, bool block = false) {
    return get<bool>(param_name, block).value_or(false);
  }

  bool put(const char *key, const char *value, size_t value_size);
  inline bool put(std::string key, std::string dat) {
    return put(key.c_str(), dat.c_str(), dat.length());
  }
  template <class T>
  bool put(const char *param_name, const T value) {
    std::string v = std::to_string(value);
    return put(param_name, v.c_str(), v.length());
  }
};
