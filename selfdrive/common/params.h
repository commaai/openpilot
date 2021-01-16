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
  // std::string read_value(const char *key, bool block);
public:
  Params(bool persistent_param = false);
  Params(std::string path);

  bool delete_value(std::string key);
  bool read_all(std::map<std::string, std::string> &params);

  std::string get(std::string key, bool block = false);
  template <class T>
  std::optional<T> get(std::string param_name, bool block = false) {
    std::istringstream iss(get(param_name.c_str(), block));
    T value{};
    iss >> value;
    return iss.fail() ? std::nullopt : std::optional(value);
  }
  inline bool getBool(std::string param_name, bool block = false) {
    return get<bool>(param_name, block).value_or(false);
  }
  
  bool put(const char *key, const char *value, size_t value_size);
  template <class T>
  bool put(std::string param_name, T val) {
    if constexpr (std::is_same<T, const char *>::value) {
      return put(param_name.c_str(), val, strlen(val));
    } else if constexpr (std::is_same<T, std::string>::value) {
      return put(param_name.c_str(), val.c_str(), val.length());
    } else {
      std::string v = std::to_string(val);
      return put(param_name.c_str(), v.c_str(), v.length());
    }
  }
};
