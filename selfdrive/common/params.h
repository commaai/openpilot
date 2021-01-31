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
public:
  Params(bool persistent_param = false);
  Params(const std::string &path);
  bool delete_value(const std::string &key);
  bool read_all(std::map<std::string, std::string> &params);

  std::string get(const std::string &key, bool block = false);
 
  template <class T>
  std::optional<T> get(const std::string &param_name, bool block = false) {
    std::istringstream iss(get(param_name, block));
    T value{};
    iss >> value;
    return iss.fail() ? std::nullopt : std::optional(value);
  }

  inline bool getBool(const std::string &param_name, bool block = false) {
    return get<bool>(param_name, block).value_or(false);
  }

  bool put(const std::string &key, const char *value, size_t value_size);

  template <class T>
  bool put(const std::string &param_name, T val) {
    if constexpr (std::is_same<T, const char *>::value) {
      return put(param_name, val, strlen(val));
    } else if constexpr (std::is_same<T, std::string>::value) {
      return put(param_name, val.c_str(), val.length());
    } else {
      std::string v = std::to_string(val);
      return put(param_name, v.c_str(), v.length());
    }
  }
};
