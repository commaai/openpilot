#pragma once

#include <map>
#include <string>
#include <sstream>

#define ERR_NO_VALUE -33

class Params {
private:
  std::string params_path;
  const std::string keyFile(std::string key) const { return params_path + "/d/" + key; }
public:
  Params(bool persistent_param = false);
  Params(std::string path);

  int remove(const std::string& key);
  int getAll(std::map<std::string, std::string> *params);
  std::string get(const std::string& key, bool block = false);

  template <class T>
  std::optional<T> get(const std::string& key, bool block = false) {
    std::istringstream iss(get(key, block));
    T value{};
    iss >> value;
    return iss.fail() ? std::nullopt : std::optional(value);
  }

  inline bool getBool(const std::string& param_name, bool block = false) {
    return get<bool>(param_name, block).value_or(false);
  }

  int put(const std::string& key, const char* value, size_t value_size);

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
