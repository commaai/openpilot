#pragma once
#include <map>
#include <string>
#include <sstream>

class Params {
private:
  const std::string params_path;
  const std::string lock_path() const {return params_path + "/.lock";}
  const std::string key_path(std::string_view key) const {return params_path + "/d/" + key.data();}
  std::optional<std::string> read_value(std::string_view key, bool block);

public:
  Params(bool persistent_param = false);
  Params(std::string_view path);
  
  bool delete_value(std::string key);
  bool read_all(std::map<std::string, std::string> &params);

  std::string get(std::string key, bool block=false) {
    return read_value(key, block).value_or("");
  }
  template <class T>
  std::optional<T> get(std::string_view param_name, bool block=false) {
    if (auto data = read_value(param_name, block)) {
      std::istringstream iss(*data);
      T value{}; 
      iss >> value;
      if (!iss.fail()) return value;
    }
    return std::nullopt;
  }
  bool getBool(std::string_view param_name, bool block=false) {
    return get<bool>(param_name).value_or(false);
  }

  bool put(std::string_view key, const char* value, size_t value_size);
  bool put(std::string key, std::string dat) {
    return put(&key[0], &dat[0], dat.length());
  }
  template <class T>
  bool put(std::string_view param_name, const T value) {
    std::string v = std::to_string(value);
    return put(param_name, v.c_str(), v.length());
  }
};
