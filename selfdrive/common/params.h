#pragma once
#include <stddef.h>
#include <map>
#include <string>
#include <sstream>
#include <string.h>

class Params {
private:
  std::string params_path;
  std::optional<std::string> read_value(const char* key);
  // Reads a value from the params database, blocking until successful.
  std::optional<std::string> read_value_blocking(const char* key);
  std::string lock_path() const {return params_path + "/.lock";}
  std::string key_path(const char *key) const {return params_path + "/d/" + key;}
  std::string params_d_path() const {return params_path + "/d";}

public:
  Params(bool persistent_param = false);
  Params(std::string path);
  
  bool delete_value(std::string key);

  bool read_all(std::map<std::string, std::string> &params);

  std::string get(std::string key, bool block=false);
  template <class T>
  std::optional<T> get(const char* param_name, bool block=false) {
    auto read_func = block ? &Params::read_value_blocking : &Params::read_value;
    if (auto data = (this->*read_func)(param_name)) {
      std::istringstream iss(*data);
      T value{};
      iss >> value;
      if (!iss.fail()) {
        return value;
      }
    }
    return std::nullopt;
  }

  bool put(std::string key, std::string dat);
  bool put(const char* key, const char* value, size_t value_size);
  template <class T>
  bool put(const char* param_name, const T value) {
    std::string v = std::to_string(value);
    return put(param_name, v.c_str(), v.length());
  }
};
