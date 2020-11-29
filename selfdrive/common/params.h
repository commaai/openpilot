#pragma once
#include <stddef.h>
#include <map>
#include <string>
#include <sstream>
#include <string.h>
#define ERR_NO_VALUE -33

class Params {
private:
  std::string params_path;
  bool read_value(const char* key, std::string &value);
  // Reads a value from the params database, blocking until successful.
  bool read_value_blocking(const char* key, std::string &value);
  std::string lock_path() const {return params_path + "/.lock";}
  std::string key_path(const char *key) const {return params_path + "/d/" + key;}

public:
  Params(bool persistent_param = false);
  Params(std::string path);
  
  // Delete a value from the params database.
  int delete_value(std::string key);

  bool read_all(std::map<std::string, std::string> &params);

  std::string get(std::string key, bool block=false);
  template <class T>
  std::optional<T> get(const char* param_name, bool block=false) {
    auto read_func = block ? &Params::read_value_blocking : &Params::read_value;
    if (std::string data; (this->*read_func)(param_name, data)) {
      T value{};
      std::istringstream iss(data);
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
