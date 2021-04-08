#pragma once

#include <stddef.h>
#include <map>
#include <string>
#include <sstream>

#define ERR_NO_VALUE -33

class Params {
private:
  std::string params_path;

public:
  Params(bool persistent_param = false);
  Params(const std::string &path);

  // Delete a value
  int remove(const char *key);
  inline int remove(const std::string &key) {
    return remove (key.c_str());
  }

  // read all values
  int read_db_all(std::map<std::string, std::string> *params);

  // read a value
  std::string get(const char *key, bool block = false);

  inline std::string get(const std::string &key, bool block = false) {
    return get(key.c_str(), block);
  }

  template <class T>
  std::optional<T> get(const char *key, bool block = false) {
    std::istringstream iss(get(key, block));
    T value{};
    iss >> value;
    return iss.fail() ? std::nullopt : std::optional(value);
  }

  inline bool getBool(const std::string &key) {
    return getBool(key.c_str());
  }

  inline bool getBool(const char *key) {
    return get(key) == "1";
  }

  // write a value
  int put(const char* key, const char* val, size_t value_size);

  inline int put(const std::string &key, const std::string &val) {
    return put(key.c_str(), val.data(), val.size());
  }

  inline int putBool(const char *key, bool val) {
    return put(key, val ? "1" : "0", 1);
  }

  inline int putBool(const std::string &key, bool val) {
    return putBool(key.c_str(), val);
  }
};
