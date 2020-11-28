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
  Params(std::string path);

  int write_db_value(std::string key, std::string dat);
  int write_db_value(const char* key, const char* value, size_t value_size);

  // Reads a value from the params database.
  // Inputs:
  //  key: The key to read.
  //  value: A pointer where a newly allocated string containing the db value will
  //         be written.
  //  value_sz: A pointer where the size of value will be written. Does not
  //            include the NULL terminator.
  //  persistent_param: Boolean indicating if the param store in the /persist partition is to be used.
  //                    e.g. for sensor calibration files. Will not be cleared after wipe or re-install.
  //
  // Returns: false on failure, otherwise true.
  bool read_db_value(const char* key, std::string &value);

  // Delete a value from the params database.
  // Inputs are the same as read_db_value, without value and value_sz.
  int delete_db_value(std::string key);

  // Reads a value from the params database, blocking until successful.
  // Inputs are the same as read_db_value.
  bool read_db_value_blocking(const char* key, std::string &value);

  int read_db_all(std::map<std::string, std::string> *params);

  std::string get(std::string key, bool block=false);

  template <typename T>
  bool read(const char* param_name, T* value) {
    if (std::string data; read_db_value(param_name, data)) {
      T tmp_value{};
      std::istringstream iss(data);
      iss >> tmp_value;
      if (!iss.fail()) {
        *value = tmp_value;
        return true;
      }
    }
    return false;
  }

  template <class T>
  std::optional<T> get(const char* param_name) {
    T value{};
    return read(param_name, &value) ? std::optional<T>(value) : std::nullopt;
  }

  template <class T>
  int write(const char* param_name, const T& value) {
    return write_db_value(param_name, std::to_string(value));
  }
};
