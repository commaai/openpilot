#pragma once
#include <stddef.h>
#include <map>
#include <string>
#include <vector>

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
  // Returns: Negative on failure, otherwise 0.
  int read_db_value(const char* key, char** value, size_t* value_sz);

  // Delete a value from the params database.
  // Inputs are the same as read_db_value, without value and value_sz.
  int delete_db_value(std::string key);

  // Reads a value from the params database, blocking until successful.
  // Inputs are the same as read_db_value.
  int read_db_value_blocking(const char* key, char** value, size_t* value_sz);

  int read_db_all(std::map<std::string, std::string> *params);
  std::vector<char> read_db_bytes(const char* param_name);
  bool read_db_bool(const char* param_name);

  std::string get(std::string key, bool block=false);
};
