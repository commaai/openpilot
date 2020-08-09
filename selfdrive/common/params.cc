#include "common/params.h"
#include "params_helper.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif  // _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/file.h>
#include <sys/stat.h>

#include <map>
#include <string>
#include <string.h>

namespace {

template <typename T>
T* null_coalesce(T* a, T* b) {
  return a != NULL ? a : b;
}

static const char* default_params_path = null_coalesce(const_cast<const char*>(getenv("PARAMS_PATH")), "/data/params");

#ifdef QCOM
static const char* persistent_params_path = null_coalesce(const_cast<const char*>(getenv("PERSISTENT_PARAMS_PATH")), "/persist/comma/params");
#else
static const char* persistent_params_path = default_params_path;
#endif

} //namespace

int write_db_value(const char* key, const char* value, size_t value_size, bool persistent_param) {
  int result;
  const char* params_path = persistent_param ? persistent_params_path : default_params_path;
  params::Params p = params::Params::Params(params_path);
  try {
    p.put(key, value);
    result = 0;
  } catch (const exception& e) {
    result = -1;
  }
  return result; 
}

int delete_db_value(const char* key, bool persistent_param) {
  int result;
  const char* params_path = persistent_param ? persistent_params_path : default_params_path;
  params::Params p = params::Params::Params(params_path);
  try {
    p._delete(key);
    result = 0;
  } catch (const exception& e) {
    result = -1;
  }
  return result;
}

int read_db_value(const char* key, char** value, size_t* value_sz, bool persistent_param) {
  int result;
  const char* params_path = persistent_param ? persistent_params_path : default_params_path;
  params::Params p = params::Params::Params(params_path);
  try {
    string ret = p.get(key, false);
    *value = const_cast<char*>(ret.c_str());
    const size_t bytesread = sizeof(*value);
    size_t temp = (size_t) bytesread;
    *value_sz = temp;
    result = 0;
  } catch (const exception& e) {
    result = -1;
  }
  return result;
}


void read_db_value_blocking(const char* key, char** value, size_t* value_sz, bool persistent_param) {
  while (1) {
    const int result = read_db_value(key, value, value_sz, persistent_param);
    if (result == 0) {
      return;
    } else {
      // Sleep for 0.1 seconds.
      usleep(100000);
    }
  }
}


int read_db_all(std::map<std::string, std::string> *params, bool persistent_param) {
  std::string value;
  const char* params_path = persistent_param ? persistent_params_path : default_params_path;
  params::Params p = params::Params::Params(params_path);
  
  std::string key_path = std::string(params_path) +"/d";
  DIR *d = opendir(key_path.c_str());
  
  struct dirent *de = NULL;
  while ((de = readdir(d))) {
    if (!isalnum(de->d_name[0])) continue;
    std::string key = std::string(de->d_name);
    try {
      value = p.get(key, false); 
    } catch (const exception& e) {
      return -1;
    }

    (*params)[key] = value;
  }

  closedir(d);
  return 0;
}

std::vector<char> read_db_bytes(const char* param_name, bool persistent_param) {
  std::vector<char> bytes;
  char* value;
  size_t sz;
  int result = read_db_value(param_name, &value, &sz, persistent_param);
  if (result == 0) {
    bytes.assign(value, value+sz);
    free(value);
  }
  return bytes;
}


bool read_db_bool(const char* param_name, bool persistent_param) {
  std::vector<char> bytes = read_db_bytes(param_name, persistent_param);
  return bytes.size() > 0 and bytes[0] == '1';
}
