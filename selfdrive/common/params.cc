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

#include "common/util.h"
#include "common/utilpp.h"


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


static int fsync_dir(const char* path){
  int result = 0;
  int fd = open(path, O_RDONLY);

  if (fd < 0){
    result = -1;
    goto cleanup;
  }

  result = fsync(fd);
  if (result < 0) {
    goto cleanup;
  }

 cleanup:
  int result_close = 0;
  if (fd >= 0){
    result_close = close(fd);
  }

  if (result_close < 0) {
    return result_close;
  } else {
    return result;
  }
}

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
  //int lock_fd = -1;
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
/*
  // Build lock path, and open lockfile
  result = snprintf(path, sizeof(path), "%s/.lock", params_path);
  if (result < 0) {
    goto cleanup;
  }
  lock_fd = open(path, O_CREAT);

  // Take lock.
  result = flock(lock_fd, LOCK_EX);
  if (result < 0) {
    goto cleanup;
  }

  // Build key path
  result = snprintf(path, sizeof(path), "%s/d/%s", params_path, key);
  if (result < 0) {
    goto cleanup;
  }

  // Delete value.
  result = remove(path);
  if (result != 0) {
    result = ERR_NO_VALUE;
    goto cleanup;
  }

  // fsync parent directory
  result = snprintf(path, sizeof(path), "%s/d", params_path);
  if (result < 0) {
    goto cleanup;
  }

  result = fsync_dir(path);
  if (result < 0) {
    goto cleanup;
  }

cleanup:
  // Release lock.
  if (lock_fd >= 0) {
    close(lock_fd);
  }
  return result;
*/
}

int read_db_value(const char* key, char** value, size_t* value_sz, bool persistent_param) {
  int lock_fd = -1;
  char path[1024];
  int result;
  const char* params_path = persistent_param ? persistent_params_path : default_params_path;
  
  result = snprintf(path, sizeof(path), "%s/.lock", params_path);
  if (result < 0) {
    goto cleanup;
  }
  lock_fd = open(path, 0);

  result = snprintf(path, sizeof(path), "%s/d/%s", params_path, key);
  if (result < 0) {
    goto cleanup;
  }

  // Take lock.
  result = flock(lock_fd, LOCK_EX);
  if (result < 0) {
    goto cleanup;
  }

  // Read value.
  // TODO(mgraczyk): If there is a lot of contention, we can release the lock
  //                 after opening the file, before reading.
  *value = static_cast<char*>(read_file(path, value_sz));
  if (*value == NULL) {
    result = -22;
    goto cleanup;
  }

  result = 0;

cleanup:
  // Release lock.
  if (lock_fd >= 0) {
    close(lock_fd);
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
  int err = 0;
  const char* params_path = persistent_param ? persistent_params_path : default_params_path;

  std::string lock_path = util::string_format("%s/.lock", params_path);

  int lock_fd = open(lock_path.c_str(), 0);
  if (lock_fd < 0) return -1;

  err = flock(lock_fd, LOCK_EX);
  if (err < 0) return err;

  std::string key_path = util::string_format("%s/d", params_path);
  DIR *d = opendir(key_path.c_str());
  if (!d) {
    close(lock_fd);
    return -1;
  }

  struct dirent *de = NULL;
  while ((de = readdir(d))) {
    if (!isalnum(de->d_name[0])) continue;
    std::string key = std::string(de->d_name);
    std::string value = util::read_file(util::string_format("%s/%s", key_path.c_str(), key.c_str()));

    (*params)[key] = value;
  }

  closedir(d);

  close(lock_fd);
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
