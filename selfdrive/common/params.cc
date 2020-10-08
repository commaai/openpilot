#include "common/params.h"

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
#include <iostream>
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
  int fd = open(path, O_RDONLY, 0755);

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

static int ensure_dir_exists(const char* path) {
  struct stat st;
  if (stat(path, &st) == -1) {
    return mkdir(path, 0700);
  }
  return 0;
}

Params::Params(bool persistent_param){
  const char * path = persistent_param ? persistent_params_path : default_params_path;
  params_path = std::string(path);
}

Params::Params(std::string path) {
  params_path = path;
}

Params::Params(char * path) {
  params_path = std::string(path);
}


void Params::put(std::string key, std::string dat){
  write_db_value(key.c_str(), dat.c_str(), dat.length());
}

int Params::write_db_value(const char* key, const char* value, size_t value_size) {
  // Information about safely and atomically writing a file: https://lwn.net/Articles/457667/
  // 1) Create temp file
  // 2) Write data to temp file
  // 3) fsync() the temp file
  // 4) rename the temp file to the real name
  // 5) fsync() the containing directory

  int lock_fd = -1;
  int tmp_fd = -1;
  int result;
  char tmp_path[1024];
  char path[1024];
  char *tmp_dir;
  ssize_t bytes_written;

  // Make sure params path exists
  result = ensure_dir_exists(params_path.c_str());
  if (result < 0) {
    goto cleanup;
  }

  result = snprintf(path, sizeof(path), "%s/d", params_path.c_str());
  if (result < 0) {
    goto cleanup;
  }

  // See if the symlink exists, otherwise create it
  struct stat st;
  if (stat(path, &st) == -1) {
    // Create temp folder
    result = snprintf(path, sizeof(path), "%s/.tmp_XXXXXX", params_path.c_str());
    if (result < 0) {
      goto cleanup;
    }
    tmp_dir = mkdtemp(path);
    if (tmp_dir == NULL){
      goto cleanup;
    }

    // Set permissions
    result = chmod(tmp_dir, 0777);
    if (result < 0) {
      goto cleanup;
    }

    // Symlink it to temp link
    result = snprintf(tmp_path, sizeof(tmp_path), "%s.link", tmp_dir);
    if (result < 0) {
      goto cleanup;
    }
    result = symlink(tmp_dir, tmp_path);
    if (result < 0) {
      goto cleanup;
    }

    // Move symlink to <params>/d
    result = snprintf(path, sizeof(path), "%s/d", params_path.c_str());
    if (result < 0) {
      goto cleanup;
    }
    result = rename(tmp_path, path);
    if (result < 0) {
      goto cleanup;
    }
  }

  // Write value to temp.
  result =
    snprintf(tmp_path, sizeof(tmp_path), "%s/.tmp_value_XXXXXX", params_path.c_str());
  if (result < 0) {
    goto cleanup;
  }

  tmp_fd = mkstemp(tmp_path);
  bytes_written = write(tmp_fd, value, value_size);
  if (bytes_written != value_size) {
    result = -20;
    goto cleanup;
  }

  // Build lock path
  result = snprintf(path, sizeof(path), "%s/.lock", params_path.c_str());
  if (result < 0) {
    goto cleanup;
  }
  lock_fd = open(path, O_CREAT, 0775);

  // Build key path
  result = snprintf(path, sizeof(path), "%s/d/%s", params_path.c_str(), key);
  if (result < 0) {
    goto cleanup;
  }

  // Take lock.
  result = flock(lock_fd, LOCK_EX);
  if (result < 0) {
    goto cleanup;
  }

  // change permissions to 0666 for apks
  result = fchmod(tmp_fd, 0666);
  if (result < 0) {
    goto cleanup;
  }

  // fsync to force persist the changes.
  result = fsync(tmp_fd);
  if (result < 0) {
    goto cleanup;
  }

  // Move temp into place.
  result = rename(tmp_path, path);
  if (result < 0) {
    goto cleanup;
  }

  // fsync parent directory
  result = snprintf(path, sizeof(path), "%s/d", params_path.c_str());
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
  if (tmp_fd >= 0) {
    if (result < 0) {
      remove(tmp_path);
    }
    close(tmp_fd);
  }
  return result;
}

void Params::rm(std::string key){
  delete_db_value(key.c_str());
}

int Params::delete_db_value(const char* key) {
  int lock_fd = -1;
  int result;
  char path[1024];

  // Build lock path, and open lockfile
  result = snprintf(path, sizeof(path), "%s/.lock", params_path.c_str());
  if (result < 0) {
    goto cleanup;
  }
  lock_fd = open(path, O_CREAT, 0775);

  // Take lock.
  result = flock(lock_fd, LOCK_EX);
  if (result < 0) {
    goto cleanup;
  }

  // Build key path
  result = snprintf(path, sizeof(path), "%s/d/%s", params_path.c_str(), key);
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
  result = snprintf(path, sizeof(path), "%s/d", params_path.c_str());
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
}

std::string Params::get(std::string key, bool block){
  char* value;
  size_t size;

  if (block){
    if (0 == read_db_value((const char*)key.c_str(), &value, &size)){
      return std::string(value, size);
    } else {
      return "";
    }
  } else {
    read_db_value_blocking((const char*)key.c_str(), &value, &size);
    return std::string(value, size);
  }
}

int Params::read_db_value(const char* key, char** value, size_t* value_sz) {
  char path[1024];

  int result = snprintf(path, sizeof(path), "%s/d/%s", params_path.c_str(), key);
  if (result < 0) {
    return result;
  }

  *value = static_cast<char*>(read_file(path, value_sz));
  if (*value == NULL) {
    return -22;
  }
  return 0;
}

void Params::read_db_value_blocking(const char* key, char** value, size_t* value_sz) {
  while (1) {
    const int result = read_db_value(key, value, value_sz);
    if (result == 0) {
      return;
    } else {
      // Sleep for 0.1 seconds.
      usleep(100000);
    }
  }
}

int Params::read_db_all(std::map<std::string, std::string> *params) {
  int err = 0;

  std::string lock_path = util::string_format("%s/.lock", params_path.c_str());

  int lock_fd = open(lock_path.c_str(), 0);
  if (lock_fd < 0) return -1;

  err = flock(lock_fd, LOCK_SH);
  if (err < 0) {
    close(lock_fd);
    return err;
  }

  std::string key_path = util::string_format("%s/d", params_path.c_str());
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

std::vector<char> Params::read_db_bytes(const char* param_name) {
  std::vector<char> bytes;
  char* value;
  size_t sz;
  int result = read_db_value(param_name, &value, &sz);
  if (result == 0) {
    bytes.assign(value, value+sz);
    free(value);
  }
  return bytes;
}

bool Params::read_db_bool(const char* param_name) {
  std::vector<char> bytes = read_db_bytes(param_name);
  return bytes.size() > 0 and bytes[0] == '1';
}
