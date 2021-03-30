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
#include <csignal>
#include <string.h>

#include "common/util.h"


#if defined(QCOM) || defined(QCOM2)
const std::string default_params_path = "/data/params";
#else
const std::string default_params_path = util::getenv_default("HOME", "/.comma/params", "/data/params");
#endif

#if defined(QCOM) || defined(QCOM2)
const std::string persistent_params_path = "/persist/comma/params";
#else
const std::string persistent_params_path = default_params_path;
#endif


volatile sig_atomic_t params_do_exit = 0;
void params_sig_handler(int signal) {
  params_do_exit = 1;
}

static int fsync_dir(const char* path){
  int fd = open(path, O_RDONLY, 0755);
  if (fd < 0){
    return -1;
  }

  int result = fsync(fd);
  int result_close = close(fd);
  if (result_close < 0) {
    result = result_close;
  }
  return result;
}

static int mkdir_p(std::string path) {
  char * _path = (char *)path.c_str();

  mode_t prev_mask = umask(0);
  for (char *p = _path + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0'; // Temporarily truncate
      if (mkdir(_path, 0777) != 0) {
        if (errno != EEXIST) return -1;
      }
      *p = '/';
    }
  }
  if (mkdir(_path, 0777) != 0) {
    if (errno != EEXIST) return -1;
  }
  chmod(_path, 0777);
  umask(prev_mask);
  return 0;
}

static int ensure_dir_exists(std::string path) {
  // TODO: replace by std::filesystem::create_directories
  return mkdir_p(path.c_str());
}


Params::Params(bool persistent_param){
  params_path = persistent_param ? persistent_params_path : default_params_path;
}

Params::Params(std::string path) {
  params_path = path;
}

int Params::write_db_value(std::string key, std::string dat){
  return write_db_value(key.c_str(), dat.c_str(), dat.length());
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
  std::string path;
  std::string tmp_path;
  ssize_t bytes_written;

  // Make sure params path exists
  result = ensure_dir_exists(params_path);
  if (result < 0) {
    goto cleanup;
  }

  // See if the symlink exists, otherwise create it
  path = params_path + "/d";
  struct stat st;
  if (stat(path.c_str(), &st) == -1) {
    // Create temp folder
    path = params_path + "/.tmp_XXXXXX";

    char *t = mkdtemp((char*)path.c_str());
    if (t == NULL){
      goto cleanup;
    }
    std::string tmp_dir(t);

    // Set permissions
    result = chmod(tmp_dir.c_str(), 0777);
    if (result < 0) {
      goto cleanup;
    }

    // Symlink it to temp link
    tmp_path = tmp_dir + ".link";
    result = symlink(tmp_dir.c_str(), tmp_path.c_str());
    if (result < 0) {
      goto cleanup;
    }

    // Move symlink to <params>/d
    path = params_path + "/d";
    result = rename(tmp_path.c_str(), path.c_str());
    if (result < 0) {
      goto cleanup;
    }
  } else {
    // Ensure permissions are correct in case we didn't create the symlink
    result = chmod(path.c_str(), 0777);
    if (result < 0) {
      goto cleanup;
    }
  }

  // Write value to temp.
  tmp_path = params_path + "/.tmp_value_XXXXXX";
  tmp_fd = mkstemp((char*)tmp_path.c_str());
  bytes_written = write(tmp_fd, value, value_size);
  if (bytes_written < 0 || (size_t)bytes_written != value_size) {
    result = -20;
    goto cleanup;
  }

  // Build lock path
  path = params_path + "/.lock";
  lock_fd = open(path.c_str(), O_CREAT, 0775);

  // Build key path
  path = params_path + "/d/" + std::string(key);

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
  result = rename(tmp_path.c_str(), path.c_str());
  if (result < 0) {
    goto cleanup;
  }

  // fsync parent directory
  path = params_path + "/d";
  result = fsync_dir(path.c_str());
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
      remove(tmp_path.c_str());
    }
    close(tmp_fd);
  }
  return result;
}

int Params::delete_db_value(std::string key) {
  int lock_fd = -1;
  int result;
  std::string path;

  // Build lock path, and open lockfile
  path = params_path + "/.lock";
  lock_fd = open(path.c_str(), O_CREAT, 0775);

  // Take lock.
  result = flock(lock_fd, LOCK_EX);
  if (result < 0) {
    goto cleanup;
  }

  // Delete value.
  path = params_path + "/d/" + key;
  result = remove(path.c_str());
  if (result != 0) {
    result = ERR_NO_VALUE;
    goto cleanup;
  }

  // fsync parent directory
  path = params_path + "/d";
  result = fsync_dir(path.c_str());
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
  int r;

  if (block) {
    r = read_db_value_blocking((const char*)key.c_str(), &value, &size);
  } else {
    r = read_db_value((const char*)key.c_str(), &value, &size);
  }

  if (r == 0){
    std::string s(value, size);
    free(value);
    return s;
  } else {
    return "";
  }
}

int Params::read_db_value(const char* key, char** value, size_t* value_sz) {
  std::string path = params_path + "/d/" + std::string(key);
  *value = static_cast<char*>(read_file(path.c_str(), value_sz));
  if (*value == NULL) {
    return -22;
  }
  return 0;
}

int Params::read_db_value_blocking(const char* key, char** value, size_t* value_sz) {
  params_do_exit = 0;
  void (*prev_handler_sigint)(int) = std::signal(SIGINT, params_sig_handler);
  void (*prev_handler_sigterm)(int) = std::signal(SIGTERM, params_sig_handler);

  while (!params_do_exit) {
    const int result = read_db_value(key, value, value_sz);
    if (result == 0) {
      break;
    } else {
      util::sleep_for(100); // 0.1 s
    }
  }

  std::signal(SIGINT, prev_handler_sigint);
  std::signal(SIGTERM, prev_handler_sigterm);
  return params_do_exit; // Return 0 if we had no interrupt
}

int Params::read_db_all(std::map<std::string, std::string> *params) {
  int err = 0;

  std::string lock_path = params_path + "/.lock";

  int lock_fd = open(lock_path.c_str(), 0);
  if (lock_fd < 0) return -1;

  err = flock(lock_fd, LOCK_SH);
  if (err < 0) {
    close(lock_fd);
    return err;
  }

  std::string key_path = params_path + "/d";
  DIR *d = opendir(key_path.c_str());
  if (!d) {
    close(lock_fd);
    return -1;
  }

  struct dirent *de = NULL;
  while ((de = readdir(d))) {
    if (!isalnum(de->d_name[0])) continue;
    std::string key = std::string(de->d_name);
    std::string value = util::read_file(key_path + "/" + key);

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
