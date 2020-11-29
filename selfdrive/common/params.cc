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
const std::string persistent_params_path = "/persist/comma/params";
#else
const std::string default_params_path = getenv_default("HOME", "/.comma/params", "/data/params");
const std::string persistent_params_path = default_params_path;
#endif

volatile sig_atomic_t params_do_exit = 0;
void params_sig_handler(int signal) {
  params_do_exit = 1;
}

static int fsync_dir(const std::string &path){
  int fd = open(path.c_str(), O_RDONLY, 0755);
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

static bool ensure_dir_exists(std::string path) {
  // TODO: replace by std::filesystem::create_directories
  return mkdir_p(path.c_str()) == 0;
}


Params::Params(bool persistent_param){
  params_path = persistent_param ? persistent_params_path : default_params_path;
}

Params::Params(std::string path) {
  params_path = path;
}

bool Params::put(std::string key, std::string dat){
  return put(&key[0], dat.c_str(), dat.length());
}

static bool ensure_symlink(std::string params_path) {
  std::string path = params_path + "/d";
  if (struct stat st; stat(&path[0], &st) == -1) {
    // Create temp folder
    std::string tmp_path = params_path + "/.tmp_XXXXXX";
    char* tmp_dir = mkdtemp((char*)tmp_path.c_str());
    if (tmp_dir == NULL) {
      return false;
    }
    std::string link_path = tmp_dir;
    link_path += ".link";
    return chmod(tmp_dir, 0777) == 0 &&
           // Symlink it to temp link
           symlink(tmp_dir, &link_path[0]) == 0 &&
           // Move symlink to <params>/d
           rename(&link_path[0], &path[0]) == 0;
  } else {
    // Ensure permissions are correct in case we didn't create the symlink
    return chmod(&path[0], 0777) == 0;
  }
}

bool Params::put(const char* key, const char* value, size_t size) {
  // Information about safely and atomically writing a file: https://lwn.net/Articles/457667/
  // 1) Create temp file
  // 2) Write data to temp file
  // 3) fsync() the temp file
  // 4) rename the temp file to the real name
  // 5) fsync() the containing directory

  // Make sure params path exists and see if the symlink exists, otherwise create it
  if (!ensure_dir_exists(params_path) || !ensure_symlink(params_path)) {
    return false;
  }

  // Write value to temp.
  std::string tmp_path = params_path + "/.tmp_value_XXXXXX";
  int tmp_fd = mkstemp((char*)tmp_path.c_str());
  if (tmp_fd < 0) {
    return false;
  }

  bool ret = false;
  if (ssize_t written = write(tmp_fd, value, size); written == (ssize_t)size) {
    if (int lock_fd = open(lock_path().c_str(), O_CREAT, 0775); lock_fd >= 0) {
      ret = flock(lock_fd, LOCK_EX) == 0 &&
            // change permissions to 0666 for apks
            fchmod(tmp_fd, 0666) == 0 &&
            // fsync to force persist the changes.
            fsync(tmp_fd) == 0 &&
            // Move temp into place.
            rename(&tmp_path[0], key_path(key).c_str()) == 0 &&
            // fsync parent directory
            fsync_dir(params_d_path()) == 0;
      close(lock_fd);
    }
  }

  close(tmp_fd);
  if (!ret) {
    remove(&tmp_path[0]);
  }
  return ret;
}

bool Params::delete_value(std::string key) {
  int lock_fd = open(lock_path().c_str(), O_CREAT, 0775);
  if (lock_fd == -1) return -1;

  bool deleted = false;
  if (flock(lock_fd, LOCK_EX) == 0) {
    std::string path = key_path(&key[0]);
    deleted = access(&path[0], F_OK) == -1;
    if (!deleted) {
      deleted = remove(&path[0]) == 0 &&
                fsync_dir(params_d_path()) == 0;
    }
  }
  close(lock_fd);
  return deleted;
}

std::string Params::get(std::string key, bool block){
  std::string value;
  auto read_func = block ? &Params::read_value_blocking : &Params::read_value;
  (this->*read_func)(&key[0], value);
  return value;
}

bool Params::read_value(const char* key, std::string &value) {
  FILE* f = fopen(key_path(key).c_str(), "rb");
  if (f == nullptr) {
    return false;
  }
  fseek(f, 0, SEEK_END);
  long f_len = ftell(f);
  rewind(f);
  std::string v(f_len, '\0');
  size_t num_read = fread(&v[0], f_len, 1, f);
  fclose(f);
  if (num_read != 1) {
    return false;
  }
  value = v;
  return true;
}

bool Params::read_value_blocking(const char* key, std::string &value) {
  params_do_exit = 0;
  void (*prev_handler_sigint)(int) = std::signal(SIGINT, params_sig_handler);
  void (*prev_handler_sigterm)(int) = std::signal(SIGTERM, params_sig_handler);

  while (!params_do_exit) {
    if (read_value(key, value)) {
      break;
    }
    util::sleep_for(100); // 0.1 s
  }

  std::signal(SIGINT, prev_handler_sigint);
  std::signal(SIGTERM, prev_handler_sigterm);
  return params_do_exit == 0; // Return true if we had no interrupt
}

bool Params::read_all(std::map<std::string, std::string> &params) {
  int lock_fd = open(lock_path().c_str(), 0);
  if (lock_fd < 0) return false;
  
  bool ret =false;
  if (int err = flock(lock_fd, LOCK_SH); err == 0) {
    std::string key_path = params_d_path();
    if (DIR *d = opendir(&key_path[0]); d) {
      struct dirent *de = NULL;
      while ((de = readdir(d))) {
        if (!isalnum(de->d_name[0])) continue;
        std::string key = de->d_name;
        params[key] = util::read_file(key_path + "/" + key);
      }
      closedir(d);
      ret = true;
    }
  }
  close(lock_fd);
  return ret;
}
