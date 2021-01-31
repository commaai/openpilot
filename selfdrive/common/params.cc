#include "common/params.h"
#include <dirent.h>
#include <sys/file.h>
#include <csignal>
#include <sys/stat.h>
#include "common/util.h"

namespace {

volatile sig_atomic_t params_do_exit = 0;
void params_sig_handler(int signal) {
  params_do_exit = 1;
}

int fsync_dir(const std::string &path) {
  unique_fd fd = open(path.c_str(), O_RDONLY, 0755);
  return fd != -1 ? fsync(fd) : -1;
}

int mkdir_p(std::string path) {
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

bool ensure_dir_exists(const std::string &path) {
  // TODO: replace by std::filesystem::create_directories
  return util::file_exists(path.c_str()) ? true : mkdir_p(path) == 0;
}

bool ensure_symlink(const std::string &param_path, const std::string &key_path) {
  if (util::file_exists(key_path.c_str())) {
    // Ensure permissions are correct in case we didn't create the symlink
    return chmod(key_path.c_str(), 0777) == 0;
  } else {
    // Create temp folder
    std::string tmp_path = param_path + "/.tmp_XXXXXX";
    char *tmp_dir = mkdtemp((char *)tmp_path.c_str());
    if (tmp_dir == NULL) return false;

    char link_path[FILENAME_MAX] = {};
    snprintf(link_path, sizeof(link_path), "%s.link", tmp_dir);
    return chmod(tmp_dir, 0777) == 0 &&
           // Symlink it to temp link
           symlink(tmp_dir, link_path) == 0 &&
           // Move symlink to <params>/d
           rename(link_path, key_path.c_str()) == 0;
  }
}

}  // namespace

Params::Params(const std::string &path) : params_path(path) {}
Params::Params(bool persistent_param) {
#if defined(QCOM) || defined(QCOM2)
  params_path = !persistent_param ? "/data/params" : "/persist/comma/params";
#else
  static const std::string env_home = getenv("HOME");
  params_path = !env_home.empty() ? env_home + "/.comma/params" : "/data/params";
#endif
}

bool Params::put(const std::string &key, const char *value, size_t size) {
  // Information about safely and atomically writing a file: https://lwn.net/Articles/457667/
  // 1) Create temp file
  // 2) Write data to temp file
  // 3) fsync() the temp file
  // 4) rename the temp file to the real name
  // 5) fsync() the containing directory

  // Make sure params path exists and see if the symlink exists, otherwise create it
  const std::string path = key_path();
  if (!ensure_dir_exists(params_path) || !ensure_symlink(params_path, path)) {
    return false;
  }

  // Write value to temp.
  char tmp_path[FILENAME_MAX] = {};
  snprintf(tmp_path, sizeof(tmp_path), "%s/.tmp_value_XXXXXX", params_path.c_str());
  int tmp_fd = mkstemp(tmp_path);
  if (tmp_fd < 0) return false;

  bool ret = false;
  if (ssize_t written = write(tmp_fd, value, size); written == (ssize_t)size) {
    if (unique_fd lock_fd = open(lock_path().c_str(), O_CREAT, 0775); lock_fd != -1) {
      ret = flock(lock_fd, LOCK_EX) == 0 &&
            fchmod(tmp_fd, 0666) == 0 &&
            fsync(tmp_fd) == 0 &&
            rename(tmp_path, key_file(key.c_str()).c_str()) == 0 &&
            fsync_dir(path) == 0;
    }
  }

  close(tmp_fd);
  if (!ret) {
    remove(tmp_path);
  }
  return ret;
}

std::string Params::get(const std::string &key, bool block) {
  if (!block) return util::read_file(key_file(key));

  // blocking read until successful
  params_do_exit = 0;
  void (*prev_handler_sigint)(int) = std::signal(SIGINT, params_sig_handler);
  void (*prev_handler_sigterm)(int) = std::signal(SIGTERM, params_sig_handler);

  std::string value;
  while (!params_do_exit) {
    if (value = util::read_file(key_file(key)); value.size() > 0) {
      break;
    }
    util::sleep_for(100);  // 0.1 s
  }

  std::signal(SIGINT, prev_handler_sigint);
  std::signal(SIGTERM, prev_handler_sigterm);
  return value;
}

bool Params::read_all(std::map<std::string, std::string> &params) {
  unique_fd lock_fd = open(lock_path().c_str(), 0);
  if (lock_fd == -1 || flock(lock_fd, LOCK_SH) == -1) return false;

  DIR *d = opendir(key_path().c_str());
  if (d == nullptr) return false;

  while (struct dirent *de = readdir(d)) {
    if (isalnum(de->d_name[0]))
      params[de->d_name] = get(key_file(de->d_name));
  }
  closedir(d);
  return true;
}

bool Params::delete_value(const std::string &key) {
  unique_fd lock_fd = open(lock_path().c_str(), O_CREAT, 0775);
  if (lock_fd == -1 || flock(lock_fd, LOCK_EX) == -1) return false;

  const std::string path = key_file(key);
  return !util::file_exists(path.c_str()) || (remove(path.c_str()) == 0 && fsync_dir(key_path()) == 0);
}
