#include "params.h"
#include <dirent.h>
#include <sys/file.h>
#include <csignal>
<<<<<<< HEAD
#include <string.h>

#include "common/util.h"
=======
#include "utilpp.h"
>>>>>>> rebase master

#if defined(QCOM) || defined(QCOM2)
const std::string default_params_path = "/data/params";
const std::string persistent_params_path = "/persist/comma/params";
#else
static std::string getenv_default(const char* env_var, const char * suffix, const char* default_val) {
  const char* env_val = getenv(env_var); 
  return env_val != nullptr ? std::string(env_val) + suffix : default_val;
}
const std::string default_params_path = getenv_default("HOME", "/.comma/params", "/data/params");
const std::string persistent_params_path = default_params_path;
#endif

volatile sig_atomic_t params_do_exit = 0;
void params_sig_handler(int signal) {
  params_do_exit = 1;
}

static int fsync_dir(std::string_view path){
  if (unique_fd fd = open(path.data(), O_RDONLY, 0755); fd != -1)
    return fsync(fd);
  return -1;
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
  return util::file_exists(path) ? true : mkdir_p(path.c_str()) == 0;
}

static bool ensure_symlink(std::string_view path) {
  if (util::file_exists(path)) {
    // Ensure permissions are correct in case we didn't create the symlink
    return chmod(&path[0], 0777) == 0;
  }

  // Create temp folder
  char tmp_path[FILENAME_MAX] = {};
  snprintf(tmp_path, sizeof(tmp_path), "%s/.tmp_XXXXXX", path.data());
  char* tmp_dir = mkdtemp(tmp_path);
  if (tmp_dir == NULL) return false;

  char link_path[FILENAME_MAX] = {};
  snprintf(link_path, sizeof(link_path), "%s/.link", tmp_dir);
  return chmod(tmp_dir, 0777) == 0 &&
         // Symlink it to temp link
         symlink(tmp_dir, link_path) == 0 &&
         // Move symlink to <params>/d
         rename(link_path, &path[0]) == 0;
}

static std::optional<std::string> read_file_string(std::string_view key_path) {
  unique_fd f = open(key_path.data(), O_RDONLY);
  if (f == -1) return std::nullopt;

  if (long size = lseek(f, 0, SEEK_END); size > 0) {
    lseek(f, 0, SEEK_SET);
    if (std::string value(size, '\0'); read(f, &value[0], size) == size)
      return value;
  }
  return std::nullopt;
}

Params::Params(bool persistent_param) : params_path(persistent_param ? persistent_params_path : default_params_path) {}
Params::Params(std::string_view path) : params_path(path) {}

bool Params::put(std::string_view key, const char *value, size_t size) {
  // Information about safely and atomically writing a file: https://lwn.net/Articles/457667/
  // 1) Create temp file
  // 2) Write data to temp file
  // 3) fsync() the temp file
  // 4) rename the temp file to the real name
  // 5) fsync() the containing directory

  // Make sure params path exists and see if the symlink exists, otherwise create it
  char key_parent_path[FILENAME_MAX] = {};
  snprintf(key_parent_path, sizeof(key_parent_path), "%s/d", params_path.c_str());
  if (!ensure_dir_exists(params_path) || !ensure_symlink(key_parent_path)) {
    return false;
  }

  // Write value to temp.
  char tmp_path[FILENAME_MAX] = {};
  snprintf(tmp_path, sizeof(tmp_path), "%s/.tmp_value_XXXXXX", params_path.c_str());
  int tmp_fd = mkstemp(tmp_path);
  if (tmp_fd < 0) return false;

  bool ret = false;
  if (ssize_t written = write(tmp_fd, value, size); written == (ssize_t)size) {
    if (unique_fd lock_fd = open(lock_path().c_str(), O_CREAT, 0775); lock_fd >= 0) {
      ret = flock(lock_fd, LOCK_EX) == 0 &&
            // change permissions to 0666 for apks
            fchmod(tmp_fd, 0666) == 0 &&
            // fsync to force persist the changes.
            fsync(tmp_fd) == 0 &&
            // Move temp into place.
            rename(tmp_path, key_path(key).c_str()) == 0 &&
            // fsync parent directory
            fsync_dir(key_parent_path) == 0;
    }
  }

  close(tmp_fd);
  if (!ret) {
    remove(tmp_path);
  }
  return ret;
}

bool Params::delete_value(std::string key) {
  unique_fd lock_fd = open(lock_path().c_str(), O_CREAT, 0775);
  if (lock_fd == -1) return false;

  if (flock(lock_fd, LOCK_EX) == -1) return false;

  const std::string path = key_path(&key[0]);
  return !util::file_exists(path) ||
         (remove(&path[0]) == 0 && fsync_dir(params_path + "/d") == 0);
}

std::optional<std::string> Params::read_value(std::string_view key, bool block) {
  if (!block) {
    return read_file_string(key_path(key));
  }

  // blocking read until successful
  params_do_exit = 0;
  void (*prev_handler_sigint)(int) = std::signal(SIGINT, params_sig_handler);
  void (*prev_handler_sigterm)(int) = std::signal(SIGTERM, params_sig_handler);

  std::optional<std::string> ret = std::nullopt;
  while (!params_do_exit) {
    if (ret = read_value(key); ret) {
      break;
    }
    util::sleep_for(100); // 0.1 s
  }

  std::signal(SIGINT, prev_handler_sigint);
  std::signal(SIGTERM, prev_handler_sigterm);
  return ret;
}

bool Params::read_all(std::map<std::string, std::string> &params) {
  unique_fd lock_fd = open(lock_path().c_str(), 0);
  if (lock_fd < 0) return false;
  
  if (flock(lock_fd, LOCK_SH) == -1) return false;

  const std::string key_parent_path = params_path + "/d";
  DIR *d = opendir(&key_parent_path[0]);
  if (d == nullptr) return false;

  while (struct dirent *de = readdir(d)) {
    if (isalnum(de->d_name[0])) {
      params[de->d_name] = read_file_string(key_parent_path + "/" + de->d_name).value_or("");
    }
  }
  closedir(d);
  return true;
}
