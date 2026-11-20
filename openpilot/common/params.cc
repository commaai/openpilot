#include "common/params.h"

#include <dirent.h>
#include <sys/file.h>
#include <sys/stat.h>
#ifdef __linux__
#include <poll.h>
#include <sys/inotify.h>
#endif

#include <algorithm>
#include <csignal>
#include <filesystem>
#include <unordered_map>

#include "common/params_keys.h"
#include "common/util.h"
#include "common/hardware/hw.h"

namespace {

const std::string REMOVED = ".removed.";

bool waits_for_disk(const std::string &key) {
  auto it = keys.find(key);
  return it != keys.end() && (it->second.flags & WAIT_FOR_DISK);
}

volatile sig_atomic_t params_do_exit = 0;
void params_sig_handler(int signal) {
  params_do_exit = 1;
}

int fsync_dir(const std::string &path) {
  int result = -1;
  int fd = HANDLE_EINTR(open(path.c_str(), O_RDONLY, 0755));
  if (fd >= 0) {
    result = HANDLE_EINTR(fsync(fd));
    HANDLE_EINTR(close(fd));
  }
  return result;
}

// Information about safely and atomically writing a file: https://lwn.net/Articles/457667/
// 1) Create temp file
// 2) Write data to temp file
// 3) fsync() the temp file
// 4) rename the temp file to the real name
// 5) fsync() the containing directory
// 3 and 5 are only for the disk
int write_param(const std::string &dir, const std::string &key, const char *value, size_t value_size, bool sync) {
  // next to the real name, since a rename between directories takes a lock that's shared by the whole filesystem
  std::string tmp_path = dir + "/.tmp_value_XXXXXX";
  int tmp_fd = mkstemp(tmp_path.data());
  if (tmp_fd < 0) return -1;

  int result = -1;
  do {
    // root writes for the owner of the directory, who couldn't read it otherwise
    struct stat st = {};
    if (geteuid() == 0 && (stat(dir.c_str(), &st) != 0 || fchown(tmp_fd, st.st_uid, st.st_gid) != 0)) break;

    ssize_t bytes_written = HANDLE_EINTR(write(tmp_fd, value, value_size));
    if (bytes_written < 0 || (size_t)bytes_written != value_size) {
      result = -20;
      break;
    }
    if (sync && (result = HANDLE_EINTR(fsync(tmp_fd))) < 0) break;
    if ((result = rename(tmp_path.c_str(), (dir + "/" + key).c_str())) < 0) break;
    if (sync) result = fsync_dir(dir);
  } while (false);

  close(tmp_fd);
  if (result != 0) {
    ::unlink(tmp_path.c_str());
  }
  return result;
}

// changes whenever a file is written, replaced or not
std::pair<uint64_t, int64_t> file_id(const std::string &path) {
  struct stat st = {};
  if (stat(path.c_str(), &st) != 0) {
    return {};
  }
#ifdef __APPLE__
  const struct timespec &mtime = st.st_mtimespec;
#else
  const struct timespec &mtime = st.st_mtim;
#endif
  return {st.st_ino, mtime.tv_sec * 1000000000LL + mtime.tv_nsec};
}

std::vector<std::string> list_dir(const std::string &path) {
  std::vector<std::string> names;
  if (DIR *d = opendir(path.c_str())) {
    struct dirent *de = NULL;
    while ((de = readdir(d))) {
      if (de->d_type != DT_DIR) {
        names.push_back(de->d_name);
      }
    }
    closedir(d);
  }
  return names;
}

// Makes key_path, a symlink to a new directory with a copy of the params in copy_from.
// It's a symlink so that the directory shows up with everything already in it
bool create_params_path(const std::string &param_path, const std::string &key_path, const std::string &copy_from = {}) {
  if (util::file_exists(key_path)) {
    return true;
  }
  if (!util::file_exists(param_path) && !util::create_directories(param_path, 0775)) {
    return false;
  }

  std::string tmp_dir = param_path + "/.tmp_XXXXXX";
  if (mkdtemp(tmp_dir.data()) == NULL) {
    return false;
  }

  // root makes it for the owner of the params on the disk
  bool ok = true;
  struct stat st = {};
  if (geteuid() == 0 && stat(copy_from.c_str(), &st) == 0) {
    ok = chown(tmp_dir.c_str(), st.st_uid, st.st_gid) == 0 && chown(param_path.c_str(), st.st_uid, st.st_gid) == 0;
  }

  for (const auto &key : list_dir(copy_from)) {
    const std::string path = copy_from + "/" + key;
    if (util::starts_with(key, ".tmp_value_")) {
      unlink(path.c_str());  // from a save that was cut short
    } else if (key[0] != '.' && access(path.c_str(), R_OK) == 0) {
      std::string value = util::read_file(path);
      ok = ok && write_param(tmp_dir, key, value.data(), value.size(), false) == 0;
    }
  }

  // this fails if someone else was first, and they may have written to theirs since
  ok = ok && symlink(tmp_dir.c_str(), key_path.c_str()) == 0;
  if (!ok) {
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
  } else if (geteuid() == 0) {
    ok = lchown(key_path.c_str(), st.st_uid, st.st_gid) == 0;
  }
  return util::file_exists(key_path);
}

class FileLock {
public:
  FileLock(const std::string &fn) {
    fd_ = HANDLE_EINTR(open(fn.c_str(), O_CREAT | O_CLOEXEC, 0775));
    locked = fd_ >= 0 && HANDLE_EINTR(flock(fd_, LOCK_EX)) == 0;
  }
  ~FileLock() {
    // a process forked while this was held has the fd too, and closing ours wouldn't unlock
    if (locked) flock(fd_, LOCK_UN);
    if (fd_ >= 0) close(fd_);
  }

  bool locked = false;

private:
  int fd_ = -1;
};

} // namespace


Params::Params(const std::string &path) {
  params_prefix = "/" + util::getenv("OPENPILOT_PREFIX", "d");
  params_path = path.empty() ? Path::params() : path;
  disk_path = path.empty() ? Path::params_disk() : "";

  // once the params are there, nothing here touches the disk
  const std::string disk = disk_path.empty() ? "" : disk_path + params_prefix;
  bool ok = util::file_exists(getParamPath()) ||
            ((disk.empty() || create_params_path(disk_path, disk)) && create_params_path(params_path, getParamPath(), disk));
  if (!ok) {
    throw std::runtime_error(util::string_format(
        "Failed to ensure params path, errno=%d, path=%s, param_prefix=%s",
        errno, params_path.c_str(), params_prefix.c_str()));
  }
}

Params::~Params() {
  if (watch_fd >= 0) {
    close(watch_fd);
  }
}

std::vector<std::string> Params::allKeys() const {
  std::vector<std::string> ret;
  for (auto &p : keys) {
    ret.push_back(p.first);
  }
  return ret;
}

bool Params::checkKey(const std::string &key) {
  return keys.find(key) != keys.end();
}

ParamKeyFlag Params::getKeyFlag(const std::string &key) {
  return static_cast<ParamKeyFlag>(keys[key].flags);
}

ParamKeyType Params::getKeyType(const std::string &key) {
  return keys[key].type;
}

std::optional<std::string> Params::getKeyDefaultValue(const std::string &key) {
  return keys[key].default_value;
}

int Params::put(const char* key, const char* value, size_t value_size, bool block) {
  block = block || waits_for_disk(key);
  int result = write_param(getParamPath(), key, value, value_size, block && disk_path.empty());
  return (result == 0 && block) ? save() : result;
}

int Params::remove(const std::string &key) {
  if (disk_path.empty()) {
    return unlink(getParamPath(key).c_str());
  }
  // save() removes it from the disk when it finds this
  int result = rename(getParamPath(key).c_str(), getParamPath(REMOVED + key).c_str());
  return (result == 0 && waits_for_disk(key)) ? save() : result;
}

int Params::save() {
  if (disk_path.empty()) {
    return 0;
  }

  // any process can be saving
  FileLock lock(disk_path + "/.lock");
  if (!lock.locked) {
    return -1;
  }

  // removes sort first, so a key that was removed and written again ends up on the disk
  auto names = list_dir(getParamPath());
  std::sort(names.begin(), names.end());

  const std::string disk = disk_path + params_prefix;
  int result = 0;
  bool removed = false;
  for (const auto &name : names) {
    if (util::starts_with(name, REMOVED)) {
      const std::string key = name.substr(REMOVED.size());
      if (unlink((disk + "/" + key).c_str()) == 0) {
        removed = true;
      } else if (errno != ENOENT) {
        result = -1;
        continue;
      }
      saved.erase(key);
      unlink(getParamPath(name).c_str());
    } else if (name[0] != '.') {
      // only what was written since the last time is read
      const auto id = file_id(getParamPath(name));
      if (id.first == 0 || saved[name] == id) {
        continue;
      }
      std::string value = util::read_file(getParamPath(name));
      if (value.empty() && access(getParamPath(name).c_str(), R_OK) != 0) {
        continue;  // can't be read, or it's gone
      }
      const std::string path = disk + "/" + name;
      bool same = value == util::read_file(path) && (!value.empty() || util::file_exists(path));
      if (same || write_param(disk, name, value.data(), value.size(), true) == 0) {
        saved[name] = id;
      } else {
        result = -1;
      }
    }
  }
  if (removed) {
    result |= fsync_dir(disk);
  }
  return result;
}

void Params::waitForWrite(int timeout_ms) {
#ifdef __linux__
  if (watch_fd < 0) {
    // every write and remove is a rename in the directory
    watch_fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (inotify_add_watch(watch_fd, getParamPath().c_str(), IN_MOVED_TO) >= 0) {
      return;  // for what was written before this
    }
  }
  struct pollfd pfd = {.fd = watch_fd, .events = POLLIN};
  if (poll(&pfd, 1, timeout_ms) > 0) {
    char events[4096];
    while (read(watch_fd, events, sizeof(events)) > 0) {}
  }
#else
  util::sleep_for(timeout_ms);
#endif
}

std::string Params::get(const std::string &key, bool block) {
  if (!block) {
    return util::read_file(getParamPath(key));
  } else {
    // blocking read until successful
    params_do_exit = 0;
    void (*prev_handler_sigint)(int) = std::signal(SIGINT, params_sig_handler);
    void (*prev_handler_sigterm)(int) = std::signal(SIGTERM, params_sig_handler);

    std::string value;
    while (!params_do_exit) {
      if (value = util::read_file(getParamPath(key)); !value.empty()) {
        break;
      }
      util::sleep_for(100);  // 0.1 s
    }

    std::signal(SIGINT, prev_handler_sigint);
    std::signal(SIGTERM, prev_handler_sigterm);
    return value;
  }
}

std::map<std::string, std::string> Params::readAll() {
  std::map<std::string, std::string> ret;
  for (const auto &key : list_dir(getParamPath())) {
    if (key[0] != '.') {
      ret[key] = util::read_file(getParamPath(key));
    }
  }
  return ret;
}

void Params::clearAll(ParamKeyFlag key_flag) {
  // 1) delete params of key_flag
  // 2) delete files that are not defined in the keys.
  for (const auto &key : list_dir(getParamPath())) {
    auto it = keys.find(key);
    if (key[0] != '.' && (it == keys.end() || (it->second.flags & key_flag))) {
      remove(key);
    }
  }
}
