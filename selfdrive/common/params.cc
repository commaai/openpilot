#include "selfdrive/common/params.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif  // _GNU_SOURCE

#include <dirent.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

// keep trying if x gets interrupted by a signal
#define HANDLE_EINTR(x)                                       \
  ({                                                          \
    decltype(x) ret;                                          \
    int try_cnt = 0;                                          \
    do {                                                      \
      ret = (x);                                              \
    } while (ret == -1 && errno == EINTR && try_cnt++ < 100); \
    ret;                                                      \
  })

namespace {

const std::string default_params_path = Hardware::PC() ? util::getenv_default("HOME", "/.comma/params", "/data/params")
                                                       : "/data/params";
const std::string persistent_params_path = Hardware::PC() ? default_params_path : "/persist/comma/params";

volatile sig_atomic_t params_do_exit = 0;
void params_sig_handler(int signal) {
  params_do_exit = 1;
}

int fsync_dir(const char* path) {
  int fd = HANDLE_EINTR(open(path, O_RDONLY, 0755));
  if (fd < 0) {
    return -1;
  }

  int result = fsync(fd);
  int result_close = close(fd);
  if (result_close < 0) {
    result = result_close;
  }
  return result;
}

// TODO: replace by std::filesystem::create_directories
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

static bool create_params_path(const std::string &param_path, const std::string &key_path) {
  // Make sure params path exists
  if (!util::file_exists(param_path) && mkdir_p(param_path) != 0) {
    return false;
  }

  // See if the symlink exists, otherwise create it
  if (!util::file_exists(key_path)) {
    // 1) Create temp folder
    // 2) Set permissions
    // 3) Symlink it to temp link
    // 4) Move symlink to <params>/d

    std::string tmp_path = param_path + "/.tmp_XXXXXX";
    // this should be OK since mkdtemp just replaces characters in place
    char *tmp_dir = mkdtemp((char *)tmp_path.c_str());
    if (tmp_dir == NULL) {
      return false;
    }

    if (chmod(tmp_dir, 0777) != 0) {
      return false;
    }

    std::string link_path = std::string(tmp_dir) + ".link";
    if (symlink(tmp_dir, link_path.c_str()) != 0) {
      return false;
    }

    // don't return false if it has been created by other
    if (rename(link_path.c_str(), key_path.c_str()) != 0 && errno != EEXIST) {
      return false;
    }
  }

  // Ensure permissions are correct in case we didn't create the symlink
  return chmod(key_path.c_str(), 0777) == 0;
}

static void ensure_params_path(const std::string &params_path) {
  if (!create_params_path(params_path, params_path + "/d")) {
    throw std::runtime_error(util::string_format("Failed to ensure params path, errno=%d", errno));
  }
}

class FileLock {
 public:
  FileLock(const std::string& file_name, int op) : fn_(file_name), op_(op) {}

  void lock() {
    fd_ = HANDLE_EINTR(open(fn_.c_str(), O_CREAT, 0775));
    if (fd_ < 0) {
      LOGE("Failed to open lock file %s, errno=%d", fn_.c_str(), errno);
      return;
    }
    if (HANDLE_EINTR(flock(fd_, op_)) < 0) {
      LOGE("Failed to lock file %s, errno=%d", fn_.c_str(), errno);
    }
  }

  void unlock() { close(fd_); }

private:
  int fd_ = -1, op_;
  std::string fn_;
};

std::unordered_map<std::string, uint32_t> keys = {
    {"AccessToken", CLEAR_ON_MANAGER_START},
    {"ApiCache_DriveStats", PERSISTENT},
    {"ApiCache_Device", PERSISTENT},
    {"ApiCache_Owner", PERSISTENT},
    {"ApiCache_NavDestinations", PERSISTENT},
    {"AthenadPid", PERSISTENT},
    {"CalibrationParams", PERSISTENT},
    {"CarBatteryCapacity", PERSISTENT},
    {"CarParams", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT | CLEAR_ON_IGNITION_ON},
    {"CarParamsCache", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT},
    {"CarVin", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT | CLEAR_ON_IGNITION_ON},
    {"CommunityFeaturesToggle", PERSISTENT},
    {"ControlsReady", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT | CLEAR_ON_IGNITION_ON},
    {"DisableRadar", PERSISTENT}, // WARNING: THIS DISABLES AEB
    {"EndToEndToggle", PERSISTENT},
    {"CompletedTrainingVersion", PERSISTENT},
    {"DisablePowerDown", PERSISTENT},
    {"DisableUpdates", PERSISTENT},
    {"EnableWideCamera", PERSISTENT},
    {"DoUninstall", CLEAR_ON_MANAGER_START},
    {"DongleId", PERSISTENT},
    {"GitDiff", PERSISTENT},
    {"GitBranch", PERSISTENT},
    {"GitCommit", PERSISTENT},
    {"GitRemote", PERSISTENT},
    {"GithubSshKeys", PERSISTENT},
    {"GithubUsername", PERSISTENT},
    {"HardwareSerial", PERSISTENT},
    {"HasAcceptedTerms", PERSISTENT},
    {"IsDriverViewEnabled", CLEAR_ON_MANAGER_START},
    {"IMEI", PERSISTENT},
    {"IsLdwEnabled", PERSISTENT},
    {"IsMetric", PERSISTENT},
    {"IsOffroad", CLEAR_ON_MANAGER_START},
    {"IsOnroad", PERSISTENT},
    {"IsRHD", PERSISTENT},
    {"IsTakingSnapshot", CLEAR_ON_MANAGER_START},
    {"IsUpdateAvailable", CLEAR_ON_MANAGER_START},
    {"UploadRaw", PERSISTENT},
    {"LastAthenaPingTime", PERSISTENT},
    {"LastGPSPosition", PERSISTENT},
    {"LastUpdateException", PERSISTENT},
    {"LastUpdateTime", PERSISTENT},
    {"LiveParameters", PERSISTENT},
    {"MapboxToken", PERSISTENT},
    {"NavDestination", CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_OFF},
    {"NavSettingTime24h", PERSISTENT},
    {"OpenpilotEnabledToggle", PERSISTENT},
    {"PandaFirmware", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT},
    {"PandaFirmwareHex", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT},
    {"PandaDongleId", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT},
    {"Passive", PERSISTENT},
    {"RecordFront", PERSISTENT},
    {"RecordFrontLock", PERSISTENT},  // for the internal fleet
    {"ReleaseNotes", PERSISTENT},
    {"ShouldDoUpdate", CLEAR_ON_MANAGER_START},
    {"SubscriberInfo", PERSISTENT},
    {"SshEnabled", PERSISTENT},
    {"TermsVersion", PERSISTENT},
    {"Timezone", PERSISTENT},
    {"TrainingVersion", PERSISTENT},
    {"UpdateAvailable", CLEAR_ON_MANAGER_START},
    {"UpdateFailedCount", CLEAR_ON_MANAGER_START},
    {"Version", PERSISTENT},
    {"VisionRadarToggle", PERSISTENT},
    {"Offroad_ChargeDisabled", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT},
    {"Offroad_ConnectivityNeeded", CLEAR_ON_MANAGER_START},
    {"Offroad_ConnectivityNeededPrompt", CLEAR_ON_MANAGER_START},
    {"Offroad_TemperatureTooHigh", CLEAR_ON_MANAGER_START},
    {"Offroad_PandaFirmwareMismatch", CLEAR_ON_MANAGER_START | CLEAR_ON_PANDA_DISCONNECT},
    {"Offroad_InvalidTime", CLEAR_ON_MANAGER_START},
    {"Offroad_IsTakingSnapshot", CLEAR_ON_MANAGER_START},
    {"Offroad_NeosUpdate", CLEAR_ON_MANAGER_START},
    {"Offroad_UpdateFailed", CLEAR_ON_MANAGER_START},
    {"Offroad_HardwareUnsupported", CLEAR_ON_MANAGER_START},
    {"Offroad_UnofficialHardware", CLEAR_ON_MANAGER_START},
    {"Offroad_NvmeMissing", CLEAR_ON_MANAGER_START},
    {"ForcePowerDown", CLEAR_ON_MANAGER_START},
    {"JoystickDebugMode", CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_OFF},
};

} // namespace

Params::Params(bool persistent_param) : Params(persistent_param ? persistent_params_path : default_params_path) {}

std::once_flag default_params_path_ensured;
Params::Params(const std::string &path) : params_path(path) {
  if (path == default_params_path) {
    std::call_once(default_params_path_ensured, ensure_params_path, path);
  } else {
    ensure_params_path(path);
  }
}

bool Params::checkKey(const std::string &key) {
  return keys.find(key) != keys.end();
}

int Params::put(const char* key, const char* value, size_t value_size) {
  // Information about safely and atomically writing a file: https://lwn.net/Articles/457667/
  // 1) Create temp file
  // 2) Write data to temp file
  // 3) fsync() the temp file
  // 4) rename the temp file to the real name
  // 5) fsync() the containing directory
  std::string tmp_path = params_path + "/.tmp_value_XXXXXX";
  int tmp_fd = mkstemp((char*)tmp_path.c_str());
  if (tmp_fd < 0) return -1;

  int result = -1;
  do {
    // Write value to temp.
    ssize_t bytes_written = HANDLE_EINTR(write(tmp_fd, value, value_size));
    if (bytes_written < 0 || (size_t)bytes_written != value_size) {
      result = -20;
      break;
    }

    // change permissions to 0666 for apks
    if ((result = fchmod(tmp_fd, 0666)) < 0) break;
    // fsync to force persist the changes.
    if ((result = fsync(tmp_fd)) < 0) break;

    FileLock file_lock(params_path + "/.lock", LOCK_EX);
    std::lock_guard<FileLock> lk(file_lock);

    // Move temp into place.
    std::string path = params_path + "/d/" + std::string(key);
    if ((result = rename(tmp_path.c_str(), path.c_str())) < 0) break;

    // fsync parent directory
    path = params_path + "/d";
    result = fsync_dir(path.c_str());
  } while (false);

  close(tmp_fd);
  remove(tmp_path.c_str());
  return result;
}

int Params::remove(const char *key) {
  FileLock file_lock(params_path + "/.lock", LOCK_EX);
  std::lock_guard<FileLock> lk(file_lock);
  // Delete value.
  std::string path = params_path + "/d/" + key;
  int result = ::remove(path.c_str());
  if (result != 0) {
    result = ERR_NO_VALUE;
    return result;
  }
  // fsync parent directory
  path = params_path + "/d";
  return fsync_dir(path.c_str());
}

std::string Params::get(const char *key, bool block) {
  std::string path = params_path + "/d/" + key;
  if (!block) {
    return util::read_file(path);
  } else {
    // blocking read until successful
    params_do_exit = 0;
    void (*prev_handler_sigint)(int) = std::signal(SIGINT, params_sig_handler);
    void (*prev_handler_sigterm)(int) = std::signal(SIGTERM, params_sig_handler);

    std::string value;
    while (!params_do_exit) {
      if (value = util::read_file(path); !value.empty()) {
        break;
      }
      util::sleep_for(100);  // 0.1 s
    }

    std::signal(SIGINT, prev_handler_sigint);
    std::signal(SIGTERM, prev_handler_sigterm);
    return value;
  }
}

int Params::readAll(std::map<std::string, std::string> *params) {
  FileLock file_lock(params_path + "/.lock", LOCK_SH);
  std::lock_guard<FileLock> lk(file_lock);

  std::string key_path = params_path + "/d";
  return util::read_files_in_dir(key_path, params);
}

void Params::clearAll(ParamKeyType key_type) {
  for (auto &[key, type] : keys) {
    if (type & key_type) {
      remove(key);
    }
  }
}
