#include "logger.hpp"

#include <assert.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <fstream>
#include <streambuf>
#include <string>
#include <vector>

#ifdef QCOM
#include <cutils/properties.h>
#endif

#include "cereal/gen/cpp/log.capnp.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "common/utilpp.h"
#include "common/version.h"

typedef cereal::Sentinel::SentinelType SentinelType;

static int mkpath(char* file_path) {
  assert(file_path && *file_path);
  char* p;
  for (p = strchr(file_path + 1, '/'); p; p = strchr(p + 1, '/')) {
    *p = '\0';
    if (mkdir(file_path, 0777) == -1) {
      if (errno != EEXIST) {
        *p = '/';
        return -1;
      }
    }
    *p = '/';
  }
  return 0;
}

static void log_sentinel(std::shared_ptr<LoggerHandle> log, SentinelType type) {
  capnp::MallocMessageBuilder msg;
  auto event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto sen = event.initSentinel();
  sen.setType(type);

  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  log->log(bytes.begin(), bytes.size(), true);
}

static void append_property(const char* key, const char* value, void *cookie) {
  std::vector<std::pair<std::string, std::string> > *properties =
    (std::vector<std::pair<std::string, std::string> > *)cookie;

  properties->push_back(std::make_pair(std::string(key), std::string(value)));
}

static kj::Array<capnp::word> gen_init_data() {
  capnp::MallocMessageBuilder msg;
  auto event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto init = event.initInitData();

  init.setDeviceType(cereal::InitData::DeviceType::NEO);
  init.setVersion(capnp::Text::Reader(COMMA_VERSION));

  std::ifstream cmdline_stream("/proc/cmdline");
  std::vector<std::string> kernel_args;
  std::string buf;
  while (cmdline_stream >> buf) {
    kernel_args.push_back(buf);
  }

  auto lkernel_args = init.initKernelArgs(kernel_args.size());
  for (int i = 0; i < kernel_args.size(); i++) {
    lkernel_args.set(i, kernel_args[i]);
  }

  init.setKernelVersion(util::read_file("/proc/version"));

#ifdef QCOM
  {
    std::vector<std::pair<std::string, std::string> > properties;
    property_list(append_property, (void*)&properties);

    auto lentries = init.initAndroidProperties().initEntries(properties.size());
    for (int i = 0; i < properties.size(); i++) {
      auto lentry = lentries[i];
      lentry.setKey(properties[i].first);
      lentry.setValue(properties[i].second);
    }
  }
#endif

  const char* dongle_id = getenv("DONGLE_ID");
  if (dongle_id) {
    init.setDongleId(std::string(dongle_id));
  }

  const char* clean = getenv("CLEAN");
  if (!clean) {
    init.setDirty(true);
  }

  std::vector<char> git_commit = read_db_bytes("GitCommit");
  if (git_commit.size() > 0) {
    init.setGitCommit(capnp::Text::Reader(git_commit.data(), git_commit.size()));
  }
  std::vector<char> git_branch = read_db_bytes("GitBranch");
  if (git_branch.size() > 0) {
    init.setGitBranch(capnp::Text::Reader(git_branch.data(), git_branch.size()));
  }
  std::vector<char> git_remote = read_db_bytes("GitRemote");
  if (git_remote.size() > 0) {
    init.setGitRemote(capnp::Text::Reader(git_remote.data(), git_remote.size()));
  }
  std::vector<char> passive = read_db_bytes("Passive");
  init.setPassive(passive.size() > 0 && passive[0] == '1');
  // log params
  std::map<std::string, std::string> params;
  read_db_all(&params);
  auto lparams = init.initParams().initEntries(params.size());
  int i = 0;
  for (auto& kv : params) {
    auto lentry = lparams[i];
    lentry.setKey(kv.first);
    lentry.setValue(kv.second);
    i++;
  }
  return capnp::messageToFlatArray(msg);
}

bool LoggerHandle::open(const char* segment_path, const char* log_name, bool has_qlog) {
  char log_path[4096] = {};
  char qlog_path[4096] = {};
  char lock_path[4096] = {};

  snprintf(log_path, sizeof(log_path), "%s/%s.bz2", segment_path, log_name);
  snprintf(qlog_path, sizeof(qlog_path), "%s/qlog.bz2", segment_path);
  snprintf(lock_path, sizeof(lock_path), "%s.lock", log_path);

  int err = mkpath(log_path);
  if (err) return false;

  FILE* lock_file = fopen(lock_path, "wb");
  if (lock_file == NULL) return false;
  fclose(lock_file);

  lock_path_ = lock_path;

  log_file_ = fopen(log_path, "wb");
  if (log_file_ == NULL) return false;

  bz_file_ = BZ2_bzWriteOpen(&err, log_file_, 9, 0, 30);
  if (err != BZ_OK) return false;

  if (has_qlog) {
    qlog_file_ = fopen(qlog_path, "wb");
    if (qlog_file_ == NULL) return false;

    bz_qlog_ = BZ2_bzWriteOpen(&err, qlog_file_, 9, 0, 30);
    if (err != BZ_OK) return false;
  }
  return true;
}

void LoggerHandle::log(uint8_t* data, size_t data_size, bool in_qlog) {
  const std::lock_guard<std::mutex> lock(mutex_);
  int bzerror;
  BZ2_bzWrite(&bzerror, bz_file_, data, data_size);
  if (in_qlog && bz_qlog_ != NULL) {
    BZ2_bzWrite(&bzerror, bz_qlog_, data, data_size);
  }
}

LoggerHandle::~LoggerHandle() {
  int bzerror;
  if (bz_file_) BZ2_bzWriteClose(&bzerror, bz_file_, 0, NULL, NULL);
  if (bz_qlog_) BZ2_bzWriteClose(&bzerror, bz_qlog_, 0, NULL, NULL);
  if (qlog_file_) fclose(qlog_file_);
  if (log_file_) fclose(log_file_);
  if (lock_path_.length() > 0) unlink(lock_path_.c_str());
}

void Logger::init(const char* log_name, bool has_qlog) {
  umask(0);

  init_data_ = gen_init_data();
  has_qlog = has_qlog;

  time_t rawtime = time(NULL);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);
  strftime(route_name_, sizeof(route_name_), "%Y-%m-%d--%H-%M-%S", &timeinfo);
  log_name_ = log_name;
}

bool Logger::next(const char* root_path) {
  if (cur_handle_) log_sentinel(cur_handle_, SentinelType::END_OF_SEGMENT);

  snprintf(segment_path_, sizeof(segment_path_), "%s/%s--%d", root_path, route_name_, part_++);
  
  auto log = std::make_shared<LoggerHandle>();
  if (log->open(segment_path_, log_name_.c_str(), has_qlog_)) {
    auto bytes = init_data_.asBytes();
    log->log(bytes.begin(), bytes.size(), has_qlog_);
    log_sentinel(log, cur_handle_ ? SentinelType::START_OF_SEGMENT : SentinelType::START_OF_ROUTE);

    cur_handle_ = log;
  } else {
    LOGE("logger failed to open files");
    cur_handle_ = nullptr;
  }
  return (bool)cur_handle_;
}

void Logger::close() {
  if (cur_handle_) {
    log_sentinel(cur_handle_, SentinelType::END_OF_ROUTE);
    cur_handle_ = nullptr;
  }
}