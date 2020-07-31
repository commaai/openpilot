#include "logger.h"
#include <sys/stat.h>
#include <fstream>
#include <streambuf>
#include <vector>
#include <string>
#include "cereal/gen/cpp/log.capnp.h"
#include "common/version.h"
#include "common/util.h"

#include "common/params.h"
typedef cereal::Sentinel::SentinelType SentinelType;
static void log_sentinel(LoggerHandle* log, SentinelType type) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  log->write(msg, true);
}

static int mkpath(const char* file_path) {
  assert(file_path && *file_path);
  char* p;
  for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
    *p = '\0';
    if (mkdir(file_path, 0777)==-1) {
      if (errno != EEXIST) {
        *p = '/';
        return -1;
      }
    }
    *p = '/';
  }
  return 0;
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
  for (int i=0; i<kernel_args.size(); i++) {
    lkernel_args.set(i, kernel_args[i]);
  }

  init.setKernelVersion(util::read_file("/proc/version"));

#ifdef QCOM
  {
    std::vector<std::pair<std::string, std::string> > properties;
    property_list(append_property, (void*)&properties);

    auto lentries = init.initAndroidProperties().initEntries(properties.size());
    for (int i=0; i<properties.size(); i++) {
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
  {
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
  }
  return capnp::messageToFlatArray(msg);
}

Logger::Logger(const char* log_name, bool has_qlog) : part(-1), has_qlog(has_qlog) {
  init_data = gen_init_data();

  umask(0);

  time_t rawtime = time(NULL);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);
  strftime(route_name, sizeof(route_name), "%Y-%m-%d--%H-%M-%S", &timeinfo);
  log_name_ = log_name;
}

std::shared_ptr<LoggerHandle> Logger::openNext(const char* root_path) {
  if (cur_handle) log_sentinel(cur_handle.get(), SentinelType::END_OF_SEGMENT);

  part += 1;
  segment_path = util::string_format("%s/%s--%d", root_path, route_name, part);
  auto log = std::make_shared<LoggerHandle>();
  if (!log->open(segment_path.c_str(), log_name_.c_str(), part, has_qlog)) {
    return nullptr;
  }
  auto bytes = init_data.asBytes();
  log->write(bytes.begin(), bytes.size(), has_qlog);
  log_sentinel(log.get(), !cur_handle ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT);
  return cur_handle = log;
}

Logger::~Logger() {
  if (cur_handle) log_sentinel(cur_handle.get(), SentinelType::END_OF_ROUTE);
}

bool LoggerHandle::open(const char* segment_path, const char* log_name, int part, bool has_qlog) {
  std::string log_path = util::string_format("%s/%s.bz2", segment_path, log_name);
  std::string qlog_path = util::string_format("%s/qlog.bz2", segment_path);
  lock_path = util::string_format("%s.lock", log_path.c_str());

  int err = mkpath(log_path.c_str());
  if (err) return false;

  FILE* lock_file = fopen(lock_path.c_str(), "wb");
  if (lock_file == NULL) return false;
  fclose(lock_file);

  auto open_files = [](std::string& f_path, FILE*& f, BZFILE*& bz_f) {
    f = fopen(f_path.c_str(), "wb");
    if (f != nullptr) {
      int err;
      bz_f = BZ2_bzWriteOpen(&err, f, 9, 0, 30);
      return err == BZ_OK;
    }
    return false;
  };

  if (!open_files(log_path, log_file, bz_file) || (has_qlog && !open_files(qlog_path, qlog_file, bz_qlog))) {
    close();
    return false;
  }
  return true;
}

void LoggerHandle::write(uint8_t* data, size_t data_size, bool in_qlog) {
  const std::lock_guard<std::mutex> lock(mutex);
  int bzerror;
  BZ2_bzWrite(&bzerror, bz_file, data, data_size);
  if (in_qlog && bz_qlog != NULL) BZ2_bzWrite(&bzerror, bz_qlog, data, data_size);
}

void LoggerHandle::write(capnp::MessageBuilder& msg, bool in_qlog) {
  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  write(bytes.begin(), bytes.size(), in_qlog);
}

void LoggerHandle::close() {
  const std::lock_guard<std::mutex> lock(mutex);
  auto close_files = [](FILE*& f, BZFILE*& bz_f) {
    int bzerror;
    if (bz_f) BZ2_bzWriteClose(&bzerror, bz_f, 0, NULL, NULL);
    if (f) fclose(f);

    f = nullptr;
    bz_f = nullptr;
  };

  close_files(qlog_file, bz_qlog);
  close_files(log_file, bz_file);
  unlink(lock_path.c_str());
}
