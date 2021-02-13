#include <assert.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef QCOM
#include <cutils/properties.h>
#endif

#include "common/swaglog.h"
#include "common/params.h"
#include "common/version.h"
#include "logger.h"

// ***** logging helpers *****

void append_property(const char* key, const char* value, void *cookie) {
  std::vector<std::pair<std::string, std::string> > *properties =
    (std::vector<std::pair<std::string, std::string> > *)cookie;

  properties->push_back(std::make_pair(std::string(key), std::string(value)));
}

int logger_mkpath(char* file_path) {
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

// ***** log metadata *****
kj::Array<capnp::word> logger_build_init_data() {
  MessageBuilder msg;
  auto init = msg.initEvent().initInitData();

  if (util::file_exists("/EON")) {
    init.setDeviceType(cereal::InitData::DeviceType::NEO);
  } else if (util::file_exists("/TICI")) {
    init.setDeviceType(cereal::InitData::DeviceType::TICI);
  } else {
    init.setDeviceType(cereal::InitData::DeviceType::PC);
  }

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
  init.setDirty(!getenv("CLEAN"));

  // log params
  Params params = Params();
  init.setGitCommit(params.get("GitCommit"));
  init.setGitBranch(params.get("GitBranch"));
  init.setGitRemote(params.get("GitRemote"));
  init.setPassive(params.getBool("Passive"));
  {
    std::map<std::string, std::string> params_map;
    params.read_db_all(&params_map);
    auto lparams = init.initParams().initEntries(params_map.size());
    int i = 0;
    for (auto& kv : params_map) {
      auto lentry = lparams[i];
      lentry.setKey(kv.first);
      lentry.setValue(capnp::Data::Reader((const kj::byte*)kv.second.data(), kv.second.size()));
      i++;
    }
  }
  return capnp::messageToFlatArray(msg);
}

std::string logger_get_route_name() {
  char route_name[64] = {'\0'};
  time_t rawtime = time(NULL);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);
  strftime(route_name, sizeof(route_name), "%Y-%m-%d--%H-%M-%S", &timeinfo);
  return route_name;
}

static void log_sentinel(LoggerHandle *h, SentinelType type) {
  MessageBuilder msg;
  msg.initEvent().initSentinel().setType(type);
  h->write(msg.toBytes(), true);
}

// LoggerState

LoggerState::LoggerState(const std::string& log_root) {
  umask(0);
  init_data = logger_build_init_data();
  route_path = log_root + "/" + logger_get_route_name();
}

std::shared_ptr<LoggerHandle> LoggerState::next() {
  SentinelType sentinel_type = part >= 0 ? SentinelType::START_OF_SEGMENT : SentinelType::START_OF_ROUTE;
  cur_handle = std::make_shared<LoggerHandle>(route_path, ++part, sentinel_type, init_data.asBytes());
  return cur_handle;
}

LoggerState::~LoggerState() {
  if (cur_handle) cur_handle->end_of_route();
}

// LoggerHandle

LoggerHandle::LoggerHandle(const std::string& route_path, int part, SentinelType type, kj::ArrayPtr<kj::byte> init_data) : part(part) {
  segment_path = util::string_format("%s--%d", route_path.c_str(), part);
  const std::string log_path = segment_path + "/rlog.bz2";
  const std::string qlog_path = segment_path + "/qlog.bz2";

  // mkpath & create lock file.
  lock_path = log_path + ".lock";
  int err = logger_mkpath((char*)log_path.c_str());
  assert(err == 0);
  int flock = open(lock_path.c_str(), O_RDWR | O_CREAT);
  assert(flock != -1);
  close(flock);

  log = std::make_unique<BZFile>(log_path);
  qlog = std::make_unique<BZFile>(qlog_path);

  // log init data & sentinel type.
  write(init_data, true);
  log_sentinel(this, type);
}

void LoggerHandle::write(uint8_t* data, size_t data_size, bool in_qlog) {
  std::lock_guard lk(lock);
  log->write(data, data_size);
  if (in_qlog) qlog->write(data, data_size);
}

LoggerHandle::~LoggerHandle() {
  log_sentinel(this, end_sentinel_type);
  unlink(lock_path.c_str());
}
