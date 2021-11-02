#include "selfdrive/loggerd/logger.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#ifdef QCOM
#include <cutils/properties.h>
#endif

#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"
#include "selfdrive/common/version.h"

// ***** logging helpers *****

void append_property(const char* key, const char* value, void *cookie) {
  std::vector<std::pair<std::string, std::string> > *properties =
    (std::vector<std::pair<std::string, std::string> > *)cookie;

  properties->push_back(std::make_pair(std::string(key), std::string(value)));
}

// ***** log metadata *****
kj::Array<capnp::word> logger_build_init_data() {
  MessageBuilder msg;
  auto init = msg.initEvent().initInitData();

  if (Hardware::EON()) {
    init.setDeviceType(cereal::InitData::DeviceType::NEO);
  } else if (Hardware::TICI()) {
    init.setDeviceType(cereal::InitData::DeviceType::TICI);
  } else {
    init.setDeviceType(cereal::InitData::DeviceType::PC);
  }

  init.setVersion(COMMA_VERSION);

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
  init.setOsVersion(util::read_file("/VERSION"));

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

  init.setDirty(!getenv("CLEAN"));

  // log params
  auto params = Params();
  std::map<std::string, std::string> params_map = params.readAll();

  init.setGitCommit(params_map["GitCommit"]);
  init.setGitBranch(params_map["GitBranch"]);
  init.setGitRemote(params_map["GitRemote"]);
  init.setPassive(params.getBool("Passive"));
  init.setDongleId(params_map["DongleId"]);

  auto lparams = init.initParams().initEntries(params_map.size());
  int i = 0;
  for (auto& [key, value] : params_map) {
    auto lentry = lparams[i];
    lentry.setKey(key);
    if ( !(params.getKeyType(key) & DONT_LOG) ) {
      lentry.setValue(capnp::Data::Reader((const kj::byte*)value.data(), value.size()));
    }
    i++;

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

static void log_sentinel(Logger *h, SentinelType type, int signal = 0) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  sen.setSignal(signal);
  h->write(msg.toBytes(), true);
}

// Logger

Logger::Logger(const std::string& route_path, int part, kj::ArrayPtr<kj::byte> init_data) : part_(part) {
  segment_path_ = route_path + "--" + std::to_string(part);
  const std::string log_path = segment_path_ + "/rlog.bz2";
  const std::string qlog_path = segment_path_ + "/qlog.bz2";

  // mkpath & create lock file.
  bool ret = util::create_directories(log_path, 0775);
  assert(ret == true);
  lock_path_ = segment_path_ + "/log.lock";
  std::ofstream{lock_path_};

  log_ = std::make_unique<BZFile>(log_path.c_str());
  qlog_ = std::make_unique<BZFile>(qlog_path.c_str());

  // log init data & sentinel type.
  write(init_data, true);
  log_sentinel(this, part_ > 0 ? SentinelType::START_OF_SEGMENT : SentinelType::START_OF_ROUTE);
}

void Logger::write(uint8_t* data, size_t data_size, bool in_qlog) {
  std::lock_guard lk(lock_);
  log_->write(data, data_size);
  if (in_qlog) qlog_->write(data, data_size);
}

void Logger::end_of_route(int signal) {
  end_sentinel_type_ = SentinelType::END_OF_ROUTE;
  signal_ = signal;
}

Logger::~Logger() {
  log_sentinel(this, end_sentinel_type_, signal_);
  ::unlink(lock_path_.c_str());
}

// LoggerManager

LoggerManager::LoggerManager(const std::string& log_root) {
  route_name_ = logger_get_route_name();
  route_path_ = log_root + "/" + route_name_;
  init_data_ = logger_build_init_data();
}

std::shared_ptr<Logger> LoggerManager::next() {
  return std::make_shared<Logger>(route_path_, ++part_, init_data_.asBytes());
}
