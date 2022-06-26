#include "selfdrive/loggerd/logger.h"

#include <sys/stat.h>
#include <unistd.h>
#include <ftw.h>

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "common/params.h"
#include "common/swaglog.h"
#include "common/version.h"

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

  init.setDeviceType(Hardware::get_device_type());
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

static void log_sentinel(LoggerState *h, SentinelType type, int signal = 0) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  sen.setSignal(signal);
  auto bytes = msg.toBytes();
  h->write(bytes.begin(), bytes.size(), true);
}

LoggerState::LoggerState(const std::string& segment_path) {
  bool ret = util::create_directories(segment_path, 0775);
  assert(ret == true);

  std::string log_path = segment_path + "/rlog";
  log = std::make_unique<RawFile>(log_path);
  qlog = std::make_unique<RawFile>(segment_path + "/qlog");

  lock_file = log_path + ".lock";
  std::ofstream{lock_file};
}

void LoggerState::write(uint8_t* data, size_t data_size, bool in_qlog) {
  log->write(data, data_size);
  if (in_qlog) qlog->write(data, data_size);
}

LoggerState::~LoggerState() {
  std::remove(lock_file.c_str());
}

Logger::Logger(const std::string &log_root) {
  route_name = logger_get_route_name();
  route_path = log_root + "/" + route_name;
  init_data = logger_build_init_data();
}

bool Logger::next() {
  if (logger) {
    log_sentinel(logger.get(), cereal::Sentinel::SentinelType::END_OF_SEGMENT);
  }

  segment_path = route_path + "--" + std::to_string(++part);
  logger.reset(new LoggerState(segment_path));
  // log init data & sentinel type.
  auto bytes = init_data.asBytes();
  logger->write(bytes.begin(), bytes.size(), true);
  log_sentinel(logger.get(), part > 0 ? SentinelType::START_OF_SEGMENT : SentinelType::START_OF_ROUTE);
  return true;
}

void Logger::close(int signal) {
  log_sentinel(logger.get(), cereal::Sentinel::SentinelType::END_OF_ROUTE, signal);
  logger.reset(nullptr);
}
