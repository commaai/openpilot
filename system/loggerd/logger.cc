#include "system/loggerd/logger.h"

#include <fstream>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <random>

#include "common/params.h"
#include "common/swaglog.h"
#include "common/version.h"

// ***** log metadata *****
kj::Array<capnp::word> logger_build_init_data() {
  uint64_t wall_time = nanos_since_epoch();

  MessageBuilder msg;
  auto init = msg.initEvent().initInitData();

  init.setWallTimeNanos(wall_time);
  init.setVersion(COMMA_VERSION);
  init.setDirty(!getenv("CLEAN"));
  init.setDeviceType(Hardware::get_device_type());

  // log kernel args
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

  // log params
  auto params = Params(util::getenv("PARAMS_COPY_PATH", ""));
  std::map<std::string, std::string> params_map = params.readAll();

  init.setGitCommit(params_map["GitCommit"]);
  init.setGitCommitDate(params_map["GitCommitDate"]);
  init.setGitBranch(params_map["GitBranch"]);
  init.setGitRemote(params_map["GitRemote"]);
  init.setPassive(false);
  init.setDongleId(params_map["DongleId"]);

  auto lparams = init.initParams().initEntries(params_map.size());
  int j = 0;
  for (auto& [key, value] : params_map) {
    auto lentry = lparams[j];
    lentry.setKey(key);
    if ( !(params.getKeyType(key) & DONT_LOG) ) {
      lentry.setValue(capnp::Data::Reader((const kj::byte*)value.data(), value.size()));
    }
    j++;
  }

  // log commands
  std::vector<std::string> log_commands = {
    "df -h",  // usage for all filesystems
  };

  auto hw_logs = Hardware::get_init_logs();

  auto commands = init.initCommands().initEntries(log_commands.size() + hw_logs.size());
  for (int i = 0; i < log_commands.size(); i++) {
    auto lentry = commands[i];

    lentry.setKey(log_commands[i]);

    const std::string result = util::check_output(log_commands[i]);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)result.data(), result.size()));
  }

  int i = log_commands.size();
  for (auto &[key, value] : hw_logs) {
    auto lentry = commands[i];
    lentry.setKey(key);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)value.data(), value.size()));
    i++;
  }

  return capnp::messageToFlatArray(msg);
}

std::string logger_get_identifier(std::string key) {
  // a log identifier is a 32 bit counter, plus a 10 character unique ID.
  // e.g. 000001a3--c20ba54385

  Params params;
  uint32_t cnt;
  try {
    cnt = std::stol(params.get(key));
  } catch (std::exception &e) {
    cnt = 0;
  }
  params.put(key, std::to_string(cnt + 1));

  std::stringstream ss;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist(0, 15);
  for (int i = 0; i < 10; ++i) {
    ss << std::hex << dist(mt);
  }

  return util::string_format("%08x--%s", cnt, ss.str().c_str());
}

static void log_sentinel(LoggerState *log, SentinelType type, int eixt_signal = 0) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  sen.setSignal(eixt_signal);
  log->write(msg.toBytes(), true);
}

LoggerState::LoggerState(const std::string &log_root) {
  route_name = logger_get_identifier("RouteCount");
  route_path = log_root + "/" + route_name;
  init_data = logger_build_init_data();
}

LoggerState::~LoggerState() {
  if (rlog) {
    log_sentinel(this, SentinelType::END_OF_ROUTE, exit_signal);
    std::remove(lock_file.c_str());
  }
}

bool LoggerState::next() {
  if (rlog) {
    log_sentinel(this, SentinelType::END_OF_SEGMENT);
    std::remove(lock_file.c_str());
  }

  segment_path = route_path + "--" + std::to_string(++part);
  bool ret = util::create_directories(segment_path, 0775);
  assert(ret == true);

  const std::string rlog_path = segment_path + "/rlog";
  lock_file = rlog_path + ".lock";
  std::ofstream{lock_file};

  rlog.reset(new RawFile(rlog_path));
  qlog.reset(new RawFile(segment_path + "/qlog"));

  // log init data & sentinel type.
  write(init_data.asBytes(), true);
  log_sentinel(this, part > 0 ? SentinelType::START_OF_SEGMENT : SentinelType::START_OF_ROUTE);
  return true;
}

void LoggerState::write(uint8_t* data, size_t size, bool in_qlog) {
  rlog->write(data, size);
  if (in_qlog) qlog->write(data, size);
}
