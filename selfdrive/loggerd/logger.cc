#include "selfdrive/loggerd/logger.h"

#include <sys/stat.h>
#include <unistd.h>

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
#ifdef QCOM
#include <cutils/properties.h>
#endif

#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
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

void log_init_data(LoggerState *s) {
  auto bytes = s->init_data.asBytes();
  logger_log(s, bytes.begin(), bytes.size(), s->has_qlog);
}


static void lh_log_sentinel(LoggerHandle *h, SentinelType type) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  sen.setSignal(h->exit_signal);
  auto bytes = msg.toBytes();

  lh_log(h, bytes.begin(), bytes.size(), true);
}

// ***** logging functions *****

void logger_init(LoggerState *s, const char* log_name, bool has_qlog) {
  pthread_mutex_init(&s->lock, NULL);

  s->part = -1;
  s->has_qlog = has_qlog;
  s->route_name = logger_get_route_name();
  snprintf(s->log_name, sizeof(s->log_name), "%s", log_name);
  s->init_data = logger_build_init_data();
}

static LoggerHandle* logger_open(LoggerState *s, const char* root_path) {
  LoggerHandle *h = NULL;
  for (int i=0; i<LOGGER_MAX_HANDLES; i++) {
    if (s->handles[i].refcnt == 0) {
      h = &s->handles[i];
      break;
    }
  }
  assert(h);

  snprintf(h->segment_path, sizeof(h->segment_path),
          "%s/%s--%d", root_path, s->route_name.c_str(), s->part);

  snprintf(h->log_path, sizeof(h->log_path), "%s/%s.bz2", h->segment_path, s->log_name);
  snprintf(h->qlog_path, sizeof(h->qlog_path), "%s/qlog.bz2", h->segment_path);
  h->end_sentinel_type = SentinelType::END_OF_SEGMENT;
  h->exit_signal = 0;

  if (!util::create_directories(h->segment_path, 0775)) return nullptr;

  h->log = std::make_unique<BZFile>(h->log_path);
  if (s->has_qlog) {
    h->q_log = std::make_unique<BZFile>(h->qlog_path);
  }

  pthread_mutex_init(&h->lock, NULL);
  h->refcnt++;
  return h;
}

int logger_next(LoggerState *s, const char* root_path,
                            char* out_segment_path, size_t out_segment_path_len,
                            int* out_part) {
  bool is_start_of_route = !s->cur_handle;

  pthread_mutex_lock(&s->lock);
  s->part++;

  LoggerHandle* next_h = logger_open(s, root_path);
  if (!next_h) {
    pthread_mutex_unlock(&s->lock);
    return -1;
  }

  if (s->cur_handle) {
    lh_close(s->cur_handle);
  }
  s->cur_handle = next_h;

  if (out_segment_path) {
    snprintf(out_segment_path, out_segment_path_len, "%s", next_h->segment_path);
  }
  if (out_part) {
    *out_part = s->part;
  }

  pthread_mutex_unlock(&s->lock);

  // write beggining of log metadata
  log_init_data(s);
  lh_log_sentinel(s->cur_handle, is_start_of_route ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT);
  return 0;
}

LoggerHandle* logger_get_handle(LoggerState *s) {
  pthread_mutex_lock(&s->lock);
  LoggerHandle* h = s->cur_handle;
  if (h) {
    pthread_mutex_lock(&h->lock);
    h->refcnt++;
    pthread_mutex_unlock(&h->lock);
  }
  pthread_mutex_unlock(&s->lock);
  return h;
}

void logger_log(LoggerState *s, uint8_t* data, size_t data_size, bool in_qlog) {
  pthread_mutex_lock(&s->lock);
  if (s->cur_handle) {
    lh_log(s->cur_handle, data, data_size, in_qlog);
  }
  pthread_mutex_unlock(&s->lock);
}

void logger_close(LoggerState *s, ExitHandler *exit_handler) {
  pthread_mutex_lock(&s->lock);
  if (s->cur_handle) {
    s->cur_handle->exit_signal = exit_handler && exit_handler->signal.load();
    s->cur_handle->end_sentinel_type = SentinelType::END_OF_ROUTE;
    lh_close(s->cur_handle);
  }
  pthread_mutex_unlock(&s->lock);
}

void lh_log(LoggerHandle* h, uint8_t* data, size_t data_size, bool in_qlog) {
  pthread_mutex_lock(&h->lock);
  assert(h->refcnt > 0);
  h->log->write(data, data_size);
  if (in_qlog && h->q_log) {
    h->q_log->write(data, data_size);
  }
  pthread_mutex_unlock(&h->lock);
}

void lh_close(LoggerHandle* h) {
  pthread_mutex_lock(&h->lock);
  assert(h->refcnt > 0);
  if (h->refcnt == 1) {
    // a very ugly hack. only here can guarantee sentinel is the last msg
    pthread_mutex_unlock(&h->lock);
    lh_log_sentinel(h, h->end_sentinel_type);
    pthread_mutex_lock(&h->lock);
  }
  h->refcnt--;
  if (h->refcnt == 0) {
    h->log.reset(nullptr);
    h->q_log.reset(nullptr);
    pthread_mutex_unlock(&h->lock);
    pthread_mutex_destroy(&h->lock);
    return;
  }
  pthread_mutex_unlock(&h->lock);
}
