#include "logger.h"
#include <sys/stat.h>
#include <fstream>
#include <streambuf>
#include "common/version.h"
#ifdef QCOM
#include <cutils/properties.h>
#endif

void append_property(const char* key, const char* value, void *cookie);
kj::Array<capnp::word> gen_init_data();

typedef cereal::Sentinel::SentinelType SentinelType;

static void log_sentinel(LoggerHandle* log, SentinelType type) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  log->write(msg, true);
}

static int mkpath(char* file_path) {
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

Logger::Logger(const char* root_path, const char* log_name, bool has_qlog)
    : root_path(root_path), log_name(log_name), has_qlog(has_qlog), part(-1) {

  umask(0);

  init_data = gen_init_data();

  time_t rawtime = time(NULL);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);
  strftime(route_name, sizeof(route_name), "%Y-%m-%d--%H-%M-%S", &timeinfo);
}

std::shared_ptr<LoggerHandle> Logger::openNext() {
  if (cur_handle) log_sentinel(cur_handle.get(), SentinelType::END_OF_SEGMENT);

  part += 1;
  segment_path = util::string_format("%s/%s--%d", root_path.c_str(), route_name, part);
  auto log = std::make_shared<LoggerHandle>();
  if (!log->open(segment_path, log_name, has_qlog)) {
    return nullptr;
  }
  auto bytes = init_data.asBytes();
  log->write(bytes.begin(), bytes.size(), has_qlog);
  log_sentinel(log.get(), !cur_handle ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT);
  cur_handle = log;
  return cur_handle;
}

bool LoggerHandle::open(const std::string& segment_path, const std::string& log_name, bool has_qlog) {
  std::string log_path = util::string_format("%s/%s.bz2", segment_path.c_str(), log_name.c_str());
  std::string qlog_path = segment_path + "/qlog.bz2";
  lock_path = log_path + ".lock";

  if (0 != mkpath((char*)log_path.c_str())) return false;

  FILE* lock_file = fopen(lock_path.c_str(), "wb");
  if (lock_file == NULL) return false;
  fclose(lock_file);

  auto open_files = [](const std::string& f_path, FILE*& f, BZFILE*& bz_f) {
    f = fopen(f_path.c_str(), "wb");
    if (f != nullptr) {
      int bzerror;
      bz_f = BZ2_bzWriteOpen(&bzerror, f, 9, 0, 30);
      return bzerror == BZ_OK;
    }
    return false;
  };

  if (!open_files(log_path, log_file, bz_file) ||
      (has_qlog && !open_files(qlog_path, qlog_file, bz_qlog))) {
    close();
    return false;
  }
  return true;
}

void LoggerHandle::write(uint8_t* data, size_t data_size, bool in_qlog) {
  const std::lock_guard<std::mutex> lock(mutex);
  int bzerror;
  BZ2_bzWrite(&bzerror, bz_file, data, data_size);
  if (in_qlog && bz_qlog != NULL) {
    BZ2_bzWrite(&bzerror, bz_qlog, data, data_size);
  }
}

void LoggerHandle::write(MessageBuilder& msg, bool in_qlog) {
  auto bytes = msg.toBytes();
  write(bytes.begin(), bytes.size(), in_qlog);
}

void LoggerHandle::close() {
  auto close_files = [](FILE*& f, BZFILE*& bz_f) {
    int bzerror;
    if (bz_f) {
      BZ2_bzWriteClose(&bzerror, bz_f, 0, NULL, NULL);
      bz_f = nullptr;
    }
    if (f) {
      fclose(f);
      f = nullptr;
    }
  };

  close_files(qlog_file, bz_qlog);
  close_files(log_file, bz_file);
  unlink(lock_path.c_str());
}
