#include "logger.h"

#include <sys/stat.h>

#include <fstream>
#include <streambuf>

#include "common/version.h"
#ifdef QCOM
#include <cutils/properties.h>
#endif
#define COMPRESS_LEVEL 5

void append_property(const char* key, const char* value, void* cookie);
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
  auto log = std::make_shared<LoggerHandle>(has_qlog);
  if (!log->open(segment_path, log_name)) {
    return nullptr;
  }
  auto bytes = init_data.asBytes();
  log->write(bytes.begin(), bytes.size(), has_qlog);
  log_sentinel(log.get(), !cur_handle ? SentinelType::START_OF_ROUTE : SentinelType::START_OF_SEGMENT);
  cur_handle = log;
  return cur_handle;
}

LoggerHandle::LoggerHandle(bool has_qlog) {
  log = std::make_unique<ZSTDWriter>(COMPRESS_LEVEL);
  if (has_qlog) {
    qlog = std::make_unique<ZSTDWriter>(COMPRESS_LEVEL);
  }
}

bool LoggerHandle::open(const std::string& segment_path, const std::string& log_name) {
  std::string log_path = util::string_format("%s/%s.zst", segment_path.c_str(), log_name.c_str());
  std::string qlog_path = segment_path + "/qlog.zst";
  lock_path = log_path + ".lock";

  if (0 != mkpath((char*)log_path.c_str())) return false;

  FILE* lock_file = fopen(lock_path.c_str(), "wb");
  if (lock_file == NULL) return false;
  fclose(lock_file);

  if (!log->open(log_path.c_str()) || (qlog && !qlog->open(qlog_path.c_str()))) {
    close();
    return false;
  }
  return true;
}

void LoggerHandle::write(uint8_t* data, size_t data_size, bool in_qlog) {
  const std::lock_guard<std::mutex> lock(mutex);
  log->write(data, data_size);
  if (in_qlog) {
    qlog->write(data, data_size);
  }
}

void LoggerHandle::write(MessageBuilder& msg, bool in_qlog) {
  auto bytes = msg.toBytes();
  write(bytes.begin(), bytes.size(), in_qlog);
}

void LoggerHandle::close() {
  log->close();
  if (qlog) {
    qlog->close();
  }
  unlink(lock_path.c_str());
}

ZSTDWriter::ZSTDWriter(int cLevel) {
  cctx = ZSTD_createCCtx();
  assert(cctx != nullptr);
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, cLevel);
  buf_out.resize(ZSTD_CStreamOutSize());
  output = {.dst = buf_out.data(),
            .size = buf_out.size(),
            .pos = 0};
}

ZSTDWriter::~ZSTDWriter() {
  close();
  ZSTD_freeCCtx(cctx);
}

bool ZSTDWriter::open(const char* filename) {
  assert(file == nullptr);
  assert(ZSTD_CCtx_reset(cctx, ZSTD_reset_session_only));
  file = fopen(filename, "wb");
  return file != nullptr;
}

void ZSTDWriter::close() {
  if (file != nullptr) {
    write(nullptr, 0); // flush
    fclose(file);
  }
}

size_t ZSTDWriter::write(void* data, size_t len) {
  assert(data != nullptr || len == 0);
  bool lastChunk = data == nullptr;
  ZSTD_EndDirective const mode = lastChunk ? ZSTD_e_end : ZSTD_e_continue;
  ZSTD_inBuffer input = {data, len, 0};
  bool finished;
  do {
    size_t const remaining = ZSTD_compressStream2(cctx, &output, &input, mode);
    if (lastChunk || remaining > output.size - output.pos) {
      // lastChunk or buffer isn't big enough to store compressed content
      fwrite(buf_out.data(), 1, output.pos, file);
      output.pos = 0;
    }
    finished = lastChunk ? (remaining == 0) : (input.pos == input.size);
  } while (!finished);
  return len;
}
