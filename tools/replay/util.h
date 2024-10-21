#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include "cereal/messaging/messaging.h"

enum CameraType {
  RoadCam = 0,
  DriverCam,
  WideRoadCam
};

enum class ReplyMsgType {
  Info,
  Debug,
  Warning,
  Critical
};

typedef std::function<void(ReplyMsgType type, const std::string msg)> ReplayMessageHandler;
void installMessageHandler(ReplayMessageHandler);
void logMessage(ReplyMsgType type, const char* fmt, ...);

#define rInfo(fmt, ...) ::logMessage(ReplyMsgType::Info, fmt,  ## __VA_ARGS__)
#define rDebug(fmt, ...) ::logMessage(ReplyMsgType::Debug, fmt,  ## __VA_ARGS__)
#define rWarning(fmt, ...) ::logMessage(ReplyMsgType::Warning, fmt,  ## __VA_ARGS__)
#define rError(fmt, ...) ::logMessage(ReplyMsgType::Critical , fmt,  ## __VA_ARGS__)

class MonotonicBuffer {
public:
  MonotonicBuffer(size_t initial_size) : next_buffer_size(initial_size) {}
  ~MonotonicBuffer();
  void *allocate(size_t bytes, size_t alignment = 16ul);
  void deallocate(void *p) {}

private:
  void *current_buf = nullptr;
  size_t next_buffer_size = 0;
  size_t available = 0;
  std::deque<void *> buffers;
  static constexpr float growth_factor = 1.5;
};

std::string sha256(const std::string &str);
void precise_nano_sleep(int64_t nanoseconds, std::atomic<bool> &should_exit);
std::string decompressBZ2(const std::string &in, std::atomic<bool> *abort = nullptr);
std::string decompressBZ2(const std::byte *in, size_t in_size, std::atomic<bool> *abort = nullptr);
std::string decompressZST(const std::string &in, std::atomic<bool> *abort = nullptr);
std::string decompressZST(const std::byte *in, size_t in_size, std::atomic<bool> *abort = nullptr);
std::string getUrlWithoutQuery(const std::string &url);
size_t getRemoteFileSize(const std::string &url, std::atomic<bool> *abort = nullptr);
std::string httpGet(const std::string &url, size_t chunk_size = 0, std::atomic<bool> *abort = nullptr);

typedef std::function<void(uint64_t cur, uint64_t total, bool success)> DownloadProgressHandler;
void installDownloadProgressHandler(DownloadProgressHandler);
bool httpDownload(const std::string &url, const std::string &file, size_t chunk_size = 0, std::atomic<bool> *abort = nullptr);
std::string formattedDataSize(size_t size);
std::vector<std::string> split(std::string_view source, char delimiter);
std::string join(const std::vector<std::string> &elements, char separator);
std::string extractFileName(const std::string& file);
