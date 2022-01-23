#pragma once

#include <atomic>
#include <string>

enum class ReplyMsgType {
  Info,
  Debug,
  Warning,
  Critical
};

typedef void (*ReplayMessageHandler)(ReplyMsgType type, const char *msg);
void installMessageHandler(ReplayMessageHandler);
void logMessage(ReplyMsgType type, const char* fmt, ...);

#define rInfo(fmt, ...) logMessage(ReplyMsgType::Info, fmt,  ## __VA_ARGS__)
#define rDebug(fmt, ...) logMessage(ReplyMsgType::Debug, fmt,  ## __VA_ARGS__)
#define rWarning(fmt, ...) logMessage(ReplyMsgType::Warning, fmt,  ## __VA_ARGS__)
#define rError(fmt, ...) logMessage(ReplyMsgType::Critical , fmt,  ## __VA_ARGS__)

std::string sha256(const std::string &str);
void precise_nano_sleep(long sleep_ns);
std::string decompressBZ2(const std::string &in, std::atomic<bool> *abort = nullptr);
std::string decompressBZ2(const std::byte *in, size_t in_size, std::atomic<bool> *abort = nullptr);
void enableHttpLogging(bool enable);
std::string getUrlWithoutQuery(const std::string &url);
size_t getRemoteFileSize(const std::string &url, std::atomic<bool> *abort = nullptr);
std::string httpGet(const std::string &url, size_t chunk_size = 0, std::atomic<bool> *abort = nullptr);
bool httpDownload(const std::string &url, const std::string &file, size_t chunk_size = 0, std::atomic<bool> *abort = nullptr);
