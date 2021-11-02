#pragma once

#include <atomic>
#include <ostream>
#include <string>

enum REPLAY_FLAGS {
  REPLAY_FLAG_NONE = 0x0000,
  REPLAY_FLAG_DCAM = 0x0002,
  REPLAY_FLAG_ECAM = 0x0004,
  REPLAY_FLAG_NO_LOOP = 0x0010,
  REPLAY_FLAG_NO_FILE_CACHE = 0x0020,
  REPLAY_FLAG_QCAMERA = 0x0040,
  REPLAY_FLAG_SEND_YUV = 0x0080,
};

std::string sha256(const std::string &str);
void precise_nano_sleep(long sleep_ns);
std::string decompressBZ2(const std::string &in);
void enableHttpLogging(bool enable);
std::string getUrlWithoutQuery(const std::string &url);
size_t getRemoteFileSize(const std::string &url);
bool httpMultiPartDownload(const std::string &url, std::ostream &os, int parts, size_t content_length, std::atomic<bool> *abort = nullptr);
