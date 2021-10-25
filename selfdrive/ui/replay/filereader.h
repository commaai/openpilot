#pragma once

#include <atomic>
#include <string>

class FileReader {
public:
  FileReader(bool cache_to_local, int chunk_size = -1, int max_retries = 3)
      : cache_to_local_(cache_to_local), chunk_size_(chunk_size), max_retries_(max_retries) {}
  virtual ~FileReader() {}
  std::string read(const std::string &file, std::atomic<bool> *abort = nullptr);

private:
  std::string download(const std::string &url, std::atomic<bool> *abort);
  bool cache_to_local_;
  int chunk_size_;
  int max_retries_;
};

std::string cacheFilePath(const std::string &url);
