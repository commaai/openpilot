#pragma once

#include <atomic>
#include <string>

class FileReader {
 public:
  FileReader(bool cache_to_local, int max_retries = 3);
  virtual ~FileReader();
  std::string read(const std::string &file);
  bool isAborting() { return abort_; }
  void abort() { abort_ = true; }

 private:
  std::string download(const std::string &url);
  int max_retries_;
  std::atomic<bool> abort_ = false;
  bool cache_to_local_;
};
