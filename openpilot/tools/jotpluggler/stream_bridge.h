#pragma once

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <sys/types.h>
inline constexpr char kStreamMsgqAddress[] = "127.0.0.1";
std::string stream_bridge_whitelist(const std::vector<std::string> &services);
class ScopedMsgqPrefix {
public:
  ScopedMsgqPrefix();
  ~ScopedMsgqPrefix();
  ScopedMsgqPrefix(const ScopedMsgqPrefix &) = delete;
  ScopedMsgqPrefix &operator=(const ScopedMsgqPrefix &) = delete;
  void activate();
  void restore() noexcept;
  const std::string &prefix() const { return prefix_; }
  const std::filesystem::path &path() const { return path_; }
private:
  std::string prefix_;
  std::filesystem::path path_;
  std::optional<std::string> previous_prefix_;
  bool active_ = false;
};

class StreamBridgeProcess {
public:
  explicit StreamBridgeProcess(std::chrono::milliseconds terminate_grace = std::chrono::seconds(3));
  ~StreamBridgeProcess();
  StreamBridgeProcess(const StreamBridgeProcess &) = delete;
  StreamBridgeProcess &operator=(const StreamBridgeProcess &) = delete;
  void start(const std::filesystem::path &executable, const std::vector<std::string> &arguments = {},
             const std::filesystem::path &cleanup_path = {});
  void check_running();
  void stop() noexcept;
  bool owned() const { return pid_ > 0; }
  pid_t pid() const { return pid_; }
private:
  std::chrono::milliseconds terminate_grace_;
  std::string executable_;
  pid_t pid_ = -1;
};
