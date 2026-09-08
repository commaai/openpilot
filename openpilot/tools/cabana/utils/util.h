#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <utility>

#include "tools/cabana/core/color.h"

class SegmentTree {
public:
  SegmentTree() = default;
  void build(int n, const std::function<double(int)> &y);
  inline std::pair<double, double> minmax(int left, int right) const { return get_minmax(1, 0, size - 1, left, right); }

private:
  std::pair<double, double> get_minmax(int n, int left, int right, int range_left, int range_right) const;
  void build_tree(const std::function<double(int)> &y, int n, int left, int right);
  std::vector<std::pair<double, double>> tree;
  int size = 0;
};

// maps a linear slider position onto a log10 scale
class LogScale {
public:
  LogScale(double factor) : factor(factor) {}
  void setRange(double min, double max) {
    log_min = factor * std::log10(min);
    log_max = factor * std::log10(max);
  }
  int value(int pos, int pos_min, int pos_max) const {
    double v = log_min + (log_max - log_min) * ((pos - pos_min) / double(pos_max - pos_min));
    return std::lround(std::pow(10, v / factor));
  }
  int position(int v, int pos_min, int pos_max) const {
    double log_v = std::clamp(factor * std::log10(v), log_min, log_max);
    return pos_min + (pos_max - pos_min) * ((log_v - log_min) / (log_max - log_min));
  }

private:
  double factor, log_min = 0, log_max = 1;
};

enum class ValidState { Invalid, Intermediate, Acceptable };

// single identifier: one or more [A-Za-z0-9_], spaces rewritten to '_'
ValidState validateName(std::string &input);
// comma-separated identifiers: \w+(,\w+)*
ValidState validateNodes(const std::string &input);
// one or more non-whitespace characters (\S+)
ValidState validateNonWhitespace(const std::string &input);
// dotted IPv4 address (0-255 per octet)
ValidState validateIpAddress(const std::string &input);
// C-locale floating-point
ValidState validateDouble(const std::string &input);

namespace utils {

bool isMainThread();
// inline on the main thread, queued until drainMainThreadQueue() otherwise
void runOnMainThread(std::function<void()> fn);
void drainMainThreadQueue();
std::string homePath();
std::filesystem::path configPath();
bool getClipboardText(std::string *text);  // false if no clipboard tool is available
bool setClipboardText(const std::string &text);
std::string bootstrapSvg(const std::string &id);  // empty if unknown

// boundary conversions for the remaining Qt byte-array based state APIs
template <typename T>
std::vector<uint8_t> toBytes(const T &dat) { return {dat.begin(), dat.end()}; }

// a callback that is skipped once `alive` expired: the owner resets its token when it goes away
template <typename F>
auto guarded(const std::shared_ptr<bool> &alive, F fn) {
  return [alive = std::weak_ptr<bool>(alive), fn = std::move(fn)](auto &&...args) {
    if (!alive.expired()) fn(std::forward<decltype(args)>(args)...);
  };
}

}

// Watches SIGINT/SIGTERM via a self-pipe and a dedicated waiter thread.
// on_signal runs on the waiter thread; the caller marshals to the GUI thread.
class UnixSignalHandler {
public:
  UnixSignalHandler(std::function<void()> on_signal);
  ~UnixSignalHandler();
  static void signalHandler(int s);

private:
  inline static int sig_fd[2] = {};
  std::atomic<bool> shutting_down{false};
  std::thread waiter;
};

int num_decimals(double num);
std::filesystem::path executableDir();
