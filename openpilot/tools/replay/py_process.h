#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace PyProcess {

// Called with each line of the child's stderr (newline-terminated).
using StderrLineCallback = std::function<void(const char *line)>;

// Run a Python module (`python3 -m <module> <args...>`) and capture stdout.
// Stderr lines are passed to stderr_line_cb if set, otherwise through to the parent's stderr.
// If trim is true (default), trailing newlines are removed from the returned stdout content.
// Returns empty string on failure. If abort is signaled, kills the child process.
std::string runModule(const std::string &module, const std::vector<std::string> &args,
                      std::atomic<bool> *abort = nullptr, bool trim = true,
                      const StderrLineCallback &stderr_line_cb = nullptr);

}  // namespace PyProcess
