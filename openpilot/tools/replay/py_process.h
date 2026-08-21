#pragma once

#include <atomic>
#include <string>
#include <vector>

namespace PyProcess {

// Run a Python command and capture stdout. Stderr is left attached to the parent.
// Returns stdout content. If abort is signaled, kills the child process.
std::string runModule(const std::string &module, const std::vector<std::string> &args,
                      std::atomic<bool> *abort = nullptr, bool trim = true);

}  // namespace PyProcess
