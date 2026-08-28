#pragma once

#include <filesystem>
#include <string>

namespace utils {

std::string homePath();
std::filesystem::path configPath();
std::filesystem::path executableDir();

}  // namespace utils
