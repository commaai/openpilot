#include "tools/cabana/utils/paths.h"

#include <cstdlib>
#ifdef __APPLE__
#include <climits>
#include <mach-o/dyld.h>
#endif

#include "common/util.h"

std::string utils::homePath() {
  const char *home = ::getenv("HOME");
  return home ? home : "";
}

std::filesystem::path utils::configPath() {
#ifdef __APPLE__
  return std::filesystem::path(homePath()) / "Library/Preferences";
#else
  const char *xdg = ::getenv("XDG_CONFIG_HOME");
  return (xdg && xdg[0]) ? std::filesystem::path(xdg) : std::filesystem::path(homePath()) / ".config";
#endif
}

std::filesystem::path utils::executableDir() {
#ifdef __APPLE__
  char buf[PATH_MAX];
  uint32_t size = sizeof(buf);
  if (_NSGetExecutablePath(buf, &size) != 0) return {};
  std::error_code ec;
  auto path = std::filesystem::canonical(buf, ec);
  return (ec ? std::filesystem::path(buf) : path).parent_path();
#else
  return std::filesystem::path(util::readlink("/proc/self/exe")).parent_path();
#endif
}
