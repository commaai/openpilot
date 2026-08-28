#include "tools/cabana/utils/util.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include "common/util.h"

static const std::thread::id main_thread_id = std::this_thread::get_id();
static std::mutex main_thread_queue_mutex;
static std::vector<std::function<void()>> main_thread_queue;

bool utils::isMainThread() { return std::this_thread::get_id() == main_thread_id; }

void utils::runOnMainThread(std::function<void()> fn) {
  if (isMainThread()) {
    fn();
  } else {
    std::lock_guard lk(main_thread_queue_mutex);
    main_thread_queue.push_back(std::move(fn));
  }
}

void utils::drainMainThreadQueue() {
  std::vector<std::function<void()>> fns;
  {
    std::lock_guard lk(main_thread_queue_mutex);
    fns.swap(main_thread_queue);
  }
  for (auto &fn : fns) fn();
}

// SegmentTree

void SegmentTree::build(int n, const std::function<double(int)> &y) {
  size = n;
  tree.resize(4 * size);  // size of the tree is 4 times the size of the array
  if (size > 0) {
    build_tree(y, 1, 0, size - 1);
  }
}

void SegmentTree::build_tree(const std::function<double(int)> &y, int n, int left, int right) {
  if (left == right) {
    tree[n] = {y(left), y(left)};
  } else {
    const int mid = (left + right) >> 1;
    build_tree(y, 2 * n, left, mid);
    build_tree(y, 2 * n + 1, mid + 1, right);
    tree[n] = {std::min(tree[2 * n].first, tree[2 * n + 1].first), std::max(tree[2 * n].second, tree[2 * n + 1].second)};
  }
}

std::pair<double, double> SegmentTree::get_minmax(int n, int left, int right, int range_left, int range_right) const {
  if (range_left > right || range_right < left)
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
  if (range_left <= left && range_right >= right)
    return tree[n];
  int mid = (left + right) >> 1;
  auto l = get_minmax(2 * n, left, mid, range_left, range_right);
  auto r = get_minmax(2 * n + 1, mid + 1, right, range_left, range_right);
  return {std::min(l.first, r.first), std::max(l.second, r.second)};
}

// UnixSignalHandler

UnixSignalHandler::UnixSignalHandler(std::function<void()> on_signal) {
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sig_fd)) {
    fprintf(stderr, "Couldn't create TERM socketpair\n");
    abort();
  }

  waiter = std::thread([this, on_signal = std::move(on_signal)]() {
    int tmp = 0;
    while (::read(sig_fd[1], &tmp, sizeof(tmp)) < 0) {
      if (errno != EINTR) return;
    }
    if (shutting_down.load()) return;

    on_signal();
  });

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, UnixSignalHandler::signalHandler);
}

UnixSignalHandler::~UnixSignalHandler() {
  shutting_down.store(true);
  int dummy = 0;
  (void)!::write(sig_fd[0], &dummy, sizeof(dummy));
  if (waiter.joinable()) waiter.join();
  ::close(sig_fd[0]);
  ::close(sig_fd[1]);
}

void UnixSignalHandler::signalHandler(int s) {
  (void)!::write(sig_fd[0], &s, sizeof(s));
}

// validators

ValidState validateName(std::string &input) {
  std::replace(input.begin(), input.end(), ' ', '_');
  if (input.empty()) return ValidState::Intermediate;
  for (const unsigned char c : input) {
    if (!std::isalnum(c) && c != '_') return ValidState::Invalid;
  }
  return ValidState::Acceptable;
}

ValidState validateNodes(const std::string &input) {
  if (input.empty()) return ValidState::Intermediate;
  // Match ^\w+(,\w+)*$ ; a trailing comma is Intermediate (user still typing).
  bool need_word = true;
  for (const unsigned char c : input) {
    if (std::isalnum(c) || c == '_') {
      need_word = false;
    } else if (c == ',' && !need_word) {
      need_word = true;
    } else {
      return ValidState::Invalid;
    }
  }
  return need_word ? ValidState::Intermediate : ValidState::Acceptable;
}

ValidState validateNonWhitespace(const std::string &input) {
  if (input.empty()) return ValidState::Intermediate;
  for (const unsigned char c : input) {
    if (std::isspace(c)) return ValidState::Invalid;
  }
  return ValidState::Acceptable;
}

ValidState validateIpAddress(const std::string &input) {
  if (input.empty()) return ValidState::Intermediate;

  int dots = 0;
  int value = 0;
  bool has_digit = false;
  for (const unsigned char c : input) {
    if (std::isdigit(c)) {
      value = has_digit ? value * 10 + (c - '0') : (c - '0');
      if (value > 255) return ValidState::Invalid;
      has_digit = true;
    } else if (c == '.') {
      if (!has_digit || dots >= 3) return ValidState::Invalid;
      ++dots;
      has_digit = false;
      value = 0;
    } else {
      return ValidState::Invalid;
    }
  }
  return (dots == 3 && has_digit) ? ValidState::Acceptable : ValidState::Intermediate;
}

ValidState validateDouble(const std::string &input) {
  if (input.empty()) return ValidState::Intermediate;

  // C locale, no hex floats / p-exponents / inf / nan (strtod accepts them, the DBC parser does not)
  if (input.find_first_of("xXpP") != std::string::npos) {
    return ValidState::Invalid;
  }

  const char *start = input.c_str();
  char *end = nullptr;
  const double value = std::strtod(start, &end);
  if (end == start) {
    // Still typing a sign, decimal point, or exponent prefix.
    if (input == "-" || input == "+" || input == "." || input == "-." || input == "+.") {
      return ValidState::Intermediate;
    }
    return ValidState::Invalid;
  }
  if (*end == '\0') {
    return std::isfinite(value) ? ValidState::Acceptable : ValidState::Invalid;
  }

  // Partial exponent / trailing sign while typing (e.g. "1e", "1e-", "1.").
  for (const char *p = end; *p; ++p) {
    const char c = *p;
    if (!(c == 'e' || c == 'E' || c == '+' || c == '-' || c == '.' || (c >= '0' && c <= '9'))) {
      return ValidState::Invalid;
    }
  }
  return ValidState::Intermediate;
}

// embedded at build time from the bootstrap_icons package (see SConscript)
extern const unsigned char bootstrap_icons_svg[];
extern const size_t bootstrap_icons_svg_len;

static std::unordered_map<std::string, std::string> load_bootstrap_icons() {
  std::unordered_map<std::string, std::string> icons;

  const std::string content(reinterpret_cast<const char *>(bootstrap_icons_svg), bootstrap_icons_svg_len);
  const std::string sym_open = "<symbol ";
  const std::string sym_close = "</symbol>";
  const std::string id_attr = "id=\"";

  size_t pos = 0;
  while ((pos = content.find(sym_open, pos)) != std::string::npos) {
    size_t end = content.find(sym_close, pos);
    if (end == std::string::npos) break;
    end += sym_close.size();

    // extract id
    size_t id_start = content.find(id_attr, pos);
    if (id_start != std::string::npos && id_start < end) {
      id_start += id_attr.size();
      size_t id_end = content.find('"', id_start);
      if (id_end != std::string::npos && id_end < end) {
        std::string id = content.substr(id_start, id_end - id_start);
        std::string svg_str = content.substr(pos, end - pos);
        // replace <symbol with <svg, </symbol> with </svg>
        svg_str.replace(0, 7, "<svg");               // "<symbol" (7) -> "<svg" (4)
        svg_str.replace(svg_str.size() - 9, 9, "</svg>");  // "</symbol>" (9) -> "</svg>" (6)
        icons[id] = std::move(svg_str);
      }
    }
    pos = end;
  }
  return icons;
}

namespace utils {

std::string homePath() {
  const char *home = ::getenv("HOME");
  return home ? home : "";
}

std::filesystem::path configPath() {
#ifdef __APPLE__
  return std::filesystem::path(homePath()) / "Library/Preferences";
#else
  const char *xdg = ::getenv("XDG_CONFIG_HOME");
  return (xdg && xdg[0]) ? std::filesystem::path(xdg) : std::filesystem::path(homePath()) / ".config";
#endif
}

#ifdef __APPLE__
static const char *clipboard_read_cmds[] = {"pbpaste"};
static const char *clipboard_write_cmds[] = {"pbcopy"};
#else
static const char *clipboard_read_cmds[] = {"wl-paste --no-newline 2>/dev/null", "xclip -selection clipboard -o 2>/dev/null", "xsel -ob 2>/dev/null"};
static const char *clipboard_write_cmds[] = {"wl-copy 2>/dev/null", "xclip -selection clipboard 2>/dev/null", "xsel -ib 2>/dev/null"};
#endif

bool getClipboardText(std::string *text) {
  text->clear();
  bool has_tool = false;
  for (const char *cmd : clipboard_read_cmds) {
    FILE *f = ::popen(cmd, "r");
    if (!f) continue;
    std::string out;
    char buf[4096];
    for (size_t n; (n = ::fread(buf, 1, sizeof(buf), f)) > 0;) out.append(buf, n);
    int status = ::pclose(f);
    if (status == 0) {
      *text = std::move(out);
      return true;
    }
    has_tool |= WIFEXITED(status) && WEXITSTATUS(status) != 127;  // 127: command not found
  }
  return has_tool;  // tool present but clipboard empty
}

bool setClipboardText(const std::string &text) {
  std::signal(SIGPIPE, SIG_IGN);
  for (const char *cmd : clipboard_write_cmds) {
    FILE *f = ::popen(cmd, "w");
    if (!f) continue;
    size_t written = ::fwrite(text.data(), 1, text.size(), f);
    if (::pclose(f) == 0 && written == text.size()) return true;
  }
  return false;
}

std::string bootstrapSvg(const std::string &id) {
  static auto icons = load_bootstrap_icons();
  auto it = icons.find(id);
  return it != icons.end() ? it->second : std::string();
}

}  // namespace utils

int num_decimals(double num) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%g", num);
  const char *dot = strpbrk(buf, ".,");  // Qt sets LC_ALL from the environment so the decimal mark may be a comma
  return dot ? (int)strlen(dot + 1) : 0;
}

std::filesystem::path executableDir() {
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
