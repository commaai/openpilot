#include "tools/cabana/settings.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

#include <QAbstractButton>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QPushButton>
#include <type_traits>

#include "json11/json11.hpp"
#include "tools/cabana/utils/util.h"

const int MIN_CACHE_MINIUTES = 30;
const int MAX_CACHE_MINIUTES = 120;

Settings settings;

namespace {

std::filesystem::path settingsFile() {
  return utils::configPath() / "cabana.json";
}

struct LoadedSettings {
  json11::Json::object values;
  bool exists = false;
  bool valid = true;
};

class FileLock {
public:
  explicit FileLock(const std::filesystem::path &path) {
    fd = open(path.c_str(), O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0 || flock(fd, LOCK_EX) < 0) {
      fprintf(stderr, "failed to lock Cabana settings %s: %s\n", path.c_str(), strerror(errno));
      if (fd >= 0) close(fd);
      fd = -1;
    }
  }
  ~FileLock() {
    if (fd >= 0) close(fd);
  }
  bool isLocked() const { return fd >= 0; }

private:
  int fd = -1;
};

LoadedSettings loadSettings() {
  std::ifstream input(settingsFile());
  if (!input) return {};

  const std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  std::string error;
  auto settings_json = json11::Json::parse(contents, error);
  if (!error.empty() || !settings_json.is_object()) {
    fprintf(stderr, "failed to read Cabana settings %s%s%s\n", settingsFile().c_str(), error.empty() ? "" : ": ", error.c_str());
    return {.exists = true, .valid = false};
  }
  return {.values = settings_json.object_items(), .exists = true};
}

bool ensureSettingsDirectory() {
  const auto path = settingsFile();
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) {
    fprintf(stderr, "failed to create Cabana settings directory %s: %s\n", path.parent_path().c_str(), error.message().c_str());
    return false;
  }
  return true;
}

bool writeAll(int fd, const std::string &data) {
  size_t written = 0;
  while (written < data.size()) {
    ssize_t result = write(fd, data.data() + written, data.size() - written);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) return false;
    written += result;
  }
  return true;
}

bool saveSettings(const json11::Json::object &settings_json) {
  const auto path = settingsFile();
  const std::string contents = json11::Json(settings_json).dump();
  std::string temporary_path = path.string() + ".tmp.XXXXXX";
  int fd = mkstemp(temporary_path.data());
  if (fd < 0) {
    fprintf(stderr, "failed to create temporary Cabana settings %s: %s\n", temporary_path.c_str(), strerror(errno));
    return false;
  }

  bool success = writeAll(fd, contents) && fsync(fd) == 0;
  if (close(fd) < 0) success = false;
  if (success && rename(temporary_path.c_str(), path.c_str()) < 0) success = false;

  if (success) {
    int dir_fd = open(path.parent_path().c_str(), O_RDONLY | O_CLOEXEC);
    success = dir_fd >= 0 && fsync(dir_fd) == 0;
    if (dir_fd >= 0 && close(dir_fd) < 0) success = false;
  }

  if (!success) {
    const int saved_errno = errno;
    unlink(temporary_path.c_str());
    fprintf(stderr, "failed to save Cabana settings to %s: %s\n", path.c_str(), strerror(saved_errno));
  }
  return success;
}

bool preserveCorruptSettings() {
  const auto path = settingsFile();
  auto backup = path;
  backup += ".corrupt";
  for (int i = 1; std::filesystem::exists(backup); ++i) {
    backup = path;
    backup += ".corrupt." + std::to_string(i);
  }
  if (rename(path.c_str(), backup.c_str()) < 0) {
    fprintf(stderr, "failed to preserve corrupt Cabana settings %s: %s\n", path.c_str(), strerror(errno));
    return false;
  }
  fprintf(stderr, "preserved corrupt Cabana settings at %s\n", backup.c_str());
  return true;
}

// TODO: Remove the legacy QSettings migration after users have had time to migrate to cabana.json.
struct LegacyValue {
  std::vector<std::string> strings;
  std::string bytes;
  bool is_byte_array = false;
};

using LegacySettings = std::map<std::string, LegacyValue>;

int hexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

#ifndef __APPLE__

void appendUtf8(std::string &result, uint32_t codepoint) {
  if (codepoint <= 0x7f) {
    result.push_back(codepoint);
  } else if (codepoint <= 0x7ff) {
    result.push_back(0xc0 | (codepoint >> 6));
    result.push_back(0x80 | (codepoint & 0x3f));
  } else if (codepoint <= 0xffff) {
    result.push_back(0xe0 | (codepoint >> 12));
    result.push_back(0x80 | ((codepoint >> 6) & 0x3f));
    result.push_back(0x80 | (codepoint & 0x3f));
  } else {
    result.push_back(0xf0 | (codepoint >> 18));
    result.push_back(0x80 | ((codepoint >> 12) & 0x3f));
    result.push_back(0x80 | ((codepoint >> 6) & 0x3f));
    result.push_back(0x80 | (codepoint & 0x3f));
  }
}

LegacyValue decodeIniValue(std::string_view encoded) {
  std::vector<std::vector<uint32_t>> decoded(1);
  std::vector<bool> quoted(1, false);
  bool in_quotes = false;

  for (size_t i = 0; i < encoded.size();) {
    char c = encoded[i++];
    if (c == '"') {
      in_quotes = !in_quotes;
      quoted.back() = true;
    } else if (c == ',' && !in_quotes) {
      decoded.emplace_back();
      quoted.push_back(false);
      while (i < encoded.size() && (encoded[i] == ' ' || encoded[i] == '\t')) ++i;
    } else if (c == '\\' && i < encoded.size()) {
      c = encoded[i++];
      static const std::map<char, char> escapes = {
        {'a', '\a'}, {'b', '\b'}, {'f', '\f'}, {'n', '\n'}, {'r', '\r'}, {'t', '\t'},
        {'v', '\v'}, {'"', '"'}, {'?', '?'}, {'\'', '\''}, {'\\', '\\'},
      };
      if (auto it = escapes.find(c); it != escapes.end()) {
        decoded.back().push_back(static_cast<unsigned char>(it->second));
      } else if (c == 'x' && i < encoded.size() && hexDigit(encoded[i]) >= 0) {
        uint32_t value = 0;
        while (i < encoded.size() && hexDigit(encoded[i]) >= 0) value = (value << 4) + hexDigit(encoded[i++]);
        decoded.back().push_back(value & 0xffff);
      } else if (c >= '0' && c <= '7') {
        uint32_t value = c - '0';
        while (i < encoded.size() && encoded[i] >= '0' && encoded[i] <= '7') value = (value << 3) + (encoded[i++] - '0');
        decoded.back().push_back(value & 0xffff);
      }
    } else {
      decoded.back().push_back(static_cast<unsigned char>(c));
    }
  }

  LegacyValue result;
  for (size_t i = 0; i < decoded.size(); ++i) {
    auto &value = decoded[i];
    if (!quoted[i]) {
      while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(value.begin());
      while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
    }

    std::string string_value;
    for (size_t j = 0; j < value.size(); ++j) {
      uint32_t codepoint = value[j];
      if (codepoint >= 0xd800 && codepoint <= 0xdbff && j + 1 < value.size() && value[j + 1] >= 0xdc00 && value[j + 1] <= 0xdfff) {
        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (value[++j] - 0xdc00);
      }
      appendUtf8(string_value, codepoint);
    }
    result.strings.push_back(std::move(string_value));
  }

  if (result.strings.size() == 1 && result.strings[0] == "@Invalid()") {
    result.strings.clear();
  } else if (decoded.size() == 1) {
    static constexpr std::string_view prefix = "@ByteArray(";
    const auto &value = decoded[0];
    if (value.size() >= prefix.size() + 1 && std::equal(prefix.begin(), prefix.end(), value.begin()) && value.back() == ')') {
      result.is_byte_array = true;
      result.bytes.reserve(value.size() - prefix.size() - 1);
      for (size_t i = prefix.size(); i + 1 < value.size(); ++i) result.bytes.push_back(value[i] & 0xff);
    }
  }
  if (!result.is_byte_array) {
    for (auto &value : result.strings) {
      if (value.compare(0, 2, "@@") == 0) value.erase(0, 1);
    }
  }
  return result;
}

LegacySettings loadLegacySettings() {
  auto path = settingsFile();
  path.replace_filename("cabana.conf");
  std::ifstream input(path);
  if (!input) return {};

  LegacySettings settings;
  bool in_general_section = false;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "[General]") {
      in_general_section = true;
      continue;
    }
    if (!line.empty() && line.front() == '[') {
      in_general_section = false;
      continue;
    }
    if (!in_general_section || line.empty() || line.front() == ';') continue;
    if (auto separator = line.find('='); separator != std::string::npos) {
      settings[line.substr(0, separator)] = decodeIniValue(std::string_view(line).substr(separator + 1));
    }
  }
  return settings;
}

#else

std::string cfStringToUtf8(CFStringRef value) {
  CFIndex length = CFStringGetLength(value);
  CFIndex size = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
  std::string result(size, '\0');
  if (!CFStringGetCString(value, result.data(), size, kCFStringEncodingUTF8)) return {};
  result.resize(strlen(result.c_str()));
  return result;
}

LegacyValue cfStringValue(CFStringRef string) {
  LegacyValue value;
  if (CFStringHasPrefix(string, CFSTR("@ByteArray(")) && CFStringHasSuffix(string, CFSTR(")"))) {
    CFRange range{11, CFStringGetLength(string) - 12};
    std::vector<UniChar> data(range.length);
    CFStringGetCharacters(string, range, data.data());
    value.is_byte_array = true;
    value.bytes.reserve(data.size());
    for (UniChar c : data) value.bytes.push_back(c & 0xff);
  } else {
    std::string string_value = cfStringToUtf8(string);
    if (string_value.compare(0, 2, "@@") == 0) string_value.erase(0, 1);
    value.strings.push_back(std::move(string_value));
  }
  return value;
}

LegacySettings loadLegacySettings() {
  LegacySettings settings;
  CFDictionaryRef values = CFPreferencesCopyMultiple(nullptr, CFSTR("com.cabana"),
                                                       kCFPreferencesCurrentUser, kCFPreferencesAnyHost);
  if (values == nullptr) return settings;

  CFIndex count = CFDictionaryGetCount(values);
  std::vector<const void *> keys(count);
  std::vector<const void *> objects(count);
  CFDictionaryGetKeysAndValues(values, keys.data(), objects.data());
  for (CFIndex i = 0; i < count; ++i) {
    if (CFGetTypeID(keys[i]) != CFStringGetTypeID()) continue;
    std::string key = cfStringToUtf8(static_cast<CFStringRef>(keys[i]));
    CFTypeRef object = objects[i];
    LegacyValue value;
    if (CFGetTypeID(object) == CFBooleanGetTypeID()) {
      value.strings.push_back(CFBooleanGetValue(static_cast<CFBooleanRef>(object)) ? "true" : "false");
    } else if (CFGetTypeID(object) == CFNumberGetTypeID()) {
      int number = 0;
      if (CFNumberGetValue(static_cast<CFNumberRef>(object), kCFNumberIntType, &number)) value.strings.push_back(std::to_string(number));
    } else if (CFGetTypeID(object) == CFStringGetTypeID()) {
      value = cfStringValue(static_cast<CFStringRef>(object));
    } else if (CFGetTypeID(object) == CFDataGetTypeID()) {
      auto data = static_cast<CFDataRef>(object);
      value.is_byte_array = true;
      value.bytes.assign(reinterpret_cast<const char *>(CFDataGetBytePtr(data)), CFDataGetLength(data));
    } else if (CFGetTypeID(object) == CFArrayGetTypeID()) {
      auto array = static_cast<CFArrayRef>(object);
      for (CFIndex j = 0; j < CFArrayGetCount(array); ++j) {
        CFTypeRef item = CFArrayGetValueAtIndex(array, j);
        if (CFGetTypeID(item) != CFStringGetTypeID()) {
          value.strings.clear();
          break;
        }
        auto item_value = cfStringValue(static_cast<CFStringRef>(item));
        if (item_value.strings.size() != 1) {
          value.strings.clear();
          break;
        }
        value.strings.push_back(std::move(item_value.strings[0]));
      }
    }
    if (!value.strings.empty() || value.is_byte_array || CFGetTypeID(object) == CFArrayGetTypeID()) {
      settings.emplace(std::move(key), std::move(value));
    }
  }
  CFRelease(values);
  return settings;
}

#endif

template <typename T>
void readLegacySetting(const LegacySettings &legacy_settings, const char *key, T &value) {
  auto it = legacy_settings.find(key);
  if (it == legacy_settings.end() || it->second.strings.size() != 1) return;
  const auto &stored = it->second.strings[0];

  if constexpr (std::is_same_v<T, bool>) {
    if (stored == "true") value = true;
    if (stored == "false") value = false;
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    int number = 0;
    auto [end, error] = std::from_chars(stored.data(), stored.data() + stored.size(), number);
    if (error == std::errc{} && end == stored.data() + stored.size()) value = static_cast<T>(number);
  }
}

void readLegacySetting(const LegacySettings &legacy_settings, const char *key, std::string &value) {
  auto it = legacy_settings.find(key);
  if (it != legacy_settings.end() && it->second.strings.size() == 1) value = it->second.strings[0];
}

void readLegacySetting(const LegacySettings &legacy_settings, const char *key, std::vector<std::string> &value) {
  auto it = legacy_settings.find(key);
  if (it != legacy_settings.end() && !it->second.is_byte_array) value = it->second.strings;
}

void readLegacySetting(const LegacySettings &legacy_settings, const char *key, std::vector<uint8_t> &value) {
  auto it = legacy_settings.find(key);
  if (it != legacy_settings.end() && it->second.is_byte_array) {
    value.assign(it->second.bytes.begin(), it->second.bytes.end());
  }
}

template <typename T>
void readSetting(const json11::Json::object &settings_json, const char *key, T &value) {
  auto it = settings_json.find(key);
  if (it == settings_json.end()) return;

  if constexpr (std::is_same_v<T, bool>) {
    if (it->second.is_bool()) value = it->second.bool_value();
  } else if constexpr (std::is_integral_v<T>) {
    if (it->second.is_number()) value = it->second.int_value();
  } else if constexpr (std::is_enum_v<T>) {
    if (it->second.is_number()) value = static_cast<T>(it->second.int_value());
  }
}

void readSetting(const json11::Json::object &settings_json, const char *key, std::string &value) {
  auto it = settings_json.find(key);
  if (it != settings_json.end() && it->second.is_string()) value = it->second.string_value();
}

void readSetting(const json11::Json::object &settings_json, const char *key, std::vector<std::string> &value) {
  auto it = settings_json.find(key);
  if (it == settings_json.end() || !it->second.is_array()) return;

  std::vector<std::string> stored;
  for (const auto &item : it->second.array_items()) {
    if (!item.is_string()) return;
    stored.push_back(item.string_value());
  }
  value = std::move(stored);
}

void readSetting(const json11::Json::object &settings_json, const char *key, std::vector<uint8_t> &value) {
  auto it = settings_json.find(key);
  if (it == settings_json.end() || !it->second.is_string()) return;

  const auto &hex = it->second.string_value();
  if (hex.size() % 2 == 0 && std::all_of(hex.begin(), hex.end(), [](unsigned char c) { return std::isxdigit(c); })) {
    value.clear();
    value.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
      value.push_back((hexDigit(hex[i]) << 4) | hexDigit(hex[i + 1]));
    }
  }
}

template <typename T>
void writeSetting(json11::Json::object &settings_json, const char *key, const T &value) {
  if constexpr (std::is_same_v<T, bool>) {
    settings_json[key] = value;
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    settings_json[key] = static_cast<int>(value);
  }
}

void writeSetting(json11::Json::object &settings_json, const char *key, const std::string &value) {
  settings_json[key] = value;
}

void writeSetting(json11::Json::object &settings_json, const char *key, const std::vector<std::string> &value) {
  settings_json[key] = value;
}

void writeSetting(json11::Json::object &settings_json, const char *key, const std::vector<uint8_t> &value) {
  static const char digits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(value.size() * 2);
  for (uint8_t b : value) {
    hex.push_back(digits[b >> 4]);
    hex.push_back(digits[b & 0xf]);
  }
  settings_json[key] = hex;
}

template <class Store, class SettingOperation>
void settingsOp(Store &s, SettingOperation op) {
  op(s, "absolute_time", settings.absolute_time);
  op(s, "fps", settings.fps);
  op(s, "max_cached_minutes", settings.max_cached_minutes);
  op(s, "chart_height", settings.chart_height);
  op(s, "chart_range", settings.chart_range);
  op(s, "chart_column_count", settings.chart_column_count);
  op(s, "last_dir", settings.last_dir);
  op(s, "last_route_dir", settings.last_route_dir);
  op(s, "window_state", settings.window_state);
  op(s, "geometry", settings.geometry);
  op(s, "video_splitter_state", settings.video_splitter_state);
  op(s, "recent_files", settings.recent_files);
  op(s, "message_header_state", settings.message_header_state);
  op(s, "chart_series_type", settings.chart_series_type);
  op(s, "theme", settings.theme);
  op(s, "sparkline_range", settings.sparkline_range);
  op(s, "multiple_lines_hex", settings.multiple_lines_hex);
  op(s, "log_livestream", settings.log_livestream);
  op(s, "log_path", settings.log_path);
  op(s, "drag_direction", (int &)settings.drag_direction);
  op(s, "suppress_defined_signals", settings.suppress_defined_signals);
  op(s, "recent_dbc_file", settings.recent_dbc_file);
  op(s, "active_msg_id", settings.active_msg_id);
  op(s, "selected_msg_ids", settings.selected_msg_ids);
  op(s, "active_charts", settings.active_charts);
}

}  // namespace

Settings::Settings() {
  last_dir = last_route_dir = utils::homePath();
  log_path = utils::homePath() + "/cabana_live_stream/";
  const auto stored_settings = loadSettings();
  if (stored_settings.valid) {
    if (stored_settings.exists) {
      settingsOp(stored_settings.values, [](const auto &s, const char *key, auto &value) { readSetting(s, key, value); });
    } else {
      auto legacy_settings = loadLegacySettings();
      settingsOp(legacy_settings, [](const auto &s, const char *key, auto &value) { readLegacySetting(s, key, value); });
    }
  }
  fps = std::clamp(fps, 1, 100);
}

// Must be called before main() returns: json11's internal statistics are constructed on first
// use at runtime, so they are destroyed before this pre-main global. Saving from ~Settings
// would use them after destruction and corrupt the heap.
void Settings::save() {
  if (!ensureSettingsDirectory()) return;

  auto lock_path = settingsFile();
  lock_path += ".lock";
  FileLock lock(lock_path);
  if (!lock.isLocked()) return;

  auto stored_settings = loadSettings();
  if (!stored_settings.valid && !preserveCorruptSettings()) return;
  settingsOp(stored_settings.values, [](auto &s, const char *key, const auto &value) { writeSetting(s, key, value); });
  saveSettings(stored_settings.values);
}

// SettingsDlg

SettingsDlg::SettingsDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Settings"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QGroupBox *groupbox = new QGroupBox("General");
  QFormLayout *form_layout = new QFormLayout(groupbox);

  form_layout->addRow(tr("Color Theme"), theme = new QComboBox(this));
  theme->setToolTip(tr("You may need to restart cabana after changes theme"));
  theme->addItems({tr("Automatic"), tr("Light"), tr("Dark")});
  theme->setCurrentIndex(settings.theme);

  form_layout->addRow("FPS", fps = new QSpinBox(this));
  fps->setRange(10, 100);
  fps->setSingleStep(10);
  fps->setValue(settings.fps);

  form_layout->addRow(tr("Max Cached Minutes"), cached_minutes = new QSpinBox(this));
  cached_minutes->setRange(MIN_CACHE_MINIUTES, MAX_CACHE_MINIUTES);
  cached_minutes->setSingleStep(1);
  cached_minutes->setValue(settings.max_cached_minutes);
  main_layout->addWidget(groupbox);

  groupbox = new QGroupBox("New Signal Settings");
  form_layout = new QFormLayout(groupbox);
  form_layout->addRow(tr("Drag Direction"), drag_direction = new QComboBox(this));
  drag_direction->addItems({tr("MSB First"), tr("LSB First"), tr("Always Little Endian"), tr("Always Big Endian")});
  drag_direction->setCurrentIndex(settings.drag_direction);
  main_layout->addWidget(groupbox);

  groupbox = new QGroupBox("Chart");
  form_layout = new QFormLayout(groupbox);
  form_layout->addRow(tr("Chart Height"), chart_height = new QSpinBox(this));
  chart_height->setRange(100, 500);
  chart_height->setSingleStep(10);
  chart_height->setValue(settings.chart_height);
  main_layout->addWidget(groupbox);

  log_livestream = new QGroupBox(tr("Enable live stream logging"), this);
  log_livestream->setCheckable(true);
  log_livestream->setChecked(settings.log_livestream);
  QHBoxLayout *path_layout = new QHBoxLayout(log_livestream);
  path_layout->addWidget(log_path = new QLineEdit(QString::fromStdString(settings.log_path), this));
  log_path->setReadOnly(true);
  auto browse_btn = new QPushButton(tr("B&rowse..."));
  path_layout->addWidget(browse_btn);
  main_layout->addWidget(log_livestream);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);
  setFixedSize(400, sizeHint().height());

  QObject::connect(browse_btn, &QPushButton::clicked, [this]() {
    QString fn = QFileDialog::getExistingDirectory(
        this, tr("Log File Location"),
        QString::fromStdString(utils::homePath()),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!fn.isEmpty()) {
      log_path->setText(fn);
    }
  });
  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &SettingsDlg::save);
}

void SettingsDlg::save() {
  if (std::exchange(settings.theme, theme->currentIndex()) != settings.theme) {
    // set theme before emit changed
    utils::setTheme(settings.theme);
  }
  settings.fps = fps->value();
  settings.max_cached_minutes = cached_minutes->value();
  settings.chart_height = chart_height->value();
  settings.log_livestream = log_livestream->isChecked();
  settings.log_path = log_path->text().toStdString();
  settings.drag_direction = (Settings::DragDirection)drag_direction->currentIndex();
  emit settings.changed();
  QDialog::accept();
}
