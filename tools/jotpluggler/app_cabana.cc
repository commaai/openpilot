#include "tools/jotpluggler/app_internal.h"

#include "implot.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr float kSplitterThickness = 4.0f;
constexpr float kMinMessagesWidth = 210.0f;
constexpr float kMinCenterWidth = 240.0f;
constexpr float kMinRightWidth = 260.0f;
constexpr float kMinTopHeight = 140.0f;
constexpr float kMinBottomHeight = 120.0f;
constexpr std::array<std::array<uint8_t, 3>, 8> kSignalHighlightColors = {{
  {102, 86, 169},
  {69, 137, 255},
  {55, 171, 112},
  {232, 171, 44},
  {198, 89, 71},
  {92, 155, 181},
  {134, 172, 79},
  {150, 112, 63},
}};

std::optional<CanServiceKind> parse_can_service_kind(std::string_view service) {
  if (service == "can") return CanServiceKind::Can;
  if (service == "sendcan") return CanServiceKind::Sendcan;
  return std::nullopt;
}

const char *can_service_name(CanServiceKind service) {
  return service == CanServiceKind::Can ? "can" : "sendcan";
}

std::string format_can_address(uint32_t address) {
  char text[32];
  std::snprintf(text, sizeof(text), "0x%X", address);
  return text;
}

std::string cabana_message_id_label(const CabanaMessageSummary &message) {
  char text[32];
  std::snprintf(text, sizeof(text), "%d:%X", message.bus, message.address);
  return text;
}

std::string can_message_key(CanServiceKind service, uint8_t bus, uint32_t address) {
  return "/" + std::string(can_service_name(service)) + "/" + std::to_string(bus) + "/" + format_can_address(address);
}

int cabana_flip_bit_pos(int bit_pos) {
  return 8 * (bit_pos / 8) + 7 - (bit_pos % 8);
}

int cabana_visual_index(size_t byte_index, int bit_index) {
  return static_cast<int>(byte_index) * 8 + (7 - bit_index);
}

std::string sanitize_filename_component(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out.empty() ? "untitled" : out;
}

fs::path cabana_export_dir() {
  const char *home = std::getenv("HOME");
  fs::path root = home != nullptr ? fs::path(home) : fs::current_path();
  const fs::path downloads = root / "Downloads";
  return (fs::exists(downloads) ? downloads : root) / "jotpluggler_exports";
}

std::string csv_escape(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 2);
  out.push_back('"');
  for (char c : text) {
    if (c == '"') out.push_back('"');
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

std::string payload_hex(const std::string &data) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(data.size() * 2);
  for (unsigned char byte : data) {
    out.push_back(kHex[byte >> 4]);
    out.push_back(kHex[byte & 0xF]);
  }
  return out;
}

fs::path cabana_export_path(const AppSession &session,
                            const CabanaMessageSummary &message,
                            std::string_view kind) {
  const std::string route_part = sanitize_filename_component(session.route_name.empty() ? "stream" : session.route_name);
  char filename[256];
  std::snprintf(filename, sizeof(filename), "%s_%s_bus%d_0x%X_%.*s.csv",
                sanitize_filename_component(message.name).c_str(),
                sanitize_filename_component(message.service).c_str(),
                message.bus,
                message.address,
                static_cast<int>(kind.size()),
                kind.data());
  return cabana_export_dir() / route_part / filename;
}

std::optional<dbc::Database> load_active_dbc(const AppSession &session) {
  const std::string &dbc_name = !session.dbc_override.empty() ? session.dbc_override : session.route_data.dbc_name;
  return load_dbc_by_name(dbc_name);
}

struct BitBehaviorStats {
  double ones_ratio = 0.0;
  double flip_ratio = 0.0;
  size_t samples = 0;
};

struct BinaryMatrixLayout {
  size_t byte_count = 0;
  std::vector<std::vector<int>> cell_signals;
  std::vector<bool> is_msb;
  std::vector<bool> is_lsb;
  size_t overlapping_cells = 0;
};

bool signal_contains_bit(const CabanaSignalSummary &signal, size_t byte_index, int bit_index);
void sync_cabana_selection(AppSession *session, UiState *state);
void clear_similar_bit_results(UiState *state);
const CabanaSignalSummary *find_signal_by_path(const CabanaMessageSummary &message, std::string_view path);
bool prepare_cabana_signal_editor(const AppSession &session,
                                  UiState *state,
                                  const CabanaMessageSummary &message,
                                  const CabanaSignalSummary &signal);
bool prepare_cabana_new_signal_editor(const AppSession &session,
                                      UiState *state,
                                      const CabanaMessageSummary &message,
                                      int start_bit,
                                      int size,
                                      bool is_little_endian);

void clear_cabana_binary_drag(UiState *state) {
  state->cabana.binary_drag_active = false;
  state->cabana.binary_drag_resizing = false;
  state->cabana.binary_drag_moved = false;
  state->cabana.binary_drag_signal_is_little_endian = true;
  state->cabana.binary_drag_press_byte = -1;
  state->cabana.binary_drag_press_bit = -1;
  state->cabana.binary_drag_anchor_byte = -1;
  state->cabana.binary_drag_anchor_bit = -1;
  state->cabana.binary_drag_current_byte = -1;
  state->cabana.binary_drag_current_bit = -1;
  state->cabana.binary_drag_signal_path.clear();
}

std::string_view active_signal_path(const UiState &state) {
  if (!state.cabana.selected_signal_path.empty()) {
    return state.cabana.selected_signal_path;
  }
  return {};
}

void set_cabana_selected_bit(UiState *state, int byte_index, int bit_index) {
  const bool changed = !state->cabana.has_bit_selection
                    || state->cabana.selected_bit_byte != byte_index
                    || state->cabana.selected_bit_index != bit_index;
  state->cabana.has_bit_selection = true;
  state->cabana.selected_bit_byte = byte_index;
  state->cabana.selected_bit_index = bit_index;
  if (changed) {
    clear_similar_bit_results(state);
  }
}

bool cabana_drag_has_selection(const UiState &state) {
  return state.cabana.binary_drag_active
      && state.cabana.binary_drag_anchor_byte >= 0
      && state.cabana.binary_drag_anchor_bit >= 0
      && state.cabana.binary_drag_current_byte >= 0
      && state.cabana.binary_drag_current_bit >= 0;
}

bool cabana_drag_selection(const UiState &state, int *start_bit, int *size, bool *is_little_endian) {
  if (!cabana_drag_has_selection(state)) {
    return false;
  }
  const int anchor_visual = cabana_visual_index(static_cast<size_t>(state.cabana.binary_drag_anchor_byte),
                                                state.cabana.binary_drag_anchor_bit);
  const int current_visual = cabana_visual_index(static_cast<size_t>(state.cabana.binary_drag_current_byte),
                                                 state.cabana.binary_drag_current_bit);
  const int anchor_bit_pos = state.cabana.binary_drag_anchor_byte * 8 + state.cabana.binary_drag_anchor_bit;
  const int current_bit_pos = state.cabana.binary_drag_current_byte * 8 + state.cabana.binary_drag_current_bit;

  bool little_endian = true;
  if (state.cabana.binary_drag_resizing) {
    little_endian = state.cabana.binary_drag_signal_is_little_endian;
  } else {
    // Match old Cabana's default MsbFirst drag direction.
    little_endian = current_visual < anchor_visual;
  }

  const int visual_min = std::min(anchor_visual, current_visual);
  *start_bit = little_endian ? std::min(anchor_bit_pos, current_bit_pos)
                             : cabana_flip_bit_pos(visual_min);
  *size = little_endian ? std::abs(current_bit_pos - anchor_bit_pos) + 1
                        : std::abs(cabana_flip_bit_pos(current_bit_pos) - cabana_flip_bit_pos(anchor_bit_pos)) + 1;
  *is_little_endian = little_endian;
  return *size > 0;
}

bool cabana_selection_contains_bit(int start_bit, int size, bool is_little_endian, size_t byte_index, int bit_index) {
  dbc::Signal signal;
  signal.start_bit = start_bit;
  signal.size = size;
  signal.is_little_endian = is_little_endian;
  dbc::updateMsbLsb(&signal);
  CabanaSignalSummary summary{
    .start_bit = signal.start_bit,
    .msb = signal.msb,
    .lsb = signal.lsb,
    .size = signal.size,
    .is_little_endian = signal.is_little_endian,
    .has_bit_range = true,
  };
  return signal_contains_bit(summary, byte_index, bit_index);
}

bool cabana_drag_selection_contains_bit(const UiState &state, size_t byte_index, int bit_index) {
  int start_bit = 0;
  int size = 0;
  bool is_little_endian = true;
  return cabana_drag_selection(state, &start_bit, &size, &is_little_endian)
      && cabana_selection_contains_bit(start_bit, size, is_little_endian, byte_index, bit_index);
}

bool cabana_drag_selection_has_neighbor(const UiState &state, int byte_index, int bit_index) {
  if (byte_index < 0 || bit_index < 0 || bit_index > 7) {
    return false;
  }
  return cabana_drag_selection_contains_bit(state, static_cast<size_t>(byte_index), bit_index);
}

bool queue_binary_drag_apply(AppSession *session,
                             const CabanaMessageSummary &summary,
                             UiState *state,
                             int release_byte,
                             int release_bit,
                             const CabanaSignalSummary *clicked_signal) {
  if (!state->cabana.binary_drag_active) {
    return false;
  }
  if (state->cabana.binary_drag_moved) {
    int start_bit = 0;
    int size = 0;
    bool is_little_endian = true;
    if (!cabana_drag_selection(*state, &start_bit, &size, &is_little_endian) || size <= 0) {
      return false;
    }
    bool queued_apply = false;
    if (state->cabana.binary_drag_resizing) {
      const CabanaSignalSummary *target = find_signal_by_path(summary, state->cabana.binary_drag_signal_path);
      if (target != nullptr && prepare_cabana_signal_editor(*session, state, summary, *target)) {
        state->cabana_signal_editor.open = false;
        state->cabana_signal_editor.start_bit = start_bit;
        state->cabana_signal_editor.size = size;
        state->cabana_signal_editor.is_little_endian = is_little_endian;
        state->cabana.pending_apply_signal_edit = true;
        queued_apply = true;
      }
    } else if (size > 1 && prepare_cabana_new_signal_editor(*session, state, summary, start_bit, size, is_little_endian)) {
      state->cabana_signal_editor.open = false;
      state->cabana.pending_apply_signal_edit = true;
      queued_apply = true;
    }
    if (queued_apply) {
      set_cabana_selected_bit(state, release_byte, release_bit);
    }
    return queued_apply;
  }
  if (clicked_signal != nullptr) {
    state->cabana.selected_signal_path = clicked_signal->path;
  }
  set_cabana_selected_bit(state, release_byte, release_bit);
  return true;
}

bool contains_case_insensitive(std::string_view haystack, std::string_view needle) {
  if (needle.empty()) {
    return true;
  }
  const std::string hay = lowercase(haystack);
  const std::string ndl = lowercase(needle);
  return hay.find(ndl) != std::string::npos;
}

bool cabana_match_numeric_filter(std::string_view filter, double value) {
  const std::string raw = trim_copy(filter);
  if (raw.empty()) {
    return true;
  }
  const size_t dash = raw.find('-');
  if (dash != std::string::npos) {
    if (raw.find('-', dash + 1) != std::string::npos) {
      return false;
    }
    const std::string lo_text = raw.substr(0, dash);
    const std::string hi_text = raw.substr(dash + 1);
    char *lo_end = nullptr;
    char *hi_end = nullptr;
    const double lo = lo_text.empty() ? -1.0e18 : std::strtod(lo_text.c_str(), &lo_end);
    const double hi = hi_text.empty() ?  1.0e18 : std::strtod(hi_text.c_str(), &hi_end);
    if ((!lo_text.empty() && lo_end == lo_text.c_str()) || (!hi_text.empty() && hi_end == hi_text.c_str())) {
      return false;
    }
    return value >= lo && value <= hi;
  }
  char *end = nullptr;
  const double target = std::strtod(raw.c_str(), &end);
  return end != raw.c_str() && static_cast<int>(value) == static_cast<int>(target);
}

bool cabana_match_address_filter(std::string_view filter, uint32_t address) {
  const std::string raw = trim_copy(filter);
  if (raw.empty()) {
    return true;
  }
  const size_t dash = raw.find('-');
  if (dash != std::string::npos && dash > 0 && dash + 1 < raw.size()) {
    const std::string lo_text = raw.substr(0, dash);
    const std::string hi_text = raw.substr(dash + 1);
    char *lo_end = nullptr;
    char *hi_end = nullptr;
    const unsigned long lo = std::strtoul(lo_text.c_str(), &lo_end, 16);
    const unsigned long hi = std::strtoul(hi_text.c_str(), &hi_end, 16);
    if (lo_end != lo_text.c_str() && hi_end != hi_text.c_str()) {
      return address >= lo && address <= hi;
    }
  }
  return contains_case_insensitive(format_can_address(address), raw);
}

bool cabana_message_matches_filters(const CabanaMessageSummary &message,
                                    std::string_view name_filter,
                                    std::string_view bus_filter,
                                    std::string_view addr_filter,
                                    std::string_view node_filter,
                                    std::string_view freq_filter,
                                    std::string_view count_filter,
                                    std::string_view bytes_filter,
                                    const CanMessageData *message_data) {
  const bool name_matches = [&] {
    if (name_filter.empty()) {
      return true;
    }
    if (contains_case_insensitive(message.name, name_filter)) {
      return true;
    }
    return std::any_of(message.signals.begin(), message.signals.end(), [&](const CabanaSignalSummary &signal) {
      return contains_case_insensitive(signal.name, name_filter);
    });
  }();
  const bool bytes_matches = [&] {
    if (bytes_filter.empty()) {
      return true;
    }
    if (message_data == nullptr || message_data->samples.empty()) {
      return false;
    }
    return contains_case_insensitive(payload_hex(message_data->samples.back().data), trim_copy(bytes_filter));
  }();
  return name_matches
      && cabana_match_numeric_filter(bus_filter, message.bus)
      && cabana_match_address_filter(addr_filter, message.address)
      && contains_case_insensitive(message.node, node_filter)
      && cabana_match_numeric_filter(freq_filter, message.frequency_hz)
      && cabana_match_numeric_filter(count_filter, static_cast<double>(message.sample_count))
      && bytes_matches;
}

const CanMessageData *find_message_data(const AppSession &session, const CabanaMessageSummary &message) {
  const std::optional<CanServiceKind> service = parse_can_service_kind(message.service);
  if (!service.has_value()) {
    return nullptr;
  }
  const CanMessageData key{.id = CanMessageId{*service, static_cast<uint8_t>(message.bus), message.address}};
  auto it = std::lower_bound(session.route_data.can_messages.begin(),
                             session.route_data.can_messages.end(),
                             key,
                             [](const CanMessageData &a, const CanMessageData &b) {
                               return std::make_tuple(a.id.service, a.id.bus, a.id.address)
                                    < std::make_tuple(b.id.service, b.id.bus, b.id.address);
                             });
  if (it == session.route_data.can_messages.end()
      || it->id.service != key.id.service
      || it->id.bus != key.id.bus
      || it->id.address != key.id.address) {
    return nullptr;
  }
  return &*it;
}

bool prepare_cabana_signal_editor(const AppSession &session,
                                  UiState *state,
                                  const CabanaMessageSummary &message,
                                  const CabanaSignalSummary &signal) {
  const std::optional<dbc::Database> db = load_active_dbc(session);
  const dbc::Message *dbc_message = db.has_value() ? db->message(message.address) : nullptr;
  if (dbc_message == nullptr) {
    state->error_text = "No active DBC message available for editing";
    state->open_error_popup = true;
    return false;
  }
  auto it = std::find_if(dbc_message->signals.begin(), dbc_message->signals.end(), [&](const dbc::Signal &dbc_signal) {
    return dbc_signal.name == signal.name;
  });
  if (it == dbc_message->signals.end()) {
    state->error_text = "Signal not found in active DBC";
    state->open_error_popup = true;
    return false;
  }

  CabanaSignalEditorState &editor = state->cabana_signal_editor;
  editor.loaded = true;
  editor.creating = false;
  editor.message_root = message.root_path;
  editor.message_name = message.name;
  editor.service = message.service;
  editor.signal_path = signal.path;
  editor.bus = message.bus;
  editor.message_address = message.address;
  editor.original_signal_name = it->name;
  editor.signal_name = it->name;
  editor.start_bit = it->start_bit;
  editor.size = it->size;
  editor.factor = it->factor;
  editor.offset = it->offset;
  editor.min = it->min;
  editor.max = it->max;
  editor.is_signed = it->is_signed;
  editor.is_little_endian = it->is_little_endian;
  editor.type = static_cast<int>(it->type);
  editor.multiplex_value = it->multiplex_value;
  editor.receiver_name = it->receiver_name;
  editor.unit = it->unit;
  return true;
}

bool prepare_cabana_new_signal_editor(const AppSession &session,
                                      UiState *state,
                                      const CabanaMessageSummary &message,
                                      int start_bit,
                                      int size,
                                      bool is_little_endian) {
  const std::optional<dbc::Database> db = load_active_dbc(session);
  const dbc::Message *dbc_message = db.has_value() ? db->message(message.address) : nullptr;
  if (dbc_message == nullptr) {
    state->error_text = "No active DBC message available for creating a signal";
    state->open_error_popup = true;
    return false;
  }

  const int byte_index = start_bit / 8;
  const int bit_index = start_bit & 7;
  std::string base_name = "bit_" + std::to_string(byte_index) + "_" + std::to_string(bit_index);
  std::string signal_name = base_name;
  int suffix = 2;
  auto exists = [&](std::string_view candidate) {
    return std::any_of(dbc_message->signals.begin(), dbc_message->signals.end(), [&](const dbc::Signal &signal) {
      return signal.name == candidate;
    });
  };
  while (exists(signal_name)) {
    signal_name = base_name + "_" + std::to_string(suffix++);
  }

  CabanaSignalEditorState &editor = state->cabana_signal_editor;
  editor.loaded = true;
  editor.creating = true;
  editor.message_root = message.root_path;
  editor.message_name = message.name;
  editor.service = message.service;
  editor.signal_path.clear();
  editor.bus = message.bus;
  editor.message_address = message.address;
  editor.original_signal_name.clear();
  editor.signal_name = signal_name;
  editor.start_bit = start_bit;
  editor.size = size;
  editor.factor = 1.0;
  editor.offset = 0.0;
  editor.min = 0.0;
  editor.max = std::min(std::pow(2.0, static_cast<double>(std::min(size, 24))) - 1.0, 1.0e9);
  editor.is_signed = false;
  editor.is_little_endian = is_little_endian;
  editor.type = static_cast<int>(dbc::Signal::Type::Normal);
  editor.multiplex_value = 0;
  editor.receiver_name = "XXX";
  editor.unit.clear();
  return true;
}

void open_cabana_new_signal_editor(const AppSession &session,
                                   UiState *state,
                                   const CabanaMessageSummary &message,
                                   int byte_index,
                                   int bit_index) {
  if (prepare_cabana_new_signal_editor(session, state, message, byte_index * 8 + bit_index, 1, true)) {
    state->cabana_signal_editor.open = true;
  }
}

const CabanaMessageSummary *find_selected_message(const AppSession &session, const UiState &state) {
  auto it = std::find_if(session.cabana_messages.begin(), session.cabana_messages.end(), [&](const CabanaMessageSummary &message) {
    return message.root_path == state.cabana.selected_message_root;
  });
  return it == session.cabana_messages.end() ? nullptr : &*it;
}

const CabanaMessageSummary *find_message_by_root(const AppSession &session, std::string_view root_path) {
  auto it = std::find_if(session.cabana_messages.begin(), session.cabana_messages.end(), [&](const CabanaMessageSummary &message) {
    return message.root_path == root_path;
  });
  return it == session.cabana_messages.end() ? nullptr : &*it;
}

void select_cabana_message(AppSession *session, UiState *state, std::string_view root_path) {
  state->cabana.selected_message_root.assign(root_path);
  state->cabana.sync_message_tabs = true;
  if (std::find(state->cabana.open_message_roots.begin(), state->cabana.open_message_roots.end(), root_path)
      == state->cabana.open_message_roots.end()) {
    state->cabana.open_message_roots.emplace_back(root_path);
  }
  state->cabana.signal_filter[0] = '\0';
  state->cabana.selected_signal_path.clear();
  state->cabana.detail_top_auto_fit = true;
  state->cabana.has_bit_selection = false;
  clear_similar_bit_results(state);
  clear_cabana_binary_drag(state);
  sync_cabana_selection(session, state);
}

void close_cabana_message_tab(AppSession *session, UiState *state, std::string_view root_path) {
  auto &roots = state->cabana.open_message_roots;
  auto it = std::find(roots.begin(), roots.end(), root_path);
  if (it == roots.end()) {
    return;
  }
  const bool closing_selected = state->cabana.selected_message_root == root_path;
  const size_t index = static_cast<size_t>(it - roots.begin());
  roots.erase(it);
  if (!closing_selected) {
    return;
  }
  if (roots.empty()) {
    state->cabana.selected_message_root.clear();
    state->cabana.selected_signal_path.clear();
    state->cabana.has_bit_selection = false;
    clear_similar_bit_results(state);
    clear_cabana_binary_drag(state);
    return;
  }
  const size_t next_index = std::min(index, roots.size() - 1);
  select_cabana_message(session, state, roots[next_index]);
}

bool similar_bit_results_match_selection(const UiState &state) {
  return state.cabana.has_bit_selection
      && state.cabana.similar_bits_source_root == state.cabana.selected_message_root
      && state.cabana.similar_bits_source_byte == state.cabana.selected_bit_byte
      && state.cabana.similar_bits_source_bit == state.cabana.selected_bit_index;
}

void clear_similar_bit_results(UiState *state) {
  state->cabana.similar_bits_source_root.clear();
  state->cabana.similar_bits_source_byte = -1;
  state->cabana.similar_bits_source_bit = -1;
  state->cabana.similar_bit_matches.clear();
}

void poll_similar_bit_search(UiState *state) {
  if (!state->cabana.similar_bits_loading || !state->cabana.similar_bit_future.valid()) {
    return;
  }
  using namespace std::chrono_literals;
  if (state->cabana.similar_bit_future.wait_for(0ms) != std::future_status::ready) {
    return;
  }
  std::vector<CabanaSimilarBitMatch> matches = state->cabana.similar_bit_future.get();
  state->cabana.similar_bits_loading = false;
  if (similar_bit_results_match_selection(*state)) {
    state->cabana.similar_bit_matches = std::move(matches);
  } else {
    clear_similar_bit_results(state);
  }
}

void sync_cabana_selection(AppSession *session, UiState *state) {
  poll_similar_bit_search(state);
  if (!state->cabana_mode_initialized) {
    state->cabana.camera_view = sidebar_preview_camera_view(*session);
    state->cabana_mode_initialized = true;
  }
  auto &open_roots = state->cabana.open_message_roots;
  open_roots.erase(std::remove_if(open_roots.begin(), open_roots.end(), [&](const std::string &root_path) {
                     return find_message_by_root(*session, root_path) == nullptr;
                   }),
                   open_roots.end());
  if (session->cabana_messages.empty()) {
    state->cabana.selected_message_root.clear();
    state->cabana.selected_signal_path.clear();
    state->cabana.open_message_roots.clear();
    state->cabana.chart_signal_paths.clear();
    state->cabana.has_bit_selection = false;
    clear_similar_bit_results(state);
    clear_cabana_binary_drag(state);
    return;
  }
  const CabanaMessageSummary *selected = find_selected_message(*session, *state);
  if (selected == nullptr) {
    state->cabana.selected_message_root.clear();
    state->cabana.selected_signal_path.clear();
    state->cabana.has_bit_selection = false;
    clear_similar_bit_results(state);
    clear_cabana_binary_drag(state);
    return;
  }

  std::unordered_set<std::string> allowed;
  allowed.reserve(selected->signals.size());
  for (const CabanaSignalSummary &signal : selected->signals) {
    allowed.insert(signal.path);
  }
  state->cabana.chart_signal_paths.erase(
    std::remove_if(state->cabana.chart_signal_paths.begin(), state->cabana.chart_signal_paths.end(),
                   [&](const std::string &path) { return session->series_by_path.find(path) == session->series_by_path.end(); }),
    state->cabana.chart_signal_paths.end());
  if (!state->cabana.selected_signal_path.empty() && !allowed.count(state->cabana.selected_signal_path)) {
    state->cabana.selected_signal_path.clear();
  }
}

std::string format_cabana_time(double seconds) {
  seconds = std::max(0.0, seconds);
  const int total = static_cast<int>(seconds);
  const int minutes = total / 60;
  const int secs = total % 60;
  char text[32];
  std::snprintf(text, sizeof(text), "%02d:%02d", minutes, secs);
  return text;
}

void draw_cabana_message_tabs(AppSession *session, UiState *state) {
  auto &roots = state->cabana.open_message_roots;
  if (roots.size() <= 1) {
    return;
  }
  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.0f, 0.0f));
  ImGui::BeginChild("##cabana_message_tabs", ImVec2(0.0f, 28.0f), false, ImGuiWindowFlags_NoScrollbar);
  int close_index = -1;
  int close_others_index = -1;
  if (ImGui::BeginTabBar("##cabana_message_tabbar",
                         ImGuiTabBarFlags_FittingPolicyResizeDown |
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_NoTooltip)) {
  for (size_t i = 0; i < roots.size(); ++i) {
    const CabanaMessageSummary *message = find_message_by_root(*session, roots[i]);
    if (message == nullptr) {
      continue;
    }
    const std::string message_id = cabana_message_id_label(*message);
    const std::string label = message_id + "###cabana_tab_" + roots[i];
    bool open = true;
    const ImGuiTabItemFlags flags = (state->cabana.sync_message_tabs && state->cabana.selected_message_root == roots[i])
                                  ? ImGuiTabItemFlags_SetSelected
                                  : 0;
    if (ImGui::BeginTabItem(label.c_str(), &open, flags)) {
      if (!state->cabana.sync_message_tabs && state->cabana.selected_message_root != roots[i]) {
        select_cabana_message(session, state, roots[i]);
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", message->name.c_str());
      }
      if (ImGui::BeginPopupContextItem(("##cabana_tab_ctx_" + roots[i]).c_str())) {
        if (ImGui::MenuItem("Close Other Tabs", nullptr, false, roots.size() > 1)) {
          close_others_index = static_cast<int>(i);
        }
        ImGui::EndPopup();
      }
      ImGui::EndTabItem();
    }
    if (!open) {
      close_index = static_cast<int>(i);
    }
  }
    ImGui::EndTabBar();
  }
  if (close_others_index >= 0) {
    const std::string keep = roots[static_cast<size_t>(close_others_index)];
    roots.assign(1, keep);
    select_cabana_message(session, state, keep);
  } else if (close_index >= 0) {
    close_cabana_message_tab(session, state, roots[static_cast<size_t>(close_index)]);
  }
  state->cabana.sync_message_tabs = false;
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
}

bool export_raw_can_csv(const AppSession &session,
                        const CabanaMessageSummary &message,
                        std::string *error,
                        fs::path *output_path) {
  const CanMessageData *message_data = find_message_data(session, message);
  if (message_data == nullptr || message_data->samples.empty()) {
    if (error != nullptr) *error = "No raw CAN frames available";
    return false;
  }

  const fs::path output = cabana_export_path(session, message, "raw");
  fs::create_directories(output.parent_path());
  std::ofstream out(output);
  if (!out.is_open()) {
    if (error != nullptr) *error = "Failed to open raw CSV";
    return false;
  }

  out << "mono_time,bus_time,dt_ms,data_hex\n";
  for (size_t i = 0; i < message_data->samples.size(); ++i) {
    const CanFrameSample &sample = message_data->samples[i];
    out << sample.mono_time << ','
        << sample.bus_time << ',';
    if (i > 0) {
      out << 1000.0 * (sample.mono_time - message_data->samples[i - 1].mono_time);
    }
    out << ',' << csv_escape(payload_hex(sample.data)) << '\n';
  }

  if (!out.good()) {
    if (error != nullptr) *error = "Failed while writing raw CSV";
    return false;
  }
  if (output_path != nullptr) *output_path = output;
  return true;
}

bool export_decoded_can_csv(const AppSession &session,
                            const CabanaMessageSummary &message,
                            std::string *error,
                            fs::path *output_path) {
  const CanMessageData *message_data = find_message_data(session, message);
  if (message_data == nullptr || message_data->samples.empty()) {
    if (error != nullptr) *error = "No raw CAN frames available";
    return false;
  }
  if (message.signals.empty()) {
    if (error != nullptr) *error = "No decoded signals for this message";
    return false;
  }

  std::vector<const CabanaSignalSummary *> export_signals;
  std::vector<const RouteSeries *> export_series;
  std::vector<const SeriesFormat *> export_formats;
  std::vector<const EnumInfo *> export_enums;
  export_signals.reserve(message.signals.size());
  export_series.reserve(message.signals.size());
  export_formats.reserve(message.signals.size());
  export_enums.reserve(message.signals.size());

  for (const CabanaSignalSummary &signal : message.signals) {
    const RouteSeries *series = app_find_route_series(session, signal.path);
    if (series == nullptr) {
      continue;
    }
    export_signals.push_back(&signal);
    export_series.push_back(series);
    auto format_it = session.route_data.series_formats.find(signal.path);
    export_formats.push_back(format_it == session.route_data.series_formats.end() ? nullptr : &format_it->second);
    auto enum_it = session.route_data.enum_info.find(signal.path);
    export_enums.push_back(enum_it == session.route_data.enum_info.end() ? nullptr : &enum_it->second);
  }

  if (export_series.empty()) {
    if (error != nullptr) *error = "No decoded signal data available";
    return false;
  }

  const fs::path output = cabana_export_path(session, message, "decoded");
  fs::create_directories(output.parent_path());
  std::ofstream out(output);
  if (!out.is_open()) {
    if (error != nullptr) *error = "Failed to open decoded CSV";
    return false;
  }

  out << "mono_time,bus_time,dt_ms,data_hex";
  for (const CabanaSignalSummary *signal : export_signals) {
    out << ',' << csv_escape(signal->name);
  }
  out << '\n';

  for (size_t i = 0; i < message_data->samples.size(); ++i) {
    const CanFrameSample &sample = message_data->samples[i];
    out << sample.mono_time << ','
        << sample.bus_time << ',';
    if (i > 0) {
      out << 1000.0 * (sample.mono_time - message_data->samples[i - 1].mono_time);
    }
    out << ',' << csv_escape(payload_hex(sample.data));
    for (size_t j = 0; j < export_series.size(); ++j) {
      out << ',';
      const std::optional<double> value = app_sample_xy_value_at_time(
        export_series[j]->times, export_series[j]->values, false, sample.mono_time);
      if (value.has_value()) {
        out << csv_escape(export_formats[j] != nullptr
                            ? format_display_value(*value, *export_formats[j], export_enums[j])
                            : std::to_string(*value));
      }
    }
    out << '\n';
  }

  if (!out.good()) {
    if (error != nullptr) *error = "Failed while writing decoded CSV";
    return false;
  }
  if (output_path != nullptr) *output_path = output;
  return true;
}

size_t closest_can_sample_index(const CanMessageData &message, double tracker_time) {
  if (message.samples.empty()) {
    return 0;
  }
  auto it = std::lower_bound(message.samples.begin(), message.samples.end(), tracker_time,
                             [](const CanFrameSample &sample, double time) {
                               return sample.mono_time < time;
                             });
  if (it == message.samples.begin()) {
    return 0;
  }
  if (it == message.samples.end()) {
    return message.samples.size() - 1;
  }
  const size_t upper = static_cast<size_t>(it - message.samples.begin());
  const size_t lower = upper - 1;
  return std::abs(message.samples[upper].mono_time - tracker_time) < std::abs(message.samples[lower].mono_time - tracker_time)
    ? upper
    : lower;
}

uint8_t can_bit(const std::string &data, size_t byte_index, int bit_index) {
  if (byte_index >= data.size() || bit_index < 0 || bit_index > 7) {
    return 0;
  }
  return (static_cast<uint8_t>(data[byte_index]) >> bit_index) & 0x1;
}

BitBehaviorStats bit_behavior_stats(const CanMessageData &message, size_t byte_index, int bit_index) {
  BitBehaviorStats stats;
  if (message.samples.empty()) {
    return stats;
  }
  size_t ones = 0;
  size_t flips = 0;
  uint8_t prev = can_bit(message.samples.front().data, byte_index, bit_index);
  ones += prev;
  for (size_t i = 1; i < message.samples.size(); ++i) {
    const uint8_t bit = can_bit(message.samples[i].data, byte_index, bit_index);
    ones += bit;
    flips += bit != prev;
    prev = bit;
  }
  stats.samples = message.samples.size();
  stats.ones_ratio = static_cast<double>(ones) / static_cast<double>(stats.samples);
  stats.flip_ratio = stats.samples > 1 ? static_cast<double>(flips) / static_cast<double>(stats.samples - 1) : 0.0;
  return stats;
}

size_t can_message_payload_width(const CanMessageData &message) {
  size_t width = 0;
  for (const CanFrameSample &sample : message.samples) {
    width = std::max(width, sample.data.size());
  }
  return width;
}

size_t cabana_signal_byte_count(const CabanaMessageSummary &message) {
  size_t width = 0;
  for (const CabanaSignalSummary &signal : message.signals) {
    if (!signal.has_bit_range) {
      continue;
    }
    width = std::max(width, static_cast<size_t>(std::max(signal.msb / 8, signal.lsb / 8) + 1));
  }
  return width;
}

BinaryMatrixLayout build_binary_matrix_layout(const CabanaMessageSummary &message, size_t byte_count) {
  BinaryMatrixLayout layout;
  layout.byte_count = byte_count;
  layout.cell_signals.resize(byte_count * 8);
  layout.is_msb.assign(byte_count * 8, false);
  layout.is_lsb.assign(byte_count * 8, false);
  for (size_t i = 0; i < message.signals.size(); ++i) {
    const CabanaSignalSummary &signal = message.signals[i];
    if (!signal.has_bit_range) {
      continue;
    }
    for (size_t byte = 0; byte < byte_count; ++byte) {
      for (int bit = 0; bit < 8; ++bit) {
        if (signal_contains_bit(signal, byte, bit)) {
          layout.cell_signals[byte * 8 + static_cast<size_t>(bit)].push_back(static_cast<int>(i));
        }
      }
    }
    const size_t msb_byte = static_cast<size_t>(signal.msb / 8);
    const size_t lsb_byte = static_cast<size_t>(signal.lsb / 8);
    if (msb_byte < byte_count) {
      layout.is_msb[msb_byte * 8 + static_cast<size_t>(signal.msb & 7)] = true;
    }
    if (lsb_byte < byte_count) {
      layout.is_lsb[lsb_byte * 8 + static_cast<size_t>(signal.lsb & 7)] = true;
    }
  }
  for (std::vector<int> &signals : layout.cell_signals) {
    std::stable_sort(signals.begin(), signals.end(), [&](int a, int b) {
      return message.signals[static_cast<size_t>(a)].size > message.signals[static_cast<size_t>(b)].size;
    });
    if (signals.size() > 1) {
      ++layout.overlapping_cells;
    }
  }
  return layout;
}

bool cell_has_signal(const BinaryMatrixLayout &layout, int byte_index, int bit_index, int signal_index) {
  if (byte_index < 0 || bit_index < 0 || bit_index > 7) {
    return false;
  }
  if (static_cast<size_t>(byte_index) >= layout.byte_count) {
    return false;
  }
  const std::vector<int> &signals = layout.cell_signals[static_cast<size_t>(byte_index) * 8 + static_cast<size_t>(bit_index)];
  return std::find(signals.begin(), signals.end(), signal_index) != signals.end();
}

std::vector<float> compute_bit_flip_heat(const CanMessageData &message,
                                         size_t byte_count,
                                         bool live_mode,
                                         size_t tracker_index) {
  std::vector<float> heat(byte_count * 8, 0.0f);
  if (message.samples.size() < 2 || byte_count == 0) {
    return heat;
  }

  size_t begin = 0;
  size_t end = message.samples.size();
  if (live_mode) {
    end = std::min(message.samples.size(), tracker_index + 1);
    const size_t window = 96;
    begin = end > window ? end - window : 0;
  }
  if (end <= begin + 1) {
    return heat;
  }

  std::vector<uint32_t> flip_counts(byte_count * 8, 0);
  uint32_t max_count = 1;
  std::string prev = message.samples[begin].data;
  for (size_t i = begin + 1; i < end; ++i) {
    const std::string &current = message.samples[i].data;
    for (size_t byte = 0; byte < byte_count; ++byte) {
      const uint8_t before = byte < prev.size() ? static_cast<uint8_t>(prev[byte]) : 0;
      const uint8_t after = byte < current.size() ? static_cast<uint8_t>(current[byte]) : 0;
      const uint8_t diff = before ^ after;
      if (diff == 0) {
        continue;
      }
      for (int bit = 0; bit < 8; ++bit) {
        if ((diff & (1u << bit)) == 0) {
          continue;
        }
        uint32_t &count = flip_counts[byte * 8 + static_cast<size_t>(bit)];
        ++count;
        max_count = std::max(max_count, count);
      }
    }
    prev = current;
  }

  for (size_t i = 0; i < flip_counts.size(); ++i) {
    if (flip_counts[i] == 0) {
      continue;
    }
    const float frac = static_cast<float>(flip_counts[i]) / static_cast<float>(max_count);
    heat[i] = std::sqrt(frac);
  }
  return heat;
}

bool signal_charted(const UiState &state, std::string_view path) {
  return std::find(state.cabana.chart_signal_paths.begin(), state.cabana.chart_signal_paths.end(), path)
      != state.cabana.chart_signal_paths.end();
}

ImU32 signal_fill_color(size_t index, float alpha_scale, bool emphasized) {
  const auto &rgb = kSignalHighlightColors[index % kSignalHighlightColors.size()];
  const float alpha = emphasized ? std::clamp(0.34f + alpha_scale * 0.38f, 0.34f, 0.78f)
                                 : std::clamp(0.14f + alpha_scale * 0.28f, 0.14f, 0.48f);
  return ImGui::GetColorU32(color_rgb(rgb, alpha));
}

ImU32 signal_border_color(size_t index, bool emphasized) {
  const auto &rgb = kSignalHighlightColors[index % kSignalHighlightColors.size()];
  return ImGui::GetColorU32(color_rgb(rgb[0], rgb[1], rgb[2], emphasized ? 0.95f : 0.78f));
}

void draw_cell_hatching(ImDrawList *draw, const ImRect &rect, ImU32 color, float spacing) {
  for (float x = rect.Min.x - rect.GetHeight(); x < rect.Max.x; x += spacing) {
    const ImVec2 a(std::max(rect.Min.x, x), std::min(rect.Max.y, rect.Min.y + (rect.Min.x - x) + rect.GetHeight()));
    const ImVec2 b(std::min(rect.Max.x, x + rect.GetHeight()), std::max(rect.Min.y, rect.Max.y - (rect.Max.x - x)));
    draw->AddLine(a, b, color, 1.0f);
  }
}

bool signal_contains_bit(const CabanaSignalSummary &signal, size_t byte_index, int bit_index) {
  if (!signal.has_bit_range || bit_index < 0 || bit_index > 7) {
    return false;
  }
  const int msb_byte = signal.msb / 8;
  const int lsb_byte = signal.lsb / 8;
  if (msb_byte == lsb_byte) {
    return static_cast<int>(byte_index) == msb_byte
        && bit_index >= (signal.lsb & 7)
        && bit_index <= (signal.msb & 7);
  }
  for (int i = msb_byte, step = signal.is_little_endian ? -1 : 1;; i += step) {
    const int hi = i == msb_byte ? (signal.msb & 7) : 7;
    const int lo = i == lsb_byte ? (signal.lsb & 7) : 0;
    if (static_cast<int>(byte_index) == i && bit_index >= lo && bit_index <= hi) {
      return true;
    }
    if (i == lsb_byte) {
      return false;
    }
  }
}

std::vector<std::pair<const CabanaSignalSummary *, ImU32>> highlighted_signals(const CabanaMessageSummary &message, const UiState &state) {
  std::vector<std::pair<const CabanaSignalSummary *, ImU32>> out;
  for (const std::string &path : state.cabana.chart_signal_paths) {
    for (size_t i = 0; i < message.signals.size(); ++i) {
      const CabanaSignalSummary &signal = message.signals[i];
      if (signal.path != path || !signal.has_bit_range) {
        continue;
      }
      out.push_back({&signal, signal_fill_color(i, 0.5f, true)});
      break;
    }
  }
  return out;
}

bool cabana_bit_selected(const UiState &state, size_t byte_index, int bit_index) {
  return state.cabana.has_bit_selection
      && state.cabana.selected_bit_byte == static_cast<int>(byte_index)
      && state.cabana.selected_bit_index == bit_index;
}

std::vector<const CabanaSignalSummary *> selected_bit_signals(const CabanaMessageSummary &message, const UiState &state) {
  std::vector<const CabanaSignalSummary *> out;
  if (!state.cabana.has_bit_selection) {
    return out;
  }
  for (const CabanaSignalSummary &signal : message.signals) {
    if (signal_contains_bit(signal,
                            static_cast<size_t>(state.cabana.selected_bit_byte),
                            state.cabana.selected_bit_index)) {
      out.push_back(&signal);
    }
  }
  return out;
}

const CabanaSignalSummary *find_signal_by_path(const CabanaMessageSummary &message, std::string_view path) {
  auto it = std::find_if(message.signals.begin(), message.signals.end(), [&](const CabanaSignalSummary &signal) {
    return signal.path == path;
  });
  return it == message.signals.end() ? nullptr : &*it;
}

const CabanaSignalSummary *topmost_signal_at_cell(const CabanaMessageSummary &message,
                                                  const BinaryMatrixLayout &layout,
                                                  size_t byte_index,
                                                  int bit_index) {
  if (byte_index >= layout.byte_count || bit_index < 0 || bit_index > 7) {
    return nullptr;
  }
  const std::vector<int> &signals = layout.cell_signals[byte_index * 8 + static_cast<size_t>(bit_index)];
  return signals.empty() ? nullptr : &message.signals[static_cast<size_t>(signals.back())];
}

const CabanaSignalSummary *resize_signal_at_cell(const CabanaMessageSummary &message,
                                                 const BinaryMatrixLayout &layout,
                                                 size_t byte_index,
                                                 int bit_index) {
  if (byte_index >= layout.byte_count || bit_index < 0 || bit_index > 7) {
    return nullptr;
  }
  const int physical_bit = static_cast<int>(byte_index) * 8 + bit_index;
  const std::vector<int> &signals = layout.cell_signals[byte_index * 8 + static_cast<size_t>(bit_index)];
  for (int signal_index : signals) {
    const CabanaSignalSummary &signal = message.signals[static_cast<size_t>(signal_index)];
    if (signal.has_bit_range && (physical_bit == signal.msb || physical_bit == signal.lsb)) {
      return &signal;
    }
  }
  return nullptr;
}

std::vector<CabanaSimilarBitMatch> find_similar_bits_from_snapshot(const std::vector<CabanaMessageSummary> &messages,
                                                                   const std::vector<CanMessageData> &can_messages,
                                                                   const CabanaMessageSummary &source_message,
                                                                   const CanMessageData &source_data,
                                                                   size_t source_byte,
                                                                   int source_bit) {
  const BitBehaviorStats target = bit_behavior_stats(source_data, source_byte, source_bit);
  std::vector<CabanaSimilarBitMatch> matches;
  for (const CabanaMessageSummary &message : messages) {
    const std::optional<CanServiceKind> service = parse_can_service_kind(message.service);
    if (!service.has_value()) continue;
    const CanMessageData key{.id = CanMessageId{*service, static_cast<uint8_t>(message.bus), message.address}};
    auto it = std::lower_bound(can_messages.begin(), can_messages.end(), key, [](const CanMessageData &a, const CanMessageData &b) {
      return std::make_tuple(a.id.service, a.id.bus, a.id.address)
           < std::make_tuple(b.id.service, b.id.bus, b.id.address);
    });
    if (it == can_messages.end()
        || it->id.service != key.id.service
        || it->id.bus != key.id.bus
        || it->id.address != key.id.address
        || it->samples.size() < 2) {
      continue;
    }
    for (size_t byte = 0; byte < can_message_payload_width(*it); ++byte) {
      for (int bit = 0; bit < 8; ++bit) {
        if (message.root_path == source_message.root_path
            && static_cast<int>(byte) == static_cast<int>(source_byte)
            && bit == source_bit) {
          continue;
        }
        const BitBehaviorStats stats = bit_behavior_stats(*it, byte, bit);
        if (stats.samples < 2) continue;
        const double ones_diff = std::abs(stats.ones_ratio - target.ones_ratio);
        const double flip_diff = std::abs(stats.flip_ratio - target.flip_ratio);
        matches.push_back({
          .message_root = message.root_path,
          .label = message.name,
          .bus = message.bus,
          .address = message.address,
          .byte_index = static_cast<int>(byte),
          .bit_index = bit,
          .score = ones_diff * 0.65 + flip_diff * 0.35,
          .ones_ratio = stats.ones_ratio,
          .flip_ratio = stats.flip_ratio,
        });
      }
    }
  }
  std::sort(matches.begin(), matches.end(), [](const CabanaSimilarBitMatch &a, const CabanaSimilarBitMatch &b) {
    return std::tie(a.score, a.label, a.byte_index, a.bit_index)
         < std::tie(b.score, b.label, b.byte_index, b.bit_index);
  });
  if (matches.size() > 12) {
    matches.resize(12);
  }
  return matches;
}

void draw_bit_selection_panel(AppSession *session, const CabanaMessageSummary &message, UiState *state) {
  poll_similar_bit_search(state);
  if (!state->cabana.has_bit_selection) {
    return;
  }
  app_push_bold_font();
  ImGui::Text("Selected Bit: B%d.%d", state->cabana.selected_bit_byte, state->cabana.selected_bit_index);
  app_pop_bold_font();
  ImGui::SameLine();
  if (ImGui::SmallButton("Clear")) {
    state->cabana.has_bit_selection = false;
    clear_similar_bit_results(state);
    return;
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(state->cabana.similar_bits_loading);
  if (ImGui::SmallButton("Find Similar Bits")) {
    const CanMessageData *message_data = find_message_data(*session, message);
    if (message_data != nullptr) {
      state->cabana.similar_bit_matches.clear();
      state->cabana.similar_bits_source_root = message.root_path;
      state->cabana.similar_bits_source_byte = state->cabana.selected_bit_byte;
      state->cabana.similar_bits_source_bit = state->cabana.selected_bit_index;
      state->cabana.similar_bits_loading = true;
      const std::vector<CabanaMessageSummary> messages = session->cabana_messages;
      const std::vector<CanMessageData> can_messages = session->route_data.can_messages;
      const CanMessageData source_data = *message_data;
      const size_t source_byte = static_cast<size_t>(state->cabana.selected_bit_byte);
      const int source_bit = state->cabana.selected_bit_index;
      state->cabana.similar_bit_future = std::async(std::launch::async, [messages, can_messages, message, source_data, source_byte, source_bit]() {
        return find_similar_bits_from_snapshot(messages, can_messages, message, source_data, source_byte, source_bit);
      });
    }
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::SmallButton("Create Signal...")) {
    open_cabana_new_signal_editor(*session,
                                  state,
                                  message,
                                  state->cabana.selected_bit_byte,
                                  state->cabana.selected_bit_index);
  }
  const auto overlaps = selected_bit_signals(message, *state);
  if (overlaps.empty()) {
    ImGui::TextDisabled("No decoded signals cover this bit.");
  } else {
    ImGui::TextDisabled("Signals covering this bit:");
    for (size_t i = 0; i < overlaps.size(); ++i) {
      if (i > 0) ImGui::SameLine(0.0f, 8.0f);
      if (ImGui::SmallButton(overlaps[i]->name.c_str())) {
        state->cabana.selected_signal_path = overlaps[i]->path;
      }
    }
  }

  if (state->cabana.similar_bits_loading && similar_bit_results_match_selection(*state)) {
    ImGui::Spacing();
    ImGui::TextDisabled("Searching similar bits...");
  } else if (similar_bit_results_match_selection(*state) && !state->cabana.similar_bit_matches.empty()) {
    ImGui::Spacing();
    ImGui::TextDisabled("Similar bits:");
    if (ImGui::BeginTable("##cabana_similar_bits", 5,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
      ImGui::TableSetupColumn("Message", ImGuiTableColumnFlags_WidthStretch, 2.2f);
      ImGui::TableSetupColumn("Bit", ImGuiTableColumnFlags_WidthFixed, 58.0f);
      ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 58.0f);
      ImGui::TableSetupColumn("1s", ImGuiTableColumnFlags_WidthFixed, 52.0f);
      ImGui::TableSetupColumn("Flip", ImGuiTableColumnFlags_WidthFixed, 56.0f);
      ImGui::TableHeadersRow();
      for (const CabanaSimilarBitMatch &match : state->cabana.similar_bit_matches) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const std::string label = match.label + "##" + match.message_root + "_" + std::to_string(match.byte_index) + "_" + std::to_string(match.bit_index);
        if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
          select_cabana_message(session, state, match.message_root);
          state->cabana.has_bit_selection = true;
          state->cabana.selected_bit_byte = match.byte_index;
          state->cabana.selected_bit_index = match.bit_index;
        }
        ImGui::TableNextColumn();
        ImGui::Text("B%d.%d", match.byte_index, match.bit_index);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", match.score);
        ImGui::TableNextColumn();
        ImGui::Text("%.0f%%", 100.0 * match.ones_ratio);
        ImGui::TableNextColumn();
        ImGui::Text("%.0f%%", 100.0 * match.flip_ratio);
      }
      ImGui::EndTable();
    }
  }
  ImGui::Spacing();
}

void draw_can_heatmap(const CanMessageData &message,
                      const std::vector<std::pair<const CabanaSignalSummary *, ImU32>> &highlighted,
                      double tracker_time) {
  const size_t byte_count = can_message_payload_width(message);
  if (message.samples.empty() || byte_count == 0) {
    return;
  }

  app_push_bold_font();
  ImGui::TextUnformatted("History Heatmap");
  app_pop_bold_font();
  ImGui::TextDisabled("aggregated over all frames");
  ImGui::Spacing();

  const size_t row_count = byte_count * 8;
  const float avail_w = ImGui::GetContentRegionAvail().x;
  const float label_w = 42.0f;
  const float row_h = std::clamp(160.0f / std::max<float>(1.0f, static_cast<float>(row_count)), 10.0f, 16.0f);
  const float grid_h = row_h * static_cast<float>(row_count);
  const float grid_w = std::max(120.0f, avail_w - label_w - 8.0f);
  const int columns = std::max(1, std::min<int>(std::min<size_t>(220, message.samples.size()), static_cast<int>(grid_w / 4.0f)));
  const float cell_w = grid_w / static_cast<float>(columns);
  const size_t tracker_index = closest_can_sample_index(message, tracker_time);

  ImGui::InvisibleButton("##cabana_heatmap", ImVec2(label_w + grid_w, grid_h + 4.0f));
  const ImRect rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
  const ImVec2 grid_min(rect.Min.x + label_w, rect.Min.y);
  ImDrawList *draw = ImGui::GetWindowDrawList();
  const ImU32 grid_bg = ImGui::GetColorU32(color_rgb(246, 247, 249));
  const ImU32 low = ImGui::GetColorU32(color_rgb(234, 238, 243));
  const ImU32 high = ImGui::GetColorU32(color_rgb(69, 116, 201));
  const ImU32 border = ImGui::GetColorU32(color_rgb(204, 209, 214));
  draw->AddRectFilled(ImVec2(grid_min.x, rect.Min.y), ImVec2(grid_min.x + grid_w, rect.Min.y + grid_h), grid_bg, 4.0f);

  for (size_t row = 0; row < row_count; ++row) {
    const size_t byte_index = row / 8;
    const int bit_index = 7 - static_cast<int>(row % 8);
    const float y0 = rect.Min.y + static_cast<float>(row) * row_h;
    const float y1 = y0 + row_h;
    if (row % 8 == 0) {
      const std::string label = "B" + std::to_string(byte_index);
      draw->AddText(ImVec2(rect.Min.x, y0 + 1.0f), ImGui::GetColorU32(color_rgb(92, 100, 112)), label.c_str());
      if (row > 0) {
        draw->AddLine(ImVec2(rect.Min.x, y0), ImVec2(grid_min.x + grid_w, y0), border, 1.0f);
      }
    }
    for (int col = 0; col < columns; ++col) {
      const size_t start = (message.samples.size() * static_cast<size_t>(col)) / static_cast<size_t>(columns);
      const size_t end = std::max(start + 1,
                                  (message.samples.size() * static_cast<size_t>(col + 1)) / static_cast<size_t>(columns));
      size_t ones = 0;
      for (size_t i = start; i < std::min(end, message.samples.size()); ++i) {
        ones += can_bit(message.samples[i].data, byte_index, bit_index);
      }
      const float frac = static_cast<float>(ones) / static_cast<float>(std::max<size_t>(1, std::min(end, message.samples.size()) - start));
      ImU32 color = mix_color(low, high, frac);
      for (const auto &[signal, signal_color] : highlighted) {
        if (signal_contains_bit(*signal, byte_index, bit_index)) {
          color = mix_color(color, signal_color, 0.65f);
          break;
        }
      }
      const float x0 = grid_min.x + static_cast<float>(col) * cell_w;
      const float x1 = x0 + cell_w + 0.5f;
      draw->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), color);
    }
  }

  const float tracker_x = grid_min.x + cell_w * ((static_cast<float>(tracker_index) + 0.5f) * static_cast<float>(columns)
                                                  / static_cast<float>(std::max<size_t>(1, message.samples.size())));
  draw->AddLine(ImVec2(tracker_x, rect.Min.y), ImVec2(tracker_x, rect.Min.y + grid_h),
                ImGui::GetColorU32(color_rgb(36, 42, 50, 0.9f)), 2.0f);
  draw->AddRect(ImVec2(grid_min.x, rect.Min.y), ImVec2(grid_min.x + grid_w, rect.Min.y + grid_h), border, 4.0f);
}

void draw_can_frame_view(const CanMessageData &message,
                         AppSession *session,
                         const CabanaMessageSummary &summary,
                         UiState *state,
                         double tracker_time) {
  if (message.samples.empty()) {
    draw_cabana_panel_title("Binary View");
    ImGui::TextDisabled("No raw CAN frames available.");
    return;
  }
  const size_t sample_index = closest_can_sample_index(message, tracker_time);
  const CanFrameSample &sample = message.samples[sample_index];
  const CanFrameSample *prev = sample_index > 0 ? &message.samples[sample_index - 1] : nullptr;
  const size_t byte_count = std::max(can_message_payload_width(message), cabana_signal_byte_count(summary));
  const BinaryMatrixLayout layout = build_binary_matrix_layout(summary, byte_count);
  const std::vector<float> heat = compute_bit_flip_heat(message, byte_count, state->cabana.heatmap_live_mode, sample_index);
  app_push_bold_font();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Binary");
  app_pop_bold_font();
  ImGui::SameLine(0.0f, 10.0f);
  ImGui::TextDisabled("frame %.3fs", sample.mono_time);
  ImGui::SameLine(0.0f, 10.0f);
  ImGui::TextDisabled("tracker %.3fs", tracker_time);
  ImGui::SameLine(0.0f, 10.0f);
  ImGui::TextDisabled("%zu bytes", sample.data.size());
  ImGui::SameLine(0.0f, 10.0f);
  ImGui::TextDisabled("bus %d", summary.bus);
  if (sample.bus_time != 0) {
    ImGui::SameLine(0.0f, 10.0f);
    ImGui::TextDisabled("bus_time %u", sample.bus_time);
  }
  if (prev != nullptr) {
    ImGui::SameLine(0.0f, 10.0f);
    ImGui::TextDisabled("dt %.1f ms", 1000.0 * (sample.mono_time - prev->mono_time));
  }
  ImGui::Spacing();
  draw_payload_bytes(sample.data, prev == nullptr ? nullptr : &prev->data);
  if (layout.overlapping_cells > 0) {
    ImGui::SameLine(0.0f, 12.0f);
    ImGui::TextColored(color_rgb(222, 181, 86), "%zu overlap%s",
                       layout.overlapping_cells, layout.overlapping_cells == 1 ? "" : "s");
  }
  ImGui::Spacing();

  const float footer_reserve = state->cabana.has_bit_selection ? 152.0f : 8.0f;
  const float matrix_height = std::max(140.0f, ImGui::GetContentRegionAvail().y - footer_reserve);
  const float index_w = 28.0f;
  const float hex_w = 36.0f;
  const float bit_w = std::max(32.0f, (ImGui::GetContentRegionAvail().x - index_w - hex_w) / 8.0f);
  const float grid_w = bit_w * 8.0f;
  const float row_h = std::clamp((matrix_height - 4.0f) / std::max(1.0f, static_cast<float>(byte_count)),
                                 28.0f,
                                 42.0f);
  const float total_h = row_h * static_cast<float>(byte_count);
  const ImU32 base_bg = ImGui::GetColorU32(color_rgb(56, 58, 61));
  const ImU32 heat_high = ImGui::GetColorU32(color_rgb(72, 117, 202));
  const ImU32 cell_border = ImGui::GetColorU32(color_rgb(102, 106, 111, 0.72f));
  const ImU32 text_color = ImGui::GetColorU32(color_rgb(219, 223, 228));
  const ImU32 marker_color = ImGui::GetColorU32(color_rgb(175, 180, 186));
  const ImU32 invalid_hatch = ImGui::GetColorU32(color_rgb(126, 131, 139, 0.58f));
  const ImU32 selection_border = ImGui::GetColorU32(color_rgb(232, 236, 241, 0.95f));
  const ImU32 hover_border = ImGui::GetColorU32(color_rgb(232, 236, 241, 0.32f));
  const ImU32 drag_fill = ImGui::GetColorU32(color_rgb(87, 127, 219, 0.18f));
  const ImU32 drag_border = ImGui::GetColorU32(color_rgb(54, 91, 184, 0.95f));
  const bool released_this_frame = ImGui::IsMouseReleased(ImGuiMouseButton_Left);
  bool drag_release_handled = false;
  ImGui::BeginChild("##cabana_binary_grid", ImVec2(0.0f, matrix_height), false);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 mouse = ImGui::GetIO().MousePos;
  const float content_w = index_w + grid_w + hex_w;
  ImGui::InvisibleButton("##cabana_binary_grid_area", ImVec2(content_w, total_h));
  const bool area_hovered = ImGui::IsItemHovered();
  const float scroll_y = ImGui::GetScrollY();

  int hover_byte = -1;
  int hover_bit = -1;
  const bool tracking_hover = area_hovered || (state->cabana.binary_drag_active && ImGui::IsMouseDown(ImGuiMouseButton_Left));
  if (tracking_hover) {
    const float rel_x = mouse.x - (origin.x + index_w);
    const float rel_y = mouse.y - origin.y + scroll_y;
    if (rel_x >= 0.0f && rel_x < grid_w && rel_y >= 0.0f && rel_y < total_h) {
      hover_byte = std::clamp(static_cast<int>(rel_y / row_h), 0, static_cast<int>(byte_count) - 1);
      const int col = std::clamp(static_cast<int>(rel_x / bit_w), 0, 7);
      hover_bit = 7 - col;
    }
  }
  if (state->cabana.binary_drag_active && hover_byte >= 0 && hover_bit >= 0) {
    if (state->cabana.binary_drag_current_byte != hover_byte || state->cabana.binary_drag_current_bit != hover_bit) {
      state->cabana.binary_drag_moved = true;
    }
    state->cabana.binary_drag_current_byte = hover_byte;
    state->cabana.binary_drag_current_bit = hover_bit;
  }

  const int first_row = std::max(0, static_cast<int>(scroll_y / row_h) - 1);
  const int last_row = std::min(static_cast<int>(byte_count),
                                static_cast<int>((scroll_y + matrix_height) / row_h) + 2);
  ImDrawList *draw = ImGui::GetWindowDrawList();
  bool any_bit_hovered = false;

  for (int row = first_row; row < last_row; ++row) {
    const float y0 = origin.y + static_cast<float>(row) * row_h - scroll_y;
    const float y1 = y0 + row_h;
    const bool valid = static_cast<size_t>(row) < sample.data.size();

    const ImRect index_cell(ImVec2(origin.x, y0), ImVec2(origin.x + index_w, y1));
    draw->AddRectFilled(index_cell.Min, index_cell.Max, ImGui::GetColorU32(color_rgb(49, 51, 54)));
    draw->AddRect(index_cell.Min, index_cell.Max, cell_border);
    const std::string label = std::to_string(row);
    const ImVec2 label_size = ImGui::CalcTextSize(label.c_str());
    draw->AddText(ImVec2(index_cell.Min.x + (index_cell.GetWidth() - label_size.x) * 0.5f,
                         index_cell.Min.y + (index_cell.GetHeight() - label_size.y) * 0.5f),
                  ImGui::GetColorU32(color_rgb(84, 92, 103)),
                  label.c_str());

    for (int col = 0; col < 8; ++col) {
      const int bit = 7 - col;
      const size_t cell_index = static_cast<size_t>(row) * 8 + static_cast<size_t>(bit);
      const ImRect cell(ImVec2(origin.x + index_w + static_cast<float>(col) * bit_w, y0),
                        ImVec2(origin.x + index_w + static_cast<float>(col + 1) * bit_w, y1));
      const float heat_alpha = cell_index < heat.size() ? heat[cell_index] : 0.0f;
      const std::vector<int> &cell_signals = layout.cell_signals[cell_index];
      const bool hovered = hover_byte == row && hover_bit == bit;
      const CabanaSignalSummary *clicked_signal = topmost_signal_at_cell(summary, layout, static_cast<size_t>(row), bit);
      const CabanaSignalSummary *resize_signal = resize_signal_at_cell(summary, layout, static_cast<size_t>(row), bit);

      if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        clear_cabana_binary_drag(state);
        state->cabana.binary_drag_active = true;
        state->cabana.binary_drag_press_byte = row;
        state->cabana.binary_drag_press_bit = bit;
        state->cabana.binary_drag_current_byte = row;
        state->cabana.binary_drag_current_bit = bit;
        if (resize_signal != nullptr) {
          const int physical_bit = row * 8 + bit;
          const int opposite = physical_bit == resize_signal->lsb ? resize_signal->msb : resize_signal->lsb;
          state->cabana.binary_drag_resizing = true;
          state->cabana.binary_drag_signal_is_little_endian = resize_signal->is_little_endian;
          state->cabana.binary_drag_signal_path = resize_signal->path;
          state->cabana.binary_drag_anchor_byte = opposite / 8;
          state->cabana.binary_drag_anchor_bit = opposite & 7;
        } else {
          state->cabana.binary_drag_anchor_byte = row;
          state->cabana.binary_drag_anchor_bit = bit;
        }
      }

      draw->AddRectFilled(cell.Min, cell.Max, mix_color(base_bg, heat_high, heat_alpha * 0.55f));

        if (valid && !cell_signals.empty()) {
          for (int signal_index : cell_signals) {
            const CabanaSignalSummary &signal = summary.signals[static_cast<size_t>(signal_index)];
            const bool emphasized = signal_charted(*state, signal.path);
          const bool draw_left = !cell_has_signal(layout, row, bit + 1, signal_index);
          const bool draw_right = !cell_has_signal(layout, row, bit - 1, signal_index);
          const bool draw_top = !cell_has_signal(layout, row - 1, bit, signal_index);
          const bool draw_bottom = !cell_has_signal(layout, row + 1, bit, signal_index);
          ImRect inner = cell;
          inner.Min.x += draw_left ? 3.0f : 1.0f;
          inner.Max.x -= draw_right ? 3.0f : 1.0f;
          inner.Min.y += draw_top ? 2.0f : 1.0f;
          inner.Max.y -= draw_bottom ? 2.0f : 1.0f;
          draw->AddRectFilled(inner.Min, inner.Max,
                              signal_fill_color(static_cast<size_t>(signal_index), heat_alpha, emphasized),
                              2.0f);
          const ImU32 border_color = signal_border_color(static_cast<size_t>(signal_index), emphasized);
          const float thickness = emphasized ? 2.0f : 1.0f;
          if (draw_left) draw->AddLine(ImVec2(inner.Min.x, inner.Min.y), ImVec2(inner.Min.x, inner.Max.y), border_color, thickness);
          if (draw_right) draw->AddLine(ImVec2(inner.Max.x, inner.Min.y), ImVec2(inner.Max.x, inner.Max.y), border_color, thickness);
          if (draw_top) draw->AddLine(ImVec2(inner.Min.x, inner.Min.y), ImVec2(inner.Max.x, inner.Min.y), border_color, thickness);
          if (draw_bottom) draw->AddLine(ImVec2(inner.Min.x, inner.Max.y), ImVec2(inner.Max.x, inner.Max.y), border_color, thickness);
          }
          if (cell_signals.size() > 1) {
            const ImVec2 a(cell.Max.x - 10.0f, cell.Min.y + 2.0f);
            const ImVec2 b(cell.Max.x - 2.0f, cell.Min.y + 2.0f);
            const ImVec2 c(cell.Max.x - 2.0f, cell.Min.y + 10.0f);
            draw->AddTriangleFilled(a, b, c, ImGui::GetColorU32(color_rgb(223, 181, 87, 0.82f)));
          }
        } else if (!valid) {
        draw->AddRectFilled(cell.Min, cell.Max, ImGui::GetColorU32(color_rgb(47, 49, 52)));
        draw_cell_hatching(draw, cell, invalid_hatch, 7.0f);
      }

      if (cabana_drag_selection_contains_bit(*state, static_cast<size_t>(row), bit)) {
        draw->AddRectFilled(cell.Min, cell.Max, drag_fill);
        const bool draw_left = !cabana_drag_selection_has_neighbor(*state, row, bit + 1);
        const bool draw_right = !cabana_drag_selection_has_neighbor(*state, row, bit - 1);
        const bool draw_top = !cabana_drag_selection_has_neighbor(*state, row - 1, bit);
        const bool draw_bottom = !cabana_drag_selection_has_neighbor(*state, row + 1, bit);
        if (draw_left) draw->AddLine(ImVec2(cell.Min.x, cell.Min.y), ImVec2(cell.Min.x, cell.Max.y), drag_border, 2.0f);
        if (draw_right) draw->AddLine(ImVec2(cell.Max.x, cell.Min.y), ImVec2(cell.Max.x, cell.Max.y), drag_border, 2.0f);
        if (draw_top) draw->AddLine(ImVec2(cell.Min.x, cell.Min.y), ImVec2(cell.Max.x, cell.Min.y), drag_border, 2.0f);
        if (draw_bottom) draw->AddLine(ImVec2(cell.Min.x, cell.Max.y), ImVec2(cell.Max.x, cell.Max.y), drag_border, 2.0f);
      }

      draw->AddRect(cell.Min, cell.Max, cell_border);
      if (valid) {
        app_push_mono_font();
        const char bit_text[2] = {static_cast<char>(can_bit(sample.data, static_cast<size_t>(row), bit) ? '1' : '0'), '\0'};
        const ImVec2 text_size = ImGui::CalcTextSize(bit_text);
        draw->AddText(ImGui::GetFont(),
                      ImGui::GetFontSize(),
                      ImVec2(cell.Min.x + (cell.GetWidth() - text_size.x) * 0.5f,
                             cell.Min.y + (cell.GetHeight() - text_size.y) * 0.5f - 1.0f),
                      text_color,
                      bit_text);
        app_pop_mono_font();
      }
      if (layout.is_msb[cell_index] || layout.is_lsb[cell_index]) {
        const char marker[2] = {layout.is_msb[cell_index] ? 'M' : 'L', '\0'};
        draw->AddText(ImVec2(cell.Max.x - 11.0f, cell.Max.y - 14.0f), marker_color, marker);
      }
      if (cabana_bit_selected(*state, static_cast<size_t>(row), bit)) {
        draw->AddRect(cell.Min, cell.Max, selection_border, 0.0f, 0, 2.0f);
      } else if (hovered) {
        draw->AddRect(cell.Min, cell.Max, hover_border, 0.0f, 0, 1.0f);
      }
      if (hovered) {
        any_bit_hovered = true;
        if (clicked_signal != nullptr && !state->cabana.binary_drag_active) {
          ImGui::SetTooltip("%s\nstart_bit %d  size %d  lsb %d  msb %d",
                            clicked_signal->name.c_str(),
                            clicked_signal->start_bit,
                            clicked_signal->size,
                            clicked_signal->lsb,
                            clicked_signal->msb);
        }
      }
    }

    float byte_heat = 0.0f;
    for (int bit = 0; bit < 8; ++bit) {
      byte_heat = std::max(byte_heat, heat[static_cast<size_t>(row) * 8 + static_cast<size_t>(bit)]);
    }
    const ImRect hex_cell(ImVec2(origin.x + index_w + grid_w, y0),
                          ImVec2(origin.x + index_w + grid_w + hex_w, y1));
    draw->AddRectFilled(hex_cell.Min, hex_cell.Max, mix_color(base_bg, heat_high, byte_heat * 0.5f));
    draw->AddRect(hex_cell.Min, hex_cell.Max, cell_border);
    if (valid) {
      app_push_mono_font();
      char hex[4];
      std::snprintf(hex, sizeof(hex), "%02X", static_cast<unsigned char>(sample.data[static_cast<size_t>(row)]));
      const ImVec2 text_size = ImGui::CalcTextSize(hex);
      draw->AddText(ImGui::GetFont(),
                    ImGui::GetFontSize(),
                    ImVec2(hex_cell.Min.x + (hex_cell.GetWidth() - text_size.x) * 0.5f,
                           hex_cell.Min.y + (hex_cell.GetHeight() - text_size.y) * 0.5f - 1.0f),
                    text_color,
                    hex);
      app_pop_mono_font();
    } else {
      draw_cell_hatching(draw, hex_cell, invalid_hatch, 7.0f);
    }
  }

  ImGui::EndChild();
  if (released_this_frame && state->cabana.binary_drag_active && !drag_release_handled) {
    if (state->cabana.binary_drag_current_byte >= 0 && state->cabana.binary_drag_current_bit >= 0) {
      const CabanaSignalSummary *release_signal =
          topmost_signal_at_cell(summary,
                                 layout,
                                 static_cast<size_t>(state->cabana.binary_drag_current_byte),
                                 state->cabana.binary_drag_current_bit);
      queue_binary_drag_apply(session,
                              summary,
                              state,
                              state->cabana.binary_drag_current_byte,
                              state->cabana.binary_drag_current_bit,
                              release_signal);
    }
    clear_cabana_binary_drag(state);
  } else if (!any_bit_hovered && released_this_frame && state->cabana.binary_drag_active) {
    clear_cabana_binary_drag(state);
  }

  ImGui::Spacing();
  draw_bit_selection_panel(session, summary, state);
}

bool message_has_overlaps(const CabanaMessageSummary &message, const CanMessageData *message_data) {
  const size_t byte_count = std::max(message_data == nullptr ? 0 : can_message_payload_width(*message_data),
                                     cabana_signal_byte_count(message));
  if (byte_count == 0) {
    return false;
  }
  return build_binary_matrix_layout(message, byte_count).overlapping_cells > 0;
}

void draw_detail_toolbar(AppSession *session,
                         UiState *state,
                         const CabanaMessageSummary &message,
                         const CanMessageData *message_data) {
  const std::string meta = message.service + " bus " + std::to_string(message.bus)
                         + (message.has_address ? "  " + format_can_address(message.address) : std::string());
  const std::string dbc_text = session->route_data.dbc_name.empty() ? "DBC: Auto / none"
                                                                    : "DBC: " + session->route_data.dbc_name;
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 4.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 6.0f));
  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleColor(ImGuiCol_Border, cabana_border_color());
  const bool compact = ImGui::GetContentRegionAvail().x < 760.0f;
  ImGui::BeginChild("##cabana_detail_toolbar", ImVec2(0.0f, compact ? 68.0f : 40.0f), true);
  app_push_bold_font();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(message.name.c_str());
  app_pop_bold_font();
  ImGui::SameLine(0.0f, 8.0f);
  ImGui::TextDisabled("%s", meta.c_str());
  if (message.frequency_hz > 0.0) {
    ImGui::SameLine(0.0f, 10.0f);
    ImGui::TextDisabled("%.1f Hz", message.frequency_hz);
  }
  if (message_data != nullptr) {
    ImGui::SameLine(0.0f, 10.0f);
    ImGui::TextDisabled("%zu frames", message_data->samples.size());
  }

  if (compact) {
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4.0f);
  } else {
    ImGui::SameLine(std::max(340.0f, ImGui::GetWindowContentRegionMax().x * 0.46f));
  }
  ImGui::TextUnformatted("Heatmap:");
  ImGui::SameLine(0.0f, 6.0f);
  if (ImGui::RadioButton("Live", state->cabana.heatmap_live_mode)) {
    state->cabana.heatmap_live_mode = true;
  }
  ImGui::SameLine(0.0f, 8.0f);
  if (ImGui::RadioButton("All", !state->cabana.heatmap_live_mode)) {
    state->cabana.heatmap_live_mode = false;
  }
  ImGui::SameLine(0.0f, 14.0f);
  draw_cabana_toolbar_button("Edit DBC...", true, [&]() {
    state->dbc_editor.open = true;
    state->dbc_editor.loaded = false;
  });
  ImGui::SameLine(0.0f, 6.0f);
  draw_cabana_toolbar_button("Export Raw CSV", message_data != nullptr, [&]() {
    fs::path output_path;
    std::string error;
    if (export_raw_can_csv(*session, message, &error, &output_path)) {
      state->status_text = "Exported raw CSV " + output_path.filename().string();
    } else {
      state->status_text = error;
    }
  });
  ImGui::SameLine(0.0f, 6.0f);
  draw_cabana_toolbar_button("Export Decoded CSV", message_data != nullptr, [&]() {
    fs::path output_path;
    std::string error;
    if (export_decoded_can_csv(*session, message, &error, &output_path)) {
      state->status_text = "Exported decoded CSV " + output_path.filename().string();
    } else {
      state->status_text = error;
    }
  });
  if (!compact) {
    const float right_w = ImGui::CalcTextSize(dbc_text.c_str()).x + 10.0f;
    ImGui::SameLine(std::max(0.0f, ImGui::GetWindowContentRegionMax().x - right_w));
  } else {
    ImGui::SameLine(0.0f, 12.0f);
  }
  ImGui::TextDisabled("%s", dbc_text.c_str());
  ImGui::EndChild();
  ImGui::PopStyleColor(2);
  ImGui::PopStyleVar(2);
}

void draw_messages_panel(AppSession *session, UiState *state) {
  const std::string name_filter = trim_copy(state->cabana.message_filter.data());
  const std::string bus_filter = trim_copy(state->cabana.message_bus_filter.data());
  const std::string addr_filter = trim_copy(state->cabana.message_addr_filter.data());
  const std::string node_filter = trim_copy(state->cabana.message_node_filter.data());
  const std::string freq_filter = trim_copy(state->cabana.message_freq_filter.data());
  const std::string count_filter = trim_copy(state->cabana.message_count_filter.data());
  const std::string bytes_filter = trim_copy(state->cabana.message_bytes_filter.data());

  std::vector<int> filtered_indices;
  filtered_indices.reserve(session->cabana_messages.size());
  size_t filtered_signal_count = 0;
  size_t filtered_dbc_count = 0;
  for (int i = 0; i < static_cast<int>(session->cabana_messages.size()); ++i) {
    const CabanaMessageSummary &message = session->cabana_messages[static_cast<size_t>(i)];
    const CanMessageData *message_data = find_message_data(*session, message);
    if (state->cabana.suppress_defined_signals && !message.signals.empty()) {
      continue;
    }
    if (!cabana_message_matches_filters(message, name_filter, bus_filter, addr_filter, node_filter,
                                        freq_filter, count_filter, bytes_filter, message_data)) {
      continue;
    }
    filtered_indices.push_back(i);
    filtered_signal_count += message.signals.size();
    if (message.dbc_size > 0 || !message.node.empty() || !message.signals.empty()) {
      ++filtered_dbc_count;
    }
  }

  char title[160];
  std::snprintf(title, sizeof(title), "%zu Messages (%zu DBC Messages, %zu Signals)",
                filtered_indices.size(), filtered_dbc_count, filtered_signal_count);

  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 4.0f));
  ImGui::BeginChild("##cabana_messages_header", ImVec2(0.0f, 34.0f), false, ImGuiWindowFlags_NoScrollbar);
  ImGui::TextUnformatted(title);
  const float clear_w = 46.0f;
  const float checkbox_w = 126.0f;
  const float target_x = std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - clear_w - checkbox_w - 16.0f);
  ImGui::SameLine(target_x);
  if (ImGui::SmallButton("Clear")) {
    state->cabana.message_filter[0] = '\0';
    state->cabana.message_bus_filter[0] = '\0';
    state->cabana.message_addr_filter[0] = '\0';
    state->cabana.message_node_filter[0] = '\0';
    state->cabana.message_freq_filter[0] = '\0';
    state->cabana.message_count_filter[0] = '\0';
    state->cabana.message_bytes_filter[0] = '\0';
  }
  ImGui::SameLine(0.0f, 8.0f);
  ImGui::Checkbox("Suppress Signals", &state->cabana.suppress_defined_signals);
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  ImGui::Spacing();

  if (session->cabana_messages.empty()) {
    ImGui::TextDisabled("No CAN messages in this route.");
    return;
  }
  if (filtered_indices.empty()) {
    ImGui::TextDisabled("No messages match the current filters.");
    return;
  }

  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 2.0f));
  if (ImGui::BeginTable("##cabana_messages", 7,
                        ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollX |
                          ImGuiTableFlags_ScrollY |
                          ImGuiTableFlags_Borders |
                          ImGuiTableFlags_Resizable |
                          ImGuiTableFlags_SizingStretchProp,
                        ImGui::GetContentRegionAvail())) {
    ImGui::TableSetupScrollFreeze(0, 2);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoHide, 1.8f);
    ImGui::TableSetupColumn("Bus", ImGuiTableColumnFlags_WidthFixed, 46.0f);
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 70.0f);
    ImGui::TableSetupColumn("Node", ImGuiTableColumnFlags_WidthFixed, 84.0f);
    ImGui::TableSetupColumn("Hz", ImGuiTableColumnFlags_WidthFixed, 54.0f);
    ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 64.0f);
    ImGui::TableSetupColumn("Bytes", ImGuiTableColumnFlags_WidthStretch, 1.2f);
    ImGui::TableHeadersRow();

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##cabana_filter_name", "Filter", state->cabana.message_filter.data(),
                             state->cabana.message_filter.size());
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##cabana_filter_bus", "Bus", state->cabana.message_bus_filter.data(),
                             state->cabana.message_bus_filter.size());
    ImGui::TableSetColumnIndex(2);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##cabana_filter_addr", "Addr", state->cabana.message_addr_filter.data(),
                             state->cabana.message_addr_filter.size());
    ImGui::TableSetColumnIndex(3);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##cabana_filter_node", "Node", state->cabana.message_node_filter.data(),
                             state->cabana.message_node_filter.size());
    ImGui::TableSetColumnIndex(4);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##cabana_filter_freq", "Hz", state->cabana.message_freq_filter.data(),
                             state->cabana.message_freq_filter.size());
    ImGui::TableSetColumnIndex(5);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##cabana_filter_count", "Count", state->cabana.message_count_filter.data(),
                             state->cabana.message_count_filter.size());
    ImGui::TableSetColumnIndex(6);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##cabana_filter_bytes", "Bytes", state->cabana.message_bytes_filter.data(),
                             state->cabana.message_bytes_filter.size());

    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(filtered_indices.size()));
    while (clipper.Step()) {
      for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
        const CabanaMessageSummary &message = session->cabana_messages[static_cast<size_t>(filtered_indices[static_cast<size_t>(row)])];
        const bool selected = state->cabana.selected_message_root == message.root_path;
        const std::string address = message.has_address ? format_can_address(message.address) : std::string("--");

        ImGui::TableNextRow(0, 22.0f);
        ImGui::TableSetColumnIndex(0);
        if (ImGui::Selectable((message.name + "##" + message.root_path).c_str(), selected,
                              ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
          select_cabana_message(session, state, message.root_path);
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("%s", message.root_path.c_str());
        }

        ImGui::TableSetColumnIndex(1);
        const std::string bus_label = message.service == "sendcan"
                                    ? "S" + std::to_string(message.bus)
                                    : std::to_string(message.bus);
        ImGui::TextUnformatted(bus_label.c_str());
        ImGui::TableSetColumnIndex(2);
        ImGui::TextUnformatted(address.c_str());
        ImGui::TableSetColumnIndex(3);
        if (message.node.empty()) ImGui::TextDisabled("-");
        else ImGui::TextUnformatted(message.node.c_str());
        ImGui::TableSetColumnIndex(4);
        if (message.frequency_hz >= 0.95) ImGui::Text("%.0f", message.frequency_hz);
        else if (message.frequency_hz > 0.0) ImGui::Text("%.2f", message.frequency_hz);
        else ImGui::TextDisabled("-");
        ImGui::TableSetColumnIndex(5);
        ImGui::Text("%zu", message.sample_count);
        ImGui::TableSetColumnIndex(6);
        const CanMessageData *message_data = find_message_data(*session, message);
        if (message_data != nullptr && !message_data->samples.empty()) {
          const size_t current_index = state->has_tracker_time
                                     ? closest_can_sample_index(*message_data, state->tracker_time)
                                     : (message_data->samples.size() - 1);
          const CanFrameSample &last = message_data->samples[current_index];
          const CanFrameSample *prev = current_index > 0 ? &message_data->samples[current_index - 1] : nullptr;
          draw_payload_preview_boxes(("##msg_bytes_" + message.root_path).c_str(),
                                     last.data,
                                     prev == nullptr ? nullptr : &prev->data,
                                     std::max(72.0f, ImGui::GetColumnWidth() - 10.0f));
        } else {
          ImGui::TextDisabled("-");
        }
      }
    }
    ImGui::EndTable();
  }
  ImGui::PopStyleVar();
}

void draw_logs_toolbar(const AppSession &session,
                       UiState *state,
                       const CabanaMessageSummary &message,
                       bool can_show_signal_mode,
                       bool show_signal_mode) {
  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 4.0f));
  const bool compact = ImGui::GetContentRegionAvail().x < 560.0f;
  ImGui::BeginChild("##cabana_logs_toolbar", ImVec2(0.0f, compact ? 56.0f : 34.0f), false, ImGuiWindowFlags_NoScrollbar);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Display:");
  ImGui::SameLine(0.0f, 6.0f);
  ImGui::BeginDisabled(!can_show_signal_mode);
  if (ImGui::RadioButton("Signal", show_signal_mode)) {
    state->cabana.logs_hex_mode = false;
  }
  ImGui::EndDisabled();
  ImGui::SameLine(0.0f, 8.0f);
  if (ImGui::RadioButton("Hex", !show_signal_mode)) {
    state->cabana.logs_hex_mode = true;
  }

  const std::string signal_path(active_signal_path(*state));
  if (can_show_signal_mode) {
    const size_t slash = signal_path.find_last_of('/');
    const std::string signal_name = slash == std::string::npos ? signal_path : signal_path.substr(slash + 1);
    ImGui::SameLine(0.0f, 12.0f);
    ImGui::TextDisabled("%s", signal_name.c_str());
    if (show_signal_mode) {
      static constexpr const char *kOps[] = {">", "=", "!=", "<"};
      if (compact) {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4.0f);
      } else {
        ImGui::SameLine(0.0f, 10.0f);
      }
      ImGui::SetNextItemWidth(54.0f);
      ImGui::Combo("##cabana_logs_cmp", &state->cabana.logs_filter_compare, kOps, IM_ARRAYSIZE(kOps));
      ImGui::SameLine(0.0f, 6.0f);
      ImGui::SetNextItemWidth(96.0f);
      ImGui::InputTextWithHint("##cabana_logs_value", "value", state->cabana.logs_filter_value.data(),
                               state->cabana.logs_filter_value.size());
    }
  }

  const float export_w = 76.0f;
  ImGui::SameLine(std::max(0.0f, ImGui::GetWindowContentRegionMax().x - export_w));
  if (ImGui::SmallButton("Export")) {
    fs::path output_path;
    std::string error;
    const bool ok = show_signal_mode ? export_decoded_can_csv(session, message, &error, &output_path)
                                     : export_raw_can_csv(session, message, &error, &output_path);
    state->status_text = ok ? ("Exported " + output_path.filename().string()) : error;
  }
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  ImGui::Spacing();
}

void draw_message_history(const AppSession &session, UiState *state, const CabanaMessageSummary &message) {
  const CanMessageData *message_data = find_message_data(session, message);
  if (message_data == nullptr || message_data->samples.empty()) {
    ImGui::TextDisabled("No frame history available.");
    return;
  }

  const std::string signal_path(active_signal_path(*state));
  const RouteSeries *series = signal_path.empty() ? nullptr : app_find_route_series(session, signal_path);
  const auto format_it = signal_path.empty() ? session.route_data.series_formats.end() : session.route_data.series_formats.find(signal_path);
  const auto enum_it = signal_path.empty() ? session.route_data.enum_info.end() : session.route_data.enum_info.find(signal_path);
  const size_t current_index = closest_can_sample_index(*message_data, state->tracker_time);
  const bool can_show_signal_mode = series != nullptr;
  const bool show_signal_mode = can_show_signal_mode && !state->cabana.logs_hex_mode;

  draw_logs_toolbar(session, state, message, can_show_signal_mode, show_signal_mode);

  const bool have_filter = show_signal_mode && state->cabana.logs_filter_value[0] != '\0';
  const double filter_value = have_filter ? std::strtod(state->cabana.logs_filter_value.data(), nullptr) : 0.0;
  auto passes_filter = [&](double value) {
    if (!have_filter) return true;
    switch (state->cabana.logs_filter_compare) {
      case 0: return value > filter_value;
      case 1: return value == filter_value;
      case 2: return value != filter_value;
      case 3: return value < filter_value;
      default: return true;
    }
  };

  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(5.0f, 3.0f));
  const int columns = (!show_signal_mode && series != nullptr) ? 4 : 3;
  if (ImGui::BeginTable("##cabana_history", columns,
                        ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY |
                          ImGuiTableFlags_SizingStretchProp |
                          ImGuiTableFlags_BordersInnerV |
                          ImGuiTableFlags_BordersOuterH |
                          ImGuiTableFlags_NoPadOuterX,
                        ImGui::GetContentRegionAvail())) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 96.0f);
    ImGui::TableSetupColumn("dt", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    if (show_signal_mode) {
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    } else {
      ImGui::TableSetupColumn("Data", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    }
    if (!show_signal_mode && series != nullptr) {
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 108.0f);
    }
    ImGui::TableHeadersRow();

    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(message_data->samples.size()));
    while (clipper.Step()) {
      for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
        const CanFrameSample &sample = message_data->samples[static_cast<size_t>(i)];
        const CanFrameSample *prev = i > 0 ? &message_data->samples[static_cast<size_t>(i - 1)] : nullptr;
        std::optional<double> value;
        if (series != nullptr) {
          value = app_sample_xy_value_at_time(series->times, series->values, false, sample.mono_time);
          if (show_signal_mode && (!value.has_value() || !passes_filter(*value))) {
            continue;
          }
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const bool selected = static_cast<size_t>(i) == current_index;
        char label[32];
        std::snprintf(label, sizeof(label), "%.3f##frame_%d", sample.mono_time, i);
        if (ImGui::Selectable(label, selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
          state->tracker_time = sample.mono_time;
          state->has_tracker_time = true;
        }

        ImGui::TableNextColumn();
        if (prev != nullptr) {
          ImGui::Text("%.1fms", 1000.0 * (sample.mono_time - prev->mono_time));
        } else {
          ImGui::TextDisabled("--");
        }

        ImGui::TableNextColumn();
        if (show_signal_mode) {
          if (value.has_value()) {
            if (format_it != session.route_data.series_formats.end()) {
              ImGui::TextUnformatted(format_display_value(*value,
                                                          format_it->second,
                                                          enum_it == session.route_data.enum_info.end() ? nullptr : &enum_it->second).c_str());
            } else {
              ImGui::Text("%.4f", *value);
            }
          } else {
            ImGui::TextDisabled("--");
          }
        } else {
          draw_payload_bytes(sample.data, prev == nullptr ? nullptr : &prev->data);
        }

        if (!show_signal_mode && series != nullptr) {
          ImGui::TableNextColumn();
          if (value.has_value()) {
            if (format_it != session.route_data.series_formats.end()) {
              ImGui::TextUnformatted(format_display_value(*value,
                                                          format_it->second,
                                                          enum_it == session.route_data.enum_info.end() ? nullptr : &enum_it->second).c_str());
            } else {
              ImGui::Text("%.4f", *value);
            }
          } else {
            ImGui::TextDisabled("--");
          }
        }
      }
    }
    ImGui::EndTable();
  }
  ImGui::PopStyleVar();
}

float preferred_binary_view_height(const CanMessageData &message, const CabanaMessageSummary &summary, const UiState &state) {
  const size_t byte_count = std::max(can_message_payload_width(message), cabana_signal_byte_count(summary));
  const float row_h = 34.0f;
  const float header_h = 88.0f;
  const float footer_h = state.cabana.has_bit_selection ? 148.0f : 8.0f;
  return header_h + row_h * static_cast<float>(std::max<size_t>(1, byte_count)) + footer_h;
}

void draw_detail_panel(AppSession *session, UiState *state, const CabanaMessageSummary &message) {
  const CanMessageData *message_data = find_message_data(*session, message);
  draw_cabana_message_tabs(session, state);
  draw_detail_toolbar(session, state, message, message_data);
  std::vector<std::string> warnings;
  if (message_has_overlaps(message, message_data)) {
    warnings.push_back("One or more decoded signals overlap in the binary view.");
  }
  if (message.dbc_size > 0 && message_data != nullptr && !message_data->samples.empty()
      && static_cast<int>(message_data->samples.back().data.size()) != message.dbc_size) {
    warnings.push_back("Message size does not match the active DBC definition.");
  }
  draw_cabana_warning_banner(warnings);
  ImGui::Spacing();

  const float bottom_tabs_h = 30.0f;
  const float detail_content_h = std::max(0.0f, ImGui::GetContentRegionAvail().y - bottom_tabs_h);
  const float split_span = std::max(1.0f, detail_content_h - kSplitterThickness);
  const float min_top_frac = kMinTopHeight / split_span;
  const float min_bottom_frac = kMinBottomHeight / split_span;
  state->cabana.layout_center_top_frac = std::clamp(state->cabana.layout_center_top_frac,
                                                    min_top_frac,
                                                    std::max(min_top_frac, 1.0f - min_bottom_frac));
  float top_height = std::floor(split_span * state->cabana.layout_center_top_frac);
  top_height = std::clamp(top_height, kMinTopHeight, std::max(kMinTopHeight, detail_content_h - kMinBottomHeight - kSplitterThickness));
  if (state->cabana.detail_tab == 0 && state->cabana.detail_top_auto_fit && message_data != nullptr) {
    top_height = std::clamp(preferred_binary_view_height(*message_data, message, *state),
                            kMinTopHeight,
                            std::max(kMinTopHeight, detail_content_h - kMinBottomHeight - kSplitterThickness));
    state->cabana.layout_center_top_frac = top_height / split_span;
  }

  ImGui::BeginChild("##cabana_detail_content", ImVec2(0.0f, detail_content_h), false);
  if (state->cabana.detail_tab == 0) {
    ImGui::BeginChild("##cabana_msg_top", ImVec2(0.0f, top_height), false);
    if (message_data != nullptr) {
      draw_can_frame_view(*message_data, session, message, state, state->tracker_time);
    } else {
      draw_empty_panel("Binary View", "No raw CAN frames available for this message.");
    }
    ImGui::EndChild();
    if (draw_horizontal_splitter("##cabana_detail_splitter",
                                 ImGui::GetContentRegionAvail().x,
                                 kMinTopHeight,
                                 std::max(kMinTopHeight, ImGui::GetContentRegionAvail().y - kMinBottomHeight),
                                 &top_height)) {
      state->cabana.detail_top_auto_fit = false;
      state->cabana.layout_center_top_frac = std::clamp(top_height / split_span,
                                                        min_top_frac,
                                                        std::max(min_top_frac, 1.0f - min_bottom_frac));
    }
    ImGui::BeginChild("##cabana_signals_bottom", ImVec2(0.0f, 0.0f), false);
    draw_signal_panel(session, state, message);
    ImGui::EndChild();
  } else {
    if (message_data != nullptr) {
      draw_can_heatmap(*message_data, highlighted_signals(message, *state), state->tracker_time);
      ImGui::Spacing();
    }
    draw_message_history(*session, state, message);
  }
  ImGui::EndChild();
  draw_cabana_detail_tab_strip(state);
}

void draw_video_panel(AppSession *session, UiState *state, float height) {
  const auto &views = camera_view_specs();
  std::vector<const CameraViewSpec *> available_views;
  available_views.reserve(views.size());
  for (const CameraViewSpec &spec : views) {
    if (!(session->route_data.*(spec.route_member)).entries.empty()) {
      available_views.push_back(&spec);
    }
  }

  draw_cabana_panel_title("Video");

  if (available_views.empty()) {
    ImGui::BeginChild("##cabana_video_empty", ImVec2(0.0f, height), false);
    ImGui::TextDisabled("No camera streams available.");
    ImGui::EndChild();
    return;
  }

  if (std::none_of(available_views.begin(), available_views.end(), [&](const CameraViewSpec *spec) {
        return spec->view == state->cabana.camera_view;
      })) {
    state->cabana.camera_view = available_views.front()->view;
  }

  auto short_label = [](const CameraViewSpec &spec) {
    switch (spec.view) {
      case CameraViewKind::Road: return "Road";
      case CameraViewKind::Driver: return "Driver";
      case CameraViewKind::WideRoad: return "Wide";
      case CameraViewKind::QRoad: return "qRoad";
    }
    return "Cam";
  };

  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 4.0f));
  ImGui::BeginChild("##cabana_video_header", ImVec2(0.0f, 32.0f), false, ImGuiWindowFlags_NoScrollbar);
  app_push_bold_font();
  ImGui::TextUnformatted("Video");
  app_pop_bold_font();
  ImGui::SameLine(0.0f, 10.0f);
  for (size_t i = 0; i < available_views.size(); ++i) {
    const CameraViewSpec &spec = *available_views[i];
    if (i > 0) ImGui::SameLine(0.0f, 4.0f);
    const float width = spec.view == CameraViewKind::Driver ? 66.0f : 58.0f;
    if (draw_cabana_bottom_tab(("##video_" + std::to_string(i)).c_str(),
                               short_label(spec),
                               state->cabana.camera_view == spec.view,
                               width)) {
      state->cabana.camera_view = spec.view;
    }
  }
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();

  const CameraViewSpec &active_spec = camera_view_spec(state->cabana.camera_view);
  CameraFeedView *feed = session->pane_camera_feeds[static_cast<size_t>(active_spec.view)].get();
  if (feed != nullptr && state->has_tracker_time) {
    feed->update(state->tracker_time);
  }
  if (feed == nullptr) {
    ImGui::TextDisabled("Camera unavailable");
    return;
  }

  static constexpr std::array<double, 11> kPlaybackRates = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0};
  const float controls_h = 68.0f;
  feed->drawSized(ImVec2(ImGui::GetContentRegionAvail().x, std::max(0.0f, height - controls_h)),
                  session->async_route_loading,
                  true);

  const double current = state->has_tracker_time ? state->tracker_time : session->route_data.x_min;
  const double total = session->route_data.has_time_range ? session->route_data.x_max : current;
  double slider_value = current;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, cabana_panel_alt_bg());
  ImGui::PushStyleColor(ImGuiCol_Border, cabana_border_color());
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 5.0f));
  ImGui::BeginChild("##cabana_video_controls", ImVec2(0.0f, controls_h), true, ImGuiWindowFlags_NoScrollbar);
  const float button_w = 24.0f;
  if (ImGui::Button("|<", ImVec2(button_w, 0.0f))) {
    step_tracker(state, -1.0);
  }
  ImGui::SameLine(0.0f, 4.0f);
  if (ImGui::Button(state->playback_playing ? "||" : ">", ImVec2(button_w, 0.0f))) {
    state->playback_playing = !state->playback_playing;
  }
  ImGui::SameLine(0.0f, 4.0f);
  if (ImGui::Button(">|", ImVec2(button_w, 0.0f))) {
    step_tracker(state, 1.0);
  }
  ImGui::SameLine(0.0f, 8.0f);
  ImGui::TextDisabled("%s / %s", format_cabana_time(current).c_str(), format_cabana_time(total).c_str());
  ImGui::SameLine(0.0f, 12.0f);
  ImGui::Checkbox("Loop", &state->playback_loop);
  ImGui::SameLine(0.0f, 12.0f);
  char rate_label[16];
  std::snprintf(rate_label, sizeof(rate_label), "%.2gx", state->playback_rate);
  ImGui::SetNextItemWidth(72.0f);
  if (ImGui::BeginCombo("##cabana_speed", rate_label)) {
    for (double rate : kPlaybackRates) {
      char option[16];
      std::snprintf(option, sizeof(option), "%.2gx", rate);
      const bool selected = std::abs(state->playback_rate - rate) < 1.0e-9;
      if (ImGui::Selectable(option, selected)) {
        state->playback_rate = rate;
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  const ImVec2 bar_min = ImGui::GetCursorScreenPos();
  const float bar_w = ImGui::GetContentRegionAvail().x;
  const float bar_h = 4.0f;
  ImGui::Dummy(ImVec2(bar_w, bar_h));
  const ImRect bar_rect(bar_min, ImVec2(bar_min.x + bar_w, bar_min.y + bar_h));
  ImDrawList *draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(bar_rect.Min, bar_rect.Max, ImGui::GetColorU32(color_rgb(74, 77, 80)));
  if (session->route_data.has_time_range && session->route_data.x_max > session->route_data.x_min) {
    const double route_span = session->route_data.x_max - session->route_data.x_min;
    for (const TimelineEntry &entry : session->route_data.timeline) {
      const float x0 = static_cast<float>((entry.start_time - session->route_data.x_min) / route_span);
      const float x1 = static_cast<float>((entry.end_time - session->route_data.x_min) / route_span);
      const float left = std::clamp(x0, 0.0f, 1.0f);
      const float right = std::clamp(x1, 0.0f, 1.0f);
      if (right <= left) continue;
      draw->AddRectFilled(ImVec2(bar_rect.Min.x + left * bar_rect.GetWidth(), bar_rect.Min.y),
                          ImVec2(bar_rect.Min.x + right * bar_rect.GetWidth(), bar_rect.Max.y),
                          timeline_entry_color(entry.type));
    }
    const float tracker_x = static_cast<float>((current - session->route_data.x_min) / route_span);
    const float px = bar_rect.Min.x + std::clamp(tracker_x, 0.0f, 1.0f) * bar_rect.GetWidth();
    draw->AddLine(ImVec2(px, bar_rect.Min.y - 1.0f), ImVec2(px, bar_rect.Max.y + 1.0f),
                  ImGui::GetColorU32(color_rgb(232, 232, 232)), 1.5f);
  }
  ImGui::Dummy(ImVec2(0.0f, 3.0f));
  if (session->route_data.has_time_range) {
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::SliderScalar("##cabana_video_slider",
                            ImGuiDataType_Double,
                            &slider_value,
                            &session->route_data.x_min,
                            &session->route_data.x_max,
                            "")) {
      state->tracker_time = slider_value;
      state->has_tracker_time = true;
    }
  }
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(2);
}

}  // namespace

void rebuild_cabana_messages(AppSession *session) {
  std::vector<CabanaMessageSummary> messages;
  const std::optional<dbc::Database> db = load_active_dbc(*session);

  messages.reserve(session->route_data.can_messages.size());
  for (const CanMessageData &message_data : session->route_data.can_messages) {
    const dbc::Message *dbc_message = db.has_value() ? db->message(message_data.id.address) : nullptr;
    CabanaMessageSummary message{
      .root_path = can_message_key(message_data.id.service, message_data.id.bus, message_data.id.address),
      .service = can_service_name(message_data.id.service),
      .name = dbc_message != nullptr ? dbc_message->name : format_can_address(message_data.id.address),
      .node = dbc_message != nullptr ? dbc_message->transmitter : std::string(),
      .bus = static_cast<int>(message_data.id.bus),
      .address = message_data.id.address,
      .dbc_size = dbc_message != nullptr ? static_cast<int>(dbc_message->size) : -1,
      .has_address = true,
      .sample_count = message_data.samples.size(),
    };
    if (dbc_message != nullptr) {
      const std::string base_path = "/" + message.service + "/" + std::to_string(message.bus) + "/" + dbc_message->name + "/";
      message.signals.reserve(dbc_message->signals.size());
      for (const dbc::Signal &dbc_signal : dbc_message->signals) {
        const std::string path = base_path + dbc_signal.name;
        if (session->series_by_path.find(path) == session->series_by_path.end()) {
          continue;
        }
        message.signals.push_back(CabanaSignalSummary{
          .path = path,
          .name = dbc_signal.name,
          .unit = dbc_signal.unit,
          .receiver_name = dbc_signal.receiver_name,
          .comment = dbc_signal.comment,
          .start_bit = dbc_signal.start_bit,
          .msb = dbc_signal.msb,
          .lsb = dbc_signal.lsb,
          .size = dbc_signal.size,
          .factor = dbc_signal.factor,
          .offset = dbc_signal.offset,
          .min = dbc_signal.min,
          .max = dbc_signal.max,
          .type = static_cast<int>(dbc_signal.type),
          .multiplex_value = dbc_signal.multiplex_value,
          .value_description_count = static_cast<int>(dbc_signal.value_descriptions.size()),
          .is_signed = dbc_signal.is_signed,
          .is_little_endian = dbc_signal.is_little_endian,
          .has_bit_range = true,
        });
      }
    }
    if (message_data.samples.size() > 1
        && message_data.samples.back().mono_time > message_data.samples.front().mono_time) {
      message.frequency_hz = static_cast<double>(message_data.samples.size() - 1)
                           / (message_data.samples.back().mono_time - message_data.samples.front().mono_time);
    }
    messages.push_back(std::move(message));
  }

  std::sort(messages.begin(), messages.end(), [](const CabanaMessageSummary &a, const CabanaMessageSummary &b) {
    return std::make_tuple(a.service, a.bus, a.has_address ? 0 : 1, a.address, a.name)
         < std::make_tuple(b.service, b.bus, b.has_address ? 0 : 1, b.address, b.name);
  });
  session->cabana_messages = std::move(messages);
}

void draw_cabana_mode(AppSession *session, const UiMetrics &ui, UiState *state) {
  sync_cabana_selection(session, state);

  ImGui::SetNextWindowPos(ImVec2(ui.content_x, ui.content_y));
  ImGui::SetNextWindowSize(ImVec2(ui.content_w, ui.content_h));
  push_cabana_mode_style();
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoSavedSettings;
  if (ImGui::Begin("##cabana_mode_host", nullptr, flags)) {
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const CabanaMessageSummary *message = find_selected_message(*session, *state);
    const float min_center_width = message != nullptr ? kMinCenterWidth : 140.0f;
    const float content_h = std::max(200.0f, avail.y);
    const float usable_w = std::max(kMinMessagesWidth + min_center_width + kMinRightWidth,
                                    avail.x - 2.0f * kSplitterThickness);
    const float min_left_frac = kMinMessagesWidth / usable_w;
    const float min_center_frac = min_center_width / usable_w;
    const float min_right_frac = kMinRightWidth / usable_w;

    state->cabana.layout_left_frac = std::clamp(state->cabana.layout_left_frac,
                                                min_left_frac,
                                                std::max(min_left_frac, 1.0f - min_center_frac - min_right_frac));
    state->cabana.layout_center_frac = std::clamp(state->cabana.layout_center_frac,
                                                  min_center_frac,
                                                  std::max(min_center_frac, 1.0f - state->cabana.layout_left_frac - min_right_frac));

    float messages_width = std::floor(usable_w * state->cabana.layout_left_frac);
    float center_width = std::floor(usable_w * state->cabana.layout_center_frac);
    float right_width = usable_w - messages_width - center_width;
    if (right_width < kMinRightWidth) {
      right_width = kMinRightWidth;
      center_width = std::max(min_center_width, usable_w - messages_width - right_width);
      messages_width = std::max(kMinMessagesWidth, usable_w - center_width - right_width);
    }
    center_width = std::max(min_center_width, center_width);
    state->cabana.layout_left_frac = messages_width / usable_w;
    state->cabana.layout_center_frac = center_width / usable_w;

    const ImVec2 origin = ImGui::GetCursorPos();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::BeginChild("##cabana_messages_panel", ImVec2(messages_width, content_h), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar);
    draw_messages_panel(session, state);
    ImGui::EndChild();
    ImGui::PopStyleVar();

    const float center_height = content_h;
    ImGui::SetCursorPos(ImVec2(origin.x + messages_width, origin.y));
    draw_vertical_splitter("##cabana_left_splitter", content_h, kMinMessagesWidth,
                           std::max(kMinMessagesWidth, usable_w - min_center_width - right_width),
                           &messages_width);
    messages_width = std::clamp(messages_width, kMinMessagesWidth, std::max(kMinMessagesWidth, usable_w - min_center_width - right_width));
    center_width = usable_w - messages_width - right_width;
    state->cabana.layout_left_frac = messages_width / usable_w;
    state->cabana.layout_center_frac = center_width / usable_w;

    ImGui::SetCursorPos(ImVec2(origin.x + messages_width + kSplitterThickness, origin.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::BeginChild("##cabana_detail_panel", ImVec2(center_width, center_height), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar);
    if (message == nullptr) {
      draw_cabana_welcome_panel();
    } else {
      draw_detail_panel(session, state, *message);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::SetCursorPos(ImVec2(origin.x + messages_width + kSplitterThickness + center_width, origin.y));
    draw_right_splitter("##cabana_right_splitter", content_h, kMinRightWidth,
                        std::max(kMinRightWidth, usable_w - messages_width - min_center_width),
                        &right_width);
    right_width = std::clamp(right_width, kMinRightWidth, std::max(kMinRightWidth, usable_w - messages_width - min_center_width));
    center_width = usable_w - messages_width - right_width;
    state->cabana.layout_center_frac = center_width / usable_w;

    ImGui::SetCursorPos(ImVec2(origin.x + messages_width + center_width + 2.0f * kSplitterThickness, origin.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::BeginChild("##cabana_right_panel", ImVec2(right_width, center_height), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar);
    const float right_avail_y = ImGui::GetContentRegionAvail().y;
    const float right_split_span = std::max(1.0f, right_avail_y - kSplitterThickness);
    const float min_right_top_frac = kMinTopHeight / right_split_span;
    const float min_right_bottom_frac = kMinBottomHeight / right_split_span;
    state->cabana.layout_right_top_frac = std::clamp(state->cabana.layout_right_top_frac,
                                                     min_right_top_frac,
                                                     std::max(min_right_top_frac, 1.0f - min_right_bottom_frac));
    float right_top_height = std::clamp(std::floor(right_split_span * state->cabana.layout_right_top_frac),
                                        kMinTopHeight,
                                        std::max(kMinTopHeight, right_avail_y - kMinBottomHeight - kSplitterThickness));
    draw_video_panel(session, state, right_top_height);
    if (draw_horizontal_splitter("##cabana_right_hsplit",
                                 ImGui::GetContentRegionAvail().x,
                                 kMinTopHeight,
                                 std::max(kMinTopHeight, ImGui::GetContentRegionAvail().y - kMinBottomHeight),
                                 &right_top_height)) {
      state->cabana.layout_right_top_frac = std::clamp(right_top_height / right_split_span,
                                                       min_right_top_frac,
                                                       std::max(min_right_top_frac, 1.0f - min_right_bottom_frac));
    }
    draw_chart_panel(session, state, message);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::SetCursorPos(ImVec2(origin.x, origin.y));
    ImGui::Dummy(avail);
  }
  ImGui::End();
  pop_cabana_mode_style();
  if (state->cabana.pending_apply_signal_edit) {
    state->cabana.pending_apply_signal_edit = false;
    apply_cabana_signal_edit(session, state);
  }
  if (state->cabana.pending_delete_signal) {
    state->cabana.pending_delete_signal = false;
    apply_cabana_signal_delete(session, state);
  }
}
