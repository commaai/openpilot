#include "tools/cabana/utils/util.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <QColor>
#include <QFontDatabase>
#include <QPixmapCache>
#include <QPainterPath>
#include <unordered_map>
#include "common/util.h"

// SegmentTree

void SegmentTree::build(const std::vector<QPointF> &arr) {
  size = arr.size();
  tree.resize(4 * size);  // size of the tree is 4 times the size of the array
  if (size > 0) {
    build_tree(arr, 1, 0, size - 1);
  }
}

void SegmentTree::build_tree(const std::vector<QPointF> &arr, int n, int left, int right) {
  if (left == right) {
    const double y = arr[left].y();
    tree[n] = {y, y};
  } else {
    const int mid = (left + right) >> 1;
    build_tree(arr, 2 * n, left, mid);
    build_tree(arr, 2 * n + 1, mid + 1, right);
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

// MessageBytesDelegate

MessageBytesDelegate::MessageBytesDelegate(QObject *parent, bool multiple_lines)
    : font_metrics(QApplication::font()), multiple_lines(multiple_lines), QStyledItemDelegate(parent) {
  fixed_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
  byte_size = QFontMetrics(fixed_font).size(Qt::TextSingleLine, "00 ") + QSize(0, 2);
  for (int i = 0; i < 256; ++i) {
    hex_text_table[i].setText(QStringLiteral("%1").arg(i, 2, 16, QLatin1Char('0')).toUpper());
    hex_text_table[i].prepare({}, fixed_font);
  }
  h_margin = QApplication::style()->pixelMetric(QStyle::PM_FocusFrameHMargin) + 1;
  v_margin = QApplication::style()->pixelMetric(QStyle::PM_FocusFrameVMargin) + 1;
}

QSize MessageBytesDelegate::sizeForBytes(int n) const {
  int rows = multiple_lines ? std::max(1, n / 8) : 1;
  return {(n / rows) * byte_size.width() + h_margin * 2, rows * byte_size.height() + v_margin * 2};
}

QSize MessageBytesDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto data = index.data(BytesRole);
  return sizeForBytes(data.isValid() ? static_cast<std::vector<uint8_t> *>(data.value<void *>())->size() : 0);
}

void MessageBytesDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  if (option.state & QStyle::State_Selected) {
    painter->fillRect(option.rect, option.palette.brush(QPalette::Normal, QPalette::Highlight));
  }

  QRect item_rect = option.rect.adjusted(h_margin, v_margin, -h_margin, -v_margin);
  QColor highlighted_color = option.palette.color(QPalette::HighlightedText);
  auto text_color = index.data(Qt::ForegroundRole).value<QColor>();
  bool inactive = text_color.isValid();
  if (!inactive) {
    text_color = option.palette.color(QPalette::Text);
  }
  auto data = index.data(BytesRole);
  if (!data.isValid()) {
    painter->setFont(option.font);
    painter->setPen(option.state & QStyle::State_Selected ? highlighted_color : text_color);
    QString text = font_metrics.elidedText(index.data(Qt::DisplayRole).toString(), Qt::ElideRight, item_rect.width());
    painter->drawText(item_rect, Qt::AlignLeft | Qt::AlignVCenter, text);
    return;
  }

  // Paint hex column
  const auto &bytes = *static_cast<std::vector<uint8_t> *>(data.value<void *>());
  const auto &colors = *static_cast<std::vector<CabanaColor> *>(index.data(ColorsRole).value<void *>());

  painter->setFont(fixed_font);
  const QPen text_pen(option.state & QStyle::State_Selected ? highlighted_color : text_color);
  const QPoint pt = item_rect.topLeft();
  for (int i = 0; i < bytes.size(); ++i) {
    int row = !multiple_lines ? 0 : i / 8;
    int column = !multiple_lines ? i : i % 8;
    QRect r({pt.x() + column * byte_size.width(), pt.y() + row * byte_size.height()}, byte_size);

    if (!inactive && i < colors.size() && colors[i].alpha() > 0) {
      if (option.state & QStyle::State_Selected) {
        painter->setPen(option.palette.color(QPalette::Text));
        painter->fillRect(r, option.palette.color(QPalette::Window));
      }
      painter->fillRect(r, toQColor(colors[i]));
    } else {
      painter->setPen(text_pen);
    }
    utils::drawStaticText(painter, r, hex_text_table[bytes[i]]);
  }
}

// TabBar

int TabBar::addTab(const QString &text) {
  int index = QTabBar::addTab(text);
  QToolButton *btn = new ToolButton("x", tr("Close Tab"));
  int width = style()->pixelMetric(QStyle::PM_TabCloseIndicatorWidth, nullptr, btn);
  int height = style()->pixelMetric(QStyle::PM_TabCloseIndicatorHeight, nullptr, btn);
  btn->setFixedSize({width, height});
  setTabButton(index, QTabBar::RightSide, btn);
  QObject::connect(btn, &QToolButton::clicked, this, &TabBar::closeTabClicked);
  return index;
}

void TabBar::closeTabClicked() {
  QObject *object = sender();
  for (int i = 0; i < count(); ++i) {
    if (tabButton(i, QTabBar::RightSide) == object) {
      emit tabCloseRequested(i);
      break;
    }
  }
}

// UnixSignalHandler

UnixSignalHandler::UnixSignalHandler() {
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sig_fd)) {
    qFatal("Couldn't create TERM socketpair");
  }

  waiter = std::thread([this]() {
    int tmp = 0;
    while (::read(sig_fd[1], &tmp, sizeof(tmp)) < 0) {
      if (errno != EINTR) return;
    }
    if (shutting_down.load()) return;

    // Marshal exit onto the GUI thread (qApp methods are not thread-safe).
    QMetaObject::invokeMethod(qApp, []() {
      printf("\nexiting...\n");
      qApp->closeAllWindows();
      qApp->exit();
    }, Qt::QueuedConnection);
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

// NameValidator

NameValidator::NameValidator(QObject *parent) : QValidator(parent) {}

QValidator::State NameValidator::validate(QString &input, int &pos) const {
  Q_UNUSED(pos);
  input.replace(' ', '_');
  if (input.isEmpty()) return QValidator::Intermediate;
  for (const QChar &c : input) {
    if (!c.isLetterOrNumber() && c != '_') return QValidator::Invalid;
  }
  return QValidator::Acceptable;
}

// NodeValidator

NodeValidator::NodeValidator(QObject *parent) : QValidator(parent) {}

QValidator::State NodeValidator::validate(QString &input, int &pos) const {
  Q_UNUSED(pos);
  if (input.isEmpty()) return QValidator::Intermediate;
  // Match ^\w+(,\w+)*$ ; a trailing comma is Intermediate (user still typing).
  bool need_word = true;
  for (const QChar &c : input) {
    if (c.isLetterOrNumber() || c == '_') {
      need_word = false;
    } else if (c == ',' && !need_word) {
      need_word = true;
    } else {
      return QValidator::Invalid;
    }
  }
  return need_word ? QValidator::Intermediate : QValidator::Acceptable;
}

// NonWhitespaceValidator

NonWhitespaceValidator::NonWhitespaceValidator(QObject *parent) : QValidator(parent) {}

QValidator::State NonWhitespaceValidator::validate(QString &input, int &pos) const {
  Q_UNUSED(pos);
  if (input.isEmpty()) return QValidator::Intermediate;
  for (const QChar &c : input) {
    if (c.isSpace()) return QValidator::Invalid;
  }
  return QValidator::Acceptable;
}

// IpAddressValidator

IpAddressValidator::IpAddressValidator(QObject *parent) : QValidator(parent) {}

QValidator::State IpAddressValidator::validate(QString &input, int &pos) const {
  Q_UNUSED(pos);
  if (input.isEmpty()) return QValidator::Intermediate;

  int dots = 0;
  int value = 0;
  bool has_digit = false;
  for (const QChar &c : input) {
    if (c.isDigit()) {
      value = has_digit ? value * 10 + c.digitValue() : c.digitValue();
      if (value > 255) return QValidator::Invalid;
      has_digit = true;
    } else if (c == '.') {
      if (!has_digit || dots >= 3) return QValidator::Invalid;
      ++dots;
      has_digit = false;
      value = 0;
    } else {
      return QValidator::Invalid;
    }
  }
  return (dots == 3 && has_digit) ? QValidator::Acceptable : QValidator::Intermediate;
}

DoubleValidator::DoubleValidator(QObject *parent) : QValidator(parent) {}

QValidator::State DoubleValidator::validate(QString &input, int &pos) const {
  Q_UNUSED(pos);
  if (input.isEmpty()) return QValidator::Intermediate;

  // Match QString::toDouble(): C locale, no hex floats / inf / nan.
  const std::string bytes = input.toLatin1().toStdString();
  // strtod accepts 0x… hex floats and p-exponents; QString::toDouble does not.
  if (bytes.find_first_of("xXpP") != std::string::npos) {
    return QValidator::Invalid;
  }

  const char *start = bytes.c_str();
  char *end = nullptr;
  const double value = std::strtod(start, &end);
  if (end == start) {
    // Still typing a sign, decimal point, or exponent prefix.
    if (input == "-" || input == "+" || input == "." || input == "-." || input == "+.") {
      return QValidator::Intermediate;
    }
    return QValidator::Invalid;
  }
  if (*end == '\0') {
    // Reject inf/nan (strtod accepts them; QDoubleValidator / toDouble path should not).
    return std::isfinite(value) ? QValidator::Acceptable : QValidator::Invalid;
  }

  // Partial exponent / trailing sign while typing (e.g. "1e", "1e-", "1.").
  for (const char *p = end; *p; ++p) {
    const char c = *p;
    if (!(c == 'e' || c == 'E' || c == '+' || c == '-' || c == '.' || (c >= '0' && c <= '9'))) {
      return QValidator::Invalid;
    }
  }
  return QValidator::Intermediate;
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

bool isDarkTheme() {
  QColor windowColor = QApplication::palette().color(QPalette::Window);
  return windowColor.lightness() < 128;
}

QPixmap icon(const QString &id) {
  bool dark_theme = isDarkTheme();

  QPixmap pm;
  QString key = "bootstrap_" % id % (dark_theme ? "1" : "0");
  if (!QPixmapCache::find(key, &pm)) {
    pm = bootstrapPixmap(id);
    if (dark_theme) {
      QPainter p(&pm);
      p.setCompositionMode(QPainter::CompositionMode_SourceIn);
      p.fillRect(pm.rect(), QColor("#bbbbbb"));
    }
    QPixmapCache::insert(key, pm);
  }
  return pm;
}

void setTheme(int theme) {
  auto style = QApplication::style();
  if (!style) return;

  static int prev_theme = 0;
  if (theme != prev_theme) {
    prev_theme = theme;
    QPalette new_palette;
    if (theme == DARK_THEME) {
      // "Darcula" like dark theme
      new_palette.setColor(QPalette::Window, QColor("#353535"));
      new_palette.setColor(QPalette::WindowText, QColor("#bbbbbb"));
      new_palette.setColor(QPalette::Base, QColor("#3c3f41"));
      new_palette.setColor(QPalette::AlternateBase, QColor("#3c3f41"));
      new_palette.setColor(QPalette::ToolTipBase, QColor("#3c3f41"));
      new_palette.setColor(QPalette::ToolTipText, QColor("#bbb"));
      new_palette.setColor(QPalette::Text, QColor("#bbbbbb"));
      new_palette.setColor(QPalette::Button, QColor("#3c3f41"));
      new_palette.setColor(QPalette::ButtonText, QColor("#bbbbbb"));
      new_palette.setColor(QPalette::Highlight, QColor("#2f65ca"));
      new_palette.setColor(QPalette::HighlightedText, QColor("#bbbbbb"));
      new_palette.setColor(QPalette::BrightText, QColor("#f0f0f0"));
      new_palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#777777"));
      new_palette.setColor(QPalette::Disabled, QPalette::WindowText, QColor("#777777"));
      new_palette.setColor(QPalette::Disabled, QPalette::Text, QColor("#777777"));
      new_palette.setColor(QPalette::Light, QColor("#777777"));
      new_palette.setColor(QPalette::Dark, QColor("#353535"));
    } else {
      new_palette = style->standardPalette();
    }
    qApp->setPalette(new_palette);
    style->polish(qApp);
    for (auto w : QApplication::allWidgets()) {
      w->setPalette(new_palette);
    }
  }
}

QString formatSeconds(double sec, bool include_milliseconds, bool absolute_time) {
  if (absolute_time) {
    const auto ms_total = static_cast<int64_t>(std::llround(sec * 1000.0));
    const std::time_t secs = static_cast<std::time_t>(ms_total / 1000);
    int millis = static_cast<int>(ms_total % 1000);
    if (millis < 0) millis = -millis;
    std::tm tm{};
    localtime_r(&secs, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    if (include_milliseconds) {
      return QString::asprintf("%s.%03d", buf, millis);
    }
    return QString::fromUtf8(buf);
  }

  // Relative duration (not wall-clock).
  const bool show_hours = sec > 60 * 60;
  int total_ms = static_cast<int>(std::llround(std::max(0.0, sec) * 1000.0));
  const int hours = total_ms / (3600 * 1000);
  const int minutes = (total_ms / (60 * 1000)) % 60;
  const int seconds = (total_ms / 1000) % 60;
  const int millis = total_ms % 1000;
  if (show_hours) {
    return include_milliseconds ? QString::asprintf("%02d:%02d:%02d.%03d", hours, minutes, seconds, millis)
                                : QString::asprintf("%02d:%02d:%02d", hours, minutes, seconds);
  }
  return include_milliseconds ? QString::asprintf("%02d:%02d.%03d", minutes, seconds, millis)
                              : QString::asprintf("%02d:%02d", minutes, seconds);
}

}  // namespace utils

int num_decimals(double num) {
  const QString string = QString::number(num);
  auto dot_pos = string.indexOf('.');
  return dot_pos == -1 ? 0 : string.size() - dot_pos - 1;
}

QString signalToolTip(const cabana::Signal *sig) {
  return QObject::tr(R"(
    %1<br /><span font-size:small">
    Start Bit: %2 Size: %3<br />
    MSB: %4 LSB: %5<br />
    Little Endian: %6 Signed: %7</span>
  )").arg(QString::fromStdString(sig->name)).arg(sig->start_bit).arg(sig->size).arg(sig->msb).arg(sig->lsb)
     .arg(sig->is_little_endian ? "Y" : "N").arg(sig->is_signed ? "Y" : "N");
}

void sigTermHandler(int s) {
  std::signal(s, SIG_DFL);
  qApp->quit();
}

void initApp(int argc, char *argv[], bool disable_hidpi) {
  // setup signal handlers to exit gracefully
  std::signal(SIGINT, sigTermHandler);
  std::signal(SIGTERM, sigTermHandler);

  std::filesystem::path app_dir;
#ifdef __APPLE__
  // Get the devicePixelRatio, and scale accordingly to maintain 1:1 rendering
  QApplication tmp(argc, argv);
  app_dir = QCoreApplication::applicationDirPath().toStdString();
  if (disable_hidpi) {
    qputenv("QT_SCALE_FACTOR", QString::number(1.0 / tmp.devicePixelRatio()).toLocal8Bit());
  }
#else
  app_dir = std::filesystem::path(util::readlink("/proc/self/exe")).parent_path();
#endif

  qputenv("QT_DBL_CLICK_DIST", "150");
  // ensure the current dir matches the exectuable's directory
  std::error_code ec;
  std::filesystem::current_path(app_dir, ec);
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

QPixmap bootstrapPixmap(const QString &id) {
  static auto icons = load_bootstrap_icons();

  QPixmap pixmap;
  auto it = icons.find(id.toStdString());
  if (it != icons.end()) {
    pixmap.loadFromData((const uchar *)it->second.data(), it->second.size(), "svg");
  }
  return pixmap;
}
