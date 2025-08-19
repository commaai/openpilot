#include "tools/cabana/utils/util.h"

#include <algorithm>
#include <csignal>
#include <limits>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

#include <QColor>
#include <QDateTime>
#include <QFontDatabase>
#include <QLocale>
#include <QPixmapCache>

#include "selfdrive/ui/qt/util.h"

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
  const auto &colors = *static_cast<std::vector<QColor> *>(index.data(ColorsRole).value<void *>());

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
      painter->fillRect(r, colors[i]);
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

UnixSignalHandler::UnixSignalHandler(QObject *parent) : QObject(nullptr) {
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sig_fd)) {
    qFatal("Couldn't create TERM socketpair");
  }

  sn = new QSocketNotifier(sig_fd[1], QSocketNotifier::Read, this);
  connect(sn, &QSocketNotifier::activated, this, &UnixSignalHandler::handleSigTerm);
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, UnixSignalHandler::signalHandler);
}

UnixSignalHandler::~UnixSignalHandler() {
  ::close(sig_fd[0]);
  ::close(sig_fd[1]);
}

void UnixSignalHandler::signalHandler(int s) {
  ::write(sig_fd[0], &s, sizeof(s));
}

void UnixSignalHandler::handleSigTerm() {
  sn->setEnabled(false);
  int tmp;
  ::read(sig_fd[1], &tmp, sizeof(tmp));

  printf("\nexiting...\n");
  qApp->closeAllWindows();
  qApp->exit();
}

// NameValidator

NameValidator::NameValidator(QObject *parent) : QRegExpValidator(QRegExp("^(\\w+)"), parent) {}

QValidator::State NameValidator::validate(QString &input, int &pos) const {
  input.replace(' ', '_');
  return QRegExpValidator::validate(input, pos);
}

DoubleValidator::DoubleValidator(QObject *parent) : QDoubleValidator(parent) {
  // Match locale of QString::toDouble() instead of system
  QLocale locale(QLocale::C);
  locale.setNumberOptions(QLocale::RejectGroupSeparator);
  setLocale(locale);
}

namespace utils {

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
  QString format = absolute_time ? "yyyy-MM-dd hh:mm:ss"
                                 : (sec > 60 * 60 ? "hh:mm:ss" : "mm:ss");
  if (include_milliseconds) format += ".zzz";
  return QDateTime::fromMSecsSinceEpoch(sec * 1000).toString(format);
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
  )").arg(sig->name).arg(sig->start_bit).arg(sig->size).arg(sig->msb).arg(sig->lsb)
     .arg(sig->is_little_endian ? "Y" : "N").arg(sig->is_signed ? "Y" : "N");
}
