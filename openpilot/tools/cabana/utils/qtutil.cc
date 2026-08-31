#include "tools/cabana/utils/qtutil.h"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <QColor>
#include <QFontDatabase>
#include <QPixmapCache>
#include <QPainterPath>

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

// validators

static QValidator::State toQtState(ValidState s) {
  switch (s) {
    case ValidState::Acceptable: return QValidator::Acceptable;
    case ValidState::Intermediate: return QValidator::Intermediate;
    default: return QValidator::Invalid;
  }
}

QValidator::State NameValidator::validate(QString &input, int &pos) const {
  std::string s = input.toStdString();
  auto state = validateName(s);
  input = QString::fromStdString(s);
  return toQtState(state);
}

QValidator::State NodeValidator::validate(QString &input, int &pos) const {
  return toQtState(validateNodes(input.toStdString()));
}

QValidator::State NonWhitespaceValidator::validate(QString &input, int &pos) const {
  return toQtState(validateNonWhitespace(input.toStdString()));
}

QValidator::State IpAddressValidator::validate(QString &input, int &pos) const {
  return toQtState(validateIpAddress(input.toStdString()));
}

QValidator::State DoubleValidator::validate(QString &input, int &pos) const {
  return toQtState(validateDouble(input.toLatin1().toStdString()));
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
      new_palette.setColor(QPalette::Window, toQColor(DarkTheme::window));
      new_palette.setColor(QPalette::WindowText, toQColor(DarkTheme::window_text));
      new_palette.setColor(QPalette::Base, toQColor(DarkTheme::base));
      new_palette.setColor(QPalette::AlternateBase, toQColor(DarkTheme::base));
      new_palette.setColor(QPalette::ToolTipBase, toQColor(DarkTheme::base));
      new_palette.setColor(QPalette::ToolTipText, toQColor(DarkTheme::tooltip_text));
      new_palette.setColor(QPalette::Text, toQColor(DarkTheme::text));
      new_palette.setColor(QPalette::Button, toQColor(DarkTheme::button));
      new_palette.setColor(QPalette::ButtonText, toQColor(DarkTheme::window_text));
      new_palette.setColor(QPalette::Highlight, toQColor(DarkTheme::highlight));
      new_palette.setColor(QPalette::HighlightedText, toQColor(DarkTheme::window_text));
      new_palette.setColor(QPalette::BrightText, toQColor(DarkTheme::bright_text));
      new_palette.setColor(QPalette::Disabled, QPalette::ButtonText, toQColor(DarkTheme::disabled_text));
      new_palette.setColor(QPalette::Disabled, QPalette::WindowText, toQColor(DarkTheme::disabled_text));
      new_palette.setColor(QPalette::Disabled, QPalette::Text, toQColor(DarkTheme::disabled_text));
      new_palette.setColor(QPalette::Light, toQColor(DarkTheme::light));
      new_palette.setColor(QPalette::Dark, toQColor(DarkTheme::dark));
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

}  // namespace utils

void sigTermHandler(int s) {
  std::signal(s, SIG_DFL);
  qApp->quit();
}

void initApp(int argc, char *argv[], bool disable_hidpi) {
  // setup signal handlers to exit gracefully
  std::signal(SIGINT, sigTermHandler);
  std::signal(SIGTERM, sigTermHandler);

#ifdef __APPLE__
  // Get the devicePixelRatio, and scale accordingly to maintain 1:1 rendering
  QApplication tmp(argc, argv);
  if (disable_hidpi) {
    qputenv("QT_SCALE_FACTOR", QString::number(1.0 / tmp.devicePixelRatio()).toLocal8Bit());
  }
#endif

  qputenv("QT_DBL_CLICK_DIST", "150");
  // ensure the current dir matches the exectuable's directory
  std::error_code ec;
  std::filesystem::current_path(executableDir(), ec);
}

QPixmap bootstrapPixmap(const QString &id) {
  QPixmap pixmap;
  const std::string svg = utils::bootstrapSvg(id.toStdString());
  if (!svg.empty()) {
    pixmap.loadFromData((const uchar *)svg.data(), svg.size(), "svg");
  }
  return pixmap;
}
