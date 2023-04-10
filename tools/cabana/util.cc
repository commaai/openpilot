#include "tools/cabana/util.h"

#include <QApplication>
#include <QFontDatabase>
#include <QPainter>
#include <QPixmapCache>
#include <cmath>
#include <limits>

#include "selfdrive/ui/qt/util.h"

static inline QColor blend(const QColor &a, const QColor &b) {
  return QColor((a.red() + b.red()) / 2, (a.green() + b.green()) / 2, (a.blue() + b.blue()) / 2, (a.alpha() + b.alpha()) / 2);
}

void ChangeTracker::compute(const QByteArray &dat, double ts, uint32_t freq) {
  if (prev_dat.size() != dat.size()) {
    colors.resize(dat.size());
    last_change_t.resize(dat.size());
    bit_change_counts.resize(dat.size());
    std::fill(colors.begin(), colors.end(), QColor(0, 0, 0, 0));
    std::fill(last_change_t.begin(), last_change_t.end(), ts);
  } else {
    int factor = settings.theme == DARK_THEME ? 135 : 0;
    QColor cyan = QColor(0, 187, 255, start_alpha).lighter(factor);
    QColor red = QColor(255, 0, 0, start_alpha).lighter(factor);
    QColor greyish_blue = QColor(102, 86, 169, start_alpha / 2).lighter(factor);

    for (int i = 0; i < dat.size(); ++i) {
      const uint8_t last = prev_dat[i];
      const uint8_t cur = dat[i];

      if (last != cur) {
        double delta_t = ts - last_change_t[i];
        if (delta_t * freq > periodic_threshold) {
          // Last change was while ago, choose color based on delta up or down
          colors[i] = (cur > last) ? cyan : red;
        } else {
          // Periodic changes
          colors[i] = blend(colors[i], greyish_blue);
        }

        // Track bit level changes
        for (int bit = 0; bit < 8; bit++) {
          if ((cur ^ last) & (1 << bit)) {
            bit_change_counts[i][bit] += 1;
          }
        }

        last_change_t[i] = ts;
      } else {
        // Fade out
        float alpha_delta = 1.0 / (freq + 1) / fade_time;
        colors[i].setAlphaF(std::max(0.0, colors[i].alphaF() - alpha_delta));
      }
    }
  }

  prev_dat = dat;
}

void ChangeTracker::clear() {
  prev_dat.clear();
  last_change_t.clear();
  bit_change_counts.clear();
  colors.clear();
}

// SegmentTree

void SegmentTree::build(const QVector<QPointF> &arr) {
  size = arr.size();
  tree.resize(4 * size);  // size of the tree is 4 times the size of the array
  if (size > 0) {
    build_tree(arr, 1, 0, size - 1);
  }
}

void SegmentTree::build_tree(const QVector<QPointF> &arr, int n, int left, int right) {
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

MessageBytesDelegate::MessageBytesDelegate(QObject *parent) : QStyledItemDelegate(parent) {
  fixed_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
  byte_width = QFontMetrics(fixed_font).width("00 ");
}

void MessageBytesDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto colors = index.data(ColorsRole).value<QVector<QColor>>();
  auto byte_list = index.data(BytesRole).toByteArray();

  int v_margin = option.widget->style()->pixelMetric(QStyle::PM_FocusFrameVMargin);
  int h_margin = option.widget->style()->pixelMetric(QStyle::PM_FocusFrameHMargin);
  QRect rc{option.rect.left() + h_margin, option.rect.top() + v_margin, byte_width, option.rect.height() - 2 * v_margin};

  auto color_role = option.state & QStyle::State_Selected ? QPalette::HighlightedText : QPalette::Text;
  painter->setPen(option.palette.color(color_role));
  painter->setFont(fixed_font);
  for (int i = 0; i < byte_list.size(); ++i) {
    if (i < colors.size() && colors[i].alpha() > 0) {
      painter->fillRect(rc, colors[i]);
    }
    painter->drawText(rc, Qt::AlignCenter, toHex(byte_list[i]));
    rc.moveLeft(rc.right() + 1);
  }
}

QColor getColor(const cabana::Signal *sig) {
  float h = 19 * (float)sig->lsb / 64.0;
  h = fmod(h, 1.0);

  size_t hash = qHash(sig->name);
  float s = 0.25 + 0.25 * (float)(hash & 0xff) / 255.0;
  float v = 0.75 + 0.25 * (float)((hash >> 8) & 0xff) / 255.0;

  return QColor::fromHsvF(h, s, v);
}

NameValidator::NameValidator(QObject *parent) : QRegExpValidator(QRegExp("^(\\w+)"), parent) {}

QValidator::State NameValidator::validate(QString &input, int &pos) const {
  input.replace(' ', '_');
  return QRegExpValidator::validate(input, pos);
}

namespace utils {
QPixmap icon(const QString &id) {
  bool dark_theme = settings.theme == DARK_THEME;
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
      new_palette.setColor(QPalette::Disabled, QPalette::Text, QColor("#777777"));;
      new_palette.setColor(QPalette::Light, QColor("#3c3f41"));
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

}  // namespace utils

QString toHex(uint8_t byte) {
  static std::array<QString, 256> hex = []() {
    std::array<QString, 256> ret;
    for (int i = 0; i < 256; ++i) ret[i] = QStringLiteral("%1").arg(i, 2, 16, QLatin1Char('0')).toUpper();
    return ret;
  }();
  return hex[byte];
}

int num_decimals(double num) {
  const QString string = QString::number(num);
  const QStringList split = string.split('.');
  if (split.size() == 1) {
    return 0;
  } else {
    return split[1].size();
  }
}
