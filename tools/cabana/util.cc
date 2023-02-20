#include "tools/cabana/util.h"

#include <QApplication>
#include <QFontDatabase>
#include <QPainter>
#include <QPixmapCache>
#include <QDebug>

#include <limits>
#include <cmath>

#include "selfdrive/ui/qt/util.h"

static QColor blend(QColor a, QColor b) {
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
    for (int i = 0; i < dat.size(); ++i) {
      const uint8_t last = prev_dat[i];
      const uint8_t cur = dat[i];

      if (last != cur) {
        double delta_t = ts - last_change_t[i];
        if (delta_t * freq > periodic_threshold) {
          // Last change was while ago, choose color based on delta up or down
          if (cur > last) {
            colors[i] = QColor(0, 187, 255, start_alpha);  // Cyan
          } else {
            colors[i] = QColor(255, 0, 0, start_alpha);  // Red
          }
        } else {
          // Periodic changes
          colors[i] = blend(colors[i], QColor(102, 86, 169, start_alpha / 2));  // Greyish/Blue
        }

        // Track bit level changes
        for (int bit = 0; bit < 8; bit++){
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

// MessageBytesDelegate

MessageBytesDelegate::MessageBytesDelegate(QObject *parent) : QStyledItemDelegate(parent) {
  fixed_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
}

void MessageBytesDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  auto color_role = option.state & QStyle::State_Selected ? QPalette::HighlightedText: QPalette::Text;
  painter->setPen(option.palette.color(color_role));
  painter->setFont(fixed_font);
  int space = painter->boundingRect(option.rect, option.displayAlignment, " ").width();
  QRect pos = painter->boundingRect(option.rect, option.displayAlignment, "00").adjusted(0, 0, 2, 0);
  pos.moveLeft(pos.x() + space);
  int m = space / 2;
  const QMargins margins(m, m, m, m);

  auto colors = index.data(Qt::UserRole).value<QVector<QColor>>();
  auto byte_list = index.data(Qt::DisplayRole).toString().split(" ");
  for (int i = 0; i < byte_list.size(); ++i) {
    if (i < colors.size() && colors[i].alpha() > 0) {
      painter->fillRect(pos.marginsAdded(margins), colors[i]);
    }
    painter->drawText(pos, Qt::AlignCenter, byte_list[i]);
    pos.moveLeft(pos.right() + space);
  }
}

QColor getColor(const Signal *sig) {
  float h = 19 * (float)sig->lsb / 64.0;
  h = fmod(h, 1.0);

  size_t hash = qHash(sig->name);
  float s = 0.25 + 0.25 * (float)(hash & 0xff) / 255.0;
  float v = 0.75 + 0.25 * (float)((hash >> 8) & 0xff) / 255.0;

  return QColor::fromHsvF(h, s, v);
}

NameValidator::NameValidator(QObject *parent) : QRegExpValidator(QRegExp("^(\\w+)"), parent) { }

QValidator::State NameValidator::validate(QString &input, int &pos) const {
  input.replace(' ', '_');
  return QRegExpValidator::validate(input, pos);
}

namespace utils {
QPixmap icon(const QString &id) {
  static bool dark_theme = QApplication::style()->standardPalette().color(QPalette::WindowText).value() >
                           QApplication::style()->standardPalette().color(QPalette::Background).value();
  QPixmap pm;
  QString key = "bootstrap_" % id % (dark_theme ? "1" : "0");
  if (!QPixmapCache::find(key, &pm)) {
    pm = bootstrapPixmap(id);
    if (dark_theme) {
      QPainter p(&pm);
      p.setCompositionMode(QPainter::CompositionMode_SourceIn);
      p.fillRect(pm.rect(), Qt::lightGray);
    }
    QPixmapCache::insert(key, pm);
  }
  return pm;
}
}  // namespace utils

QToolButton *toolButton(const QString &icon, const QString &tooltip) {
  auto btn = new QToolButton();
  btn->setIcon(utils::icon(icon));
  btn->setToolTip(tooltip);
  btn->setAutoRaise(true);
  return btn;
};
