#include "tools/cabana/util.h"

#include <QApplication>
#include <QFontDatabase>
#include <QPainter>

#include "selfdrive/ui/qt/util.h"

static QColor blend(QColor a, QColor b) {
  return QColor((a.red() + b.red()) / 2, (a.green() + b.green()) / 2, (a.blue() + b.blue()) / 2, (a.alpha() + b.alpha()) / 2);
}

void ChangeTracker::compute(const QByteArray &dat, double ts, uint32_t freq) {
  if (prev_dat.size() != dat.size()) {
    colors.resize(dat.size());
    last_change_t.resize(dat.size());
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
  colors.clear();
}

QList<QVariant> ChangeTracker::toVariantList(const QVector<QColor> &colors) {
  QList<QVariant> ret;
  ret.reserve(colors.size());
  for (auto &c : colors) ret.append(c);
  return ret;
}

// MessageBytesDelegate

MessageBytesDelegate::MessageBytesDelegate(QObject *parent) : QStyledItemDelegate(parent) {
  fixed_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
}

void MessageBytesDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
  QStyleOptionViewItemV4 opt = option;
  initStyleOption(&opt, index);

  if ((option.state & QStyle::State_Selected) && (option.state & QStyle::State_Active)) {
    painter->setPen(option.palette.color(QPalette::HighlightedText));
  } else {
    painter->setPen(option.palette.color(QPalette::Text));
  }

  painter->setFont(fixed_font);
  QRect space = painter->boundingRect(opt.rect, opt.displayAlignment, " ");
  QRect pos = painter->boundingRect(opt.rect, opt.displayAlignment, "00");
  pos.moveLeft(pos.x() + space.width());

  int m = space.width() / 2;
  const QMargins margins(m, m, m, m);

  QList<QVariant> colors = index.data(Qt::UserRole).toList();
  int i = 0;
  for (auto &byte : opt.text.split(" ")) {
    if (i < colors.size()) {
      painter->fillRect(pos.marginsAdded(margins), colors[i].value<QColor>());
    }
    painter->drawText(pos, opt.displayAlignment, byte);
    pos.moveLeft(pos.right() + space.width());
    i++;
  }
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
  QPixmap pm = bootstrapPixmap(id);
  if (dark_theme) {
    QPainter p(&pm);
    p.setCompositionMode(QPainter::CompositionMode_SourceIn);
    p.fillRect(pm.rect(), Qt::lightGray);
  }
  return pm;
}
}  // namespace utils
