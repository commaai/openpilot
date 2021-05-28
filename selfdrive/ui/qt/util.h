#pragma once

#include <QtWidgets>

inline void configFont(QPainter &p, QString family, int size, const QString &style) {
  QFont f(family);
  f.setPixelSize(size);
  f.setStyleName(style);
  p.setFont(f);
}

inline void clearLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayout(childLayout);
    }
    delete item;
  }
}

inline QString getFormattedTimeSince(QDateTime *date) {
  date->setTimeSpec(Qt::UTC);
  int diff = date->secsTo(QDateTime::currentDateTimeUtc());
  QString formattedTime;

  if (diff < 60) {
    formattedTime = "now";
  } else if (diff < 3600) {
    formattedTime = QString::fromStdString(std::to_string(diff / 60) + " minute" + (diff >= 60 * 2 ? "s " : " ") + "ago");
  }else if (diff < 3600 * 24) {
    formattedTime = QString::fromStdString(std::to_string(diff / 3600) + " hour" + (diff >= 3600 * 2 ? "s " : " ") + "ago");
  } else if (diff < 3600 * 24 * 7) {
    formattedTime = QString::fromStdString(std::to_string(diff / (3600 * 24)) + " day" + (diff >= 3600 * 48 ? "s " : " ") + "ago");
  } else {
    formattedTime = date->date().toString();
  }

  return formattedTime;
}
