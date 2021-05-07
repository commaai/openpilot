#pragma once

#include <QtWidgets>

#include <ui.h>

class Sidebar : public QFrame {
  Q_OBJECT

public:
  explicit Sidebar(QWidget* parent = 0);

signals:
  void openSettings();

public slots:
  void update(const UIState &s);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;

private:
  QImage home_img, settings_img;

  const QRect settings_btn = QRect(50, 35, 200, 117);
  const QColor good_color = QColor(255, 255, 255);
  const QColor warning_color = QColor(218, 202, 37);
  const QColor danger_cool  = QColor(201, 34, 49);
};
