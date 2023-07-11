#pragma once

#include <QFrame>
#include <QMapboxGL>
#include <QStackedLayout>

class MapPanel : public QFrame {
  Q_OBJECT

public:
  explicit MapPanel(const QMapboxGLSettings &settings, QWidget *parent = nullptr);

private:
  void showEvent(QShowEvent *event) { emit mapWindowShown(); };

signals:
  void mapWindowShown();

private:
  QStackedLayout *content_stack;
};
