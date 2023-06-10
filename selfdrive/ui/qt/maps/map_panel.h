#pragma once

#include <QFrame>
#include <QMapboxGL>

class MapPanel : public QFrame {
  Q_OBJECT
public:
  explicit MapPanel(const QMapboxGLSettings &settings, QWidget *parent = nullptr);
};
