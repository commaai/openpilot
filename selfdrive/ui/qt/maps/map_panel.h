#pragma once

#include <QFrame>
#include <QMapboxGL>
#include <QStackedLayout>

class MapPanel : public QFrame {
  Q_OBJECT

public:
  explicit MapPanel(const QMapboxGLSettings &settings, QWidget *parent = nullptr);

signals:
  void mapPanelRequested();

public slots:
  void requestVisible(bool visible);
//  void requestMapSettings(bool settings);
  void toggleMapSettings();


private:
  QStackedLayout *content_stack;
};
