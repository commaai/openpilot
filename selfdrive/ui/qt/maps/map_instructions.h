#pragma once

#include <QHash>
#include <QHBoxLayout>
#include <QLabel>

#include "cereal/gen/cpp/log.capnp.h"

class MapInstructions : public QWidget {
  Q_OBJECT

private:
  QLabel *distance;
  QLabel *primary;
  QLabel *secondary;
  QLabel *icon_01;
  QHBoxLayout *lane_layout;
  bool is_rhd = false;
  std::vector<QLabel *> lane_labels;
  QHash<QString, QPixmap> pixmap_cache;

public:
  MapInstructions(QWidget * parent=nullptr);
  void buildPixmapCache();
  QString getDistance(float d);
  void updateInstructions(cereal::NavInstruction::Reader instruction);
};
