#include "selfdrive/ui/qt/maps/map_instructions.h"

#include <QDir>
#include <QVBoxLayout>

#include "selfdrive/ui/ui.h"

const QString ICON_SUFFIX = ".png";

MapInstructions::MapInstructions(QWidget *parent) : QWidget(parent) {
  is_rhd = Params().getBool("IsRhdDetected");
  QHBoxLayout *main_layout = new QHBoxLayout(this);
  main_layout->setContentsMargins(11, 50, 11, 11);
  main_layout->addWidget(icon_01 = new QLabel, 0, Qt::AlignTop);

  QWidget *right_container = new QWidget(this);
  right_container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  QVBoxLayout *layout = new QVBoxLayout(right_container);

  layout->addWidget(distance = new QLabel);
  distance->setStyleSheet(R"(font-size: 90px;)");

  layout->addWidget(primary = new QLabel);
  primary->setStyleSheet(R"(font-size: 60px;)");
  primary->setWordWrap(true);

  layout->addWidget(secondary = new QLabel);
  secondary->setStyleSheet(R"(font-size: 50px;)");
  secondary->setWordWrap(true);

  layout->addLayout(lane_layout = new QHBoxLayout);
  main_layout->addWidget(right_container);

  setStyleSheet("color:white");
  QPalette pal = palette();
  pal.setColor(QPalette::Background, QColor(0, 0, 0, 150));
  setAutoFillBackground(true);
  setPalette(pal);

  buildPixmapCache();
}

void MapInstructions::buildPixmapCache() {
  QDir dir("../assets/navigation");
  for (QString fn : dir.entryList({"*" + ICON_SUFFIX}, QDir::Files)) {
    QPixmap pm(dir.filePath(fn));
    QString key = fn.left(fn.size() - ICON_SUFFIX.length());
    pm = pm.scaledToWidth(200, Qt::SmoothTransformation);

    // Maneuver icons
    pixmap_cache[key] = pm;
    // lane direction icons
    if (key.contains("turn_")) {
      pixmap_cache["lane_" + key] = pm.scaled({125, 125}, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    }

    // for rhd, reflect direction and then flip
    if (key.contains("_left")) {
      pixmap_cache["rhd_" + key.replace("_left", "_right")] = pm.transformed(QTransform().scale(-1, 1));
    } else if (key.contains("_right")) {
      pixmap_cache["rhd_" + key.replace("_right", "_left")] = pm.transformed(QTransform().scale(-1, 1));
    }
  }
}

QString MapInstructions::getDistance(float d) {
  d = std::max(d, 0.0f);
  if (uiState()->scene.is_metric) {
    return (d > 500) ? QString::number(d / 1000, 'f', 1) + tr(" km")
                     : QString::number(50 * int(d / 50)) + tr(" m");
  } else {
    float feet = d * METER_TO_FOOT;
    return (feet > 500) ? QString::number(d * METER_TO_MILE, 'f', 1) + tr(" mi")
                        : QString::number(50 * int(feet / 50)) + tr(" ft");
  }
}

void MapInstructions::updateInstructions(cereal::NavInstruction::Reader instruction) {
  setUpdatesEnabled(false);

  // Show instruction text
  QString primary_str = QString::fromStdString(instruction.getManeuverPrimaryText());
  QString secondary_str = QString::fromStdString(instruction.getManeuverSecondaryText());

  primary->setText(primary_str);
  secondary->setVisible(secondary_str.length() > 0);
  secondary->setText(secondary_str);
  distance->setText(getDistance(instruction.getManeuverDistance()));

  // Show arrow with direction
  QString type = QString::fromStdString(instruction.getManeuverType());
  QString modifier = QString::fromStdString(instruction.getManeuverModifier());
  if (!type.isEmpty()) {
    QString fn = "direction_" + type;
    if (!modifier.isEmpty()) {
      fn += "_" + modifier;
    }
    fn = fn.replace(' ', '_');
    bool rhd = is_rhd && (fn.contains("_left") || fn.contains("_right"));
    icon_01->setPixmap(pixmap_cache[!rhd ? fn : "rhd_" + fn]);
    icon_01->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    icon_01->setVisible(true);
  }

  // Hide distance after arrival
  distance->setVisible(type != "arrive" || instruction.getManeuverDistance() > 0);

  // Show lanes
  auto lanes = instruction.getLanes();
  for (int i = 0; i < lanes.size(); ++i) {
    bool active = lanes[i].getActive();

    bool left = false, straight = false, right = false;
    for (auto const &direction : lanes[i].getDirections()) {
      left |= direction == cereal::NavInstruction::Direction::LEFT;
      right |= direction == cereal::NavInstruction::Direction::RIGHT;
      straight |= direction == cereal::NavInstruction::Direction::STRAIGHT;
    }

    // active direction has precedence
    const auto active_direction = lanes[i].getActiveDirection();
    bool active_left = active_direction == cereal::NavInstruction::Direction::LEFT;
    bool active_right = active_direction == cereal::NavInstruction::Direction::RIGHT;

    // TODO: Make more images based on active direction and combined directions
    QString fn = "lane_direction_";
    if (left && (active_left || !active)) {
      fn += "turn_left";
    } else if (right && (active_right || !active)) {
      fn += "turn_right";
    } else if (straight) {
      fn += "turn_straight";
    }

    if (!active) {
      fn += "_inactive";
    }

    QLabel *label = (i < lane_labels.size()) ? lane_labels[i] : lane_labels.emplace_back(new QLabel);
    if (!label->parentWidget()) {
      lane_layout->addWidget(label);
    }
    label->setPixmap(pixmap_cache[fn]);
    label->setVisible(true);
  }

  for (int i = lanes.size(); i < lane_labels.size(); ++i) {
    lane_labels[i]->setVisible(false);
  }

  setUpdatesEnabled(true);
  setVisible(true);
}
