#include "selfdrive/ui/qt/maps/map_instructions.h"

#include <QDir>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/maps/map_helpers.h"
#include "selfdrive/ui/ui.h"

const QString ICON_SUFFIX = ".png";

MapInstructions::MapInstructions(QWidget *parent) : QWidget(parent) {
  is_rhd = Params().getBool("IsRhdDetected");
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(11, UI_BORDER_SIZE, 11, 11);

  QHBoxLayout *top_layout = new QHBoxLayout;
  top_layout->addWidget(icon_01 = new QLabel, 0, Qt::AlignTop);

  QVBoxLayout *right_layout = new QVBoxLayout;
  right_layout->addWidget(distance = new QLabel);
  distance->setStyleSheet(R"(font-size: 90px;)");

  right_layout->addWidget(primary = new QLabel);
  primary->setStyleSheet(R"(font-size: 60px;)");
  primary->setWordWrap(true);

  right_layout->addWidget(secondary = new QLabel);
  secondary->setStyleSheet(R"(font-size: 50px;)");
  secondary->setWordWrap(true);

  top_layout->addLayout(right_layout);

  main_layout->addLayout(top_layout);
  main_layout->addLayout(lane_layout = new QHBoxLayout);
  lane_layout->setAlignment(Qt::AlignHCenter);
  lane_layout->setSpacing(10);

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

void MapInstructions::updateInstructions(cereal::NavInstruction::Reader instruction) {
  setUpdatesEnabled(false);

  // Show instruction text
  QString primary_str = QString::fromStdString(instruction.getManeuverPrimaryText());
  QString secondary_str = QString::fromStdString(instruction.getManeuverSecondaryText());

  primary->setText(primary_str);
  secondary->setVisible(secondary_str.length() > 0);
  secondary->setText(secondary_str);

  auto distance_str_pair = map_format_distance(instruction.getManeuverDistance(), uiState()->scene.is_metric);
  distance->setText(QString("%1 %2").arg(distance_str_pair.first, distance_str_pair.second));

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
  } else {
    icon_01->setVisible(false);
  }

  // Hide distance after arrival
  distance->setVisible(type != "arrive" || instruction.getManeuverDistance() > 0);

  // Show lanes
  auto lanes = instruction.getLanes();
  for (int i = 0; i < lanes.size(); ++i) {
    bool active = lanes[i].getActive();
    const auto active_direction = lanes[i].getActiveDirection();

    // TODO: Make more images based on active direction and combined directions
    QString fn = "lane_direction_";

    // active direction has precedence
    if (active && active_direction != cereal::NavInstruction::Direction::NONE) {
      fn += "turn_" + DIRECTIONS[active_direction];
    } else {
      for (auto const &direction : lanes[i].getDirections()) {
        if (direction != cereal::NavInstruction::Direction::NONE) {
          fn += "turn_" + DIRECTIONS[direction];
          break;
        }
      }
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
