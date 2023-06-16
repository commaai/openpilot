#include "map_settings.h"

#include <vector>
#include <QDebug>

#include "common/util.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

static QString shorten(const QString &str, int max_len) {
  return str.size() > max_len ? str.left(max_len).trimmed() + "…" : str;
}

MapSettings::DestinationWidget::DestinationWidget(QWidget *parent) : ClickableWidget(parent) {
  setContentsMargins(0, 0, 0, 0);

  auto *frame = new QHBoxLayout(this);
  frame->setContentsMargins(32, 24, 32, 24);
  frame->setSpacing(40);

  icon = new QLabel(this);
  icon->setFixedSize(96, 96);
  frame->addWidget(icon);

  auto *inner_frame = new QVBoxLayout;
  inner_frame->setContentsMargins(0, 0, 0, 0);
  inner_frame->setSpacing(0);

  title = new QLabel(this);
  inner_frame->addWidget(title);

  subtitle = new QLabel(this);
  subtitle->setStyleSheet("color: #A0A0A0;");
  inner_frame->addWidget(subtitle);
  frame->addLayout(inner_frame, 1);

  action = new QLabel(this);
  action->setObjectName("action");
  action->setStyleSheet("font-size: 60px; font-weight: 600; border: none;");
  frame->addWidget(action);

  setFixedHeight(164);
  setStyleSheet(R"(
    /* default styles */
    ClickableWidget {
      background-color: #292929;
      border: 1px solid #4DFFFFFF;
      border-radius: 10px;
    }
    ClickableWidget QLabel {
      color: #FFFFFF;
      font-size: 48px;
      font-weight: 400;
    }

    /* on press */
    ClickableWidget:pressed {
      background-color: #3B3B3B;
    }
    ClickableWidget:pressed #action {
      color: #A0A0A0;
    }

    /* current destination */
    ClickableWidget[current=true] {
      background-color: #162440;
      border: 1px solid #80FFFFFF;
    }

    /* no saved destination */
    ClickableWidget[set=false] QLabel {
      color: #80FFFFFF;
    }
  )");
}

void MapSettings::DestinationWidget::set(NavDestination *destination,
                                         bool current) {
  setProperty("current", current);
  setProperty("set", true);

  auto title_text = destination->name;
  auto subtitle_text = destination->details;
  auto icon_pixmap = icons().recent;

  if (destination->isFavorite()) {
    if (destination->label == NAV_FAVORITE_LABEL_HOME) {
      title_text = tr("Home");
      icon_pixmap = icons().home;
    } else if (destination->label == NAV_FAVORITE_LABEL_WORK) {
      title_text = tr("Work");
      icon_pixmap = icons().work;
    } else {
      icon_pixmap = icons().favorite;
    }
  }

  title->setText(shorten(title_text, 26));
  subtitle->setText(shorten(subtitle_text, 26));
  subtitle->setVisible(true);
  icon->setPixmap(icon_pixmap);

  // TODO: use pixmap
  action->setText(current ? "×" : "→");
  action->setVisible(true);

  setStyleSheet(styleSheet());
}

void MapSettings::DestinationWidget::unset(const QString &label, bool current) {
  setProperty("current", current);
  setProperty("set", false);

  if (label.isEmpty()) {
    icon->setPixmap(icons().directions);
    title->setText(tr("No destination set"));
  } else {
    icon->setPixmap(label == NAV_FAVORITE_LABEL_HOME ? icons().home : icons().work);
    title->setText(tr("No %1 location set").arg(label));
  }

  subtitle->setVisible(false);
  action->setVisible(false);

  setStyleSheet(styleSheet());
}

MapSettings::MapSettings(QWidget *parent) : QFrame(parent) {
  setContentsMargins(0, 0, 0, 0);

  auto *frame = new QVBoxLayout(this);
  frame->setContentsMargins(40, 40, 40, 40);
  frame->setSpacing(32);


  auto *heading = new QHBoxLayout;
  heading->setContentsMargins(0, 0, 0, 0);
  heading->setSpacing(32);

  auto *title = new QLabel(tr("comma navigation"), this);
  title->setStyleSheet("color: #FFFFFF; font-size: 56px; font-weight: 500;");
  heading->addWidget(title, 1);

  auto *close_button = new QPushButton("×", this);
  close_button->setStyleSheet(R"(
    QPushButton {
      color: #FFFFFF;
      font-size: 60px;
      font-weight: 600;
      border: none;
    }
    QPushButton:pressed {
      color: #A0A0A0;
    }
  )");
  QObject::connect(close_button, &QPushButton::clicked, [=]() {
    emit closeSettings();
  });
  heading->addWidget(close_button);
  frame->addLayout(heading);


  auto *subheading = new QLabel(tr("manage at connect.comma.ai"), this);
  subheading->setStyleSheet("color: #A0A0A0; font-size: 48px; font-weight: 500;");
  frame->addWidget(subheading);


  current_widget = new DestinationWidget(this);
  QObject::connect(current_widget, &ClickableWidget::clicked, [=]() {
    params.remove("NavDestination");
    updateCurrentRoute();
  });
  frame->addWidget(current_widget);
  frame->addWidget(horizontal_line());


  QWidget *destinations_container = new QWidget(this);
  destinations_layout = new QVBoxLayout(destinations_container);
  destinations_layout->setContentsMargins(0, 0, 0, 0);
  destinations_layout->setSpacing(20);
  ScrollView *destinations_scroller = new ScrollView(destinations_container, this);
  frame->addWidget(destinations_scroller);


  // TODO: remove this
  cur_destinations = R"([
    {
      "save_type": "favorite",
      "label": "home",
      "place_name": "Home",
      "place_details": "123 Main St, San Francisco, CA 94103, USA"
    },
    {
      "save_type": "favorite",
      "place_name": "Target",
      "place_details": "456 Market St, San Francisco, CA 94103, USA"
    },
    {
      "save_type": "recent",
      "place_name": "Whole Foods",
      "place_details": "789 Mission St, San Francisco, CA 94103, USA"
    },
    {
      "save_type": "recent",
      "place_name": "Safeway",
      "place_details": "101 4th St, San Francisco, CA 94103, USA"
    }
  ])";

  if (auto dongle_id = getDongleId()) {
    // Fetch favorite and recent locations
    {
      QString url = CommaApi::BASE_URL + "/v1/navigation/" + *dongle_id + "/locations";
      RequestRepeater* repeater = new RequestRepeater(this, url, "ApiCache_NavDestinations", 30, true);
      QObject::connect(repeater, &RequestRepeater::requestDone, this, &MapSettings::parseResponse);
    }

    // Destination set while offline
    {
      QString url = CommaApi::BASE_URL + "/v1/navigation/" + *dongle_id + "/next";
      RequestRepeater* repeater = new RequestRepeater(this, url, "", 10, true);
      HttpRequest* deleter = new HttpRequest(this);

      QObject::connect(repeater, &RequestRepeater::requestDone, [=](const QString &resp, bool success) {
        if (success && resp != "null") {
          if (params.get("NavDestination").empty()) {
            qWarning() << "Setting NavDestination from /next" << resp;
            params.put("NavDestination", resp.toStdString());
          } else {
            qWarning() << "Got location from /next, but NavDestination already set";
          }

          // Send DELETE to clear destination server side
          deleter->sendRequest(url, HttpRequest::Method::DELETE);
        }
      });
    }
  }

  setStyleSheet(R"(
    MapSettings {
      background-color: #333333;
      border: 1px solid red;
    }
  )");
}

void MapSettings::showEvent(QShowEvent *event) {
  updateCurrentRoute();
}

void MapSettings::updateCurrentRoute() {
  auto dest = QString::fromStdString(params.get("NavDestination"));
  if (dest.size()) {
    QJsonDocument doc = QJsonDocument::fromJson(dest.trimmed().toUtf8());
    if (doc.isNull()) {
      qDebug() << "JSON Parse failed on NavDestination" << dest;
      return;
    }
    auto destination = new NavDestination(doc.object());
    if (current_destination && *destination == *current_destination) return;
    current_destination = destination;
    current_widget->set(current_destination, true);
  } else {
    current_destination = nullptr;
    current_widget->unset("", true);
  }
  if (isVisible()) refresh();
}

void MapSettings::parseResponse(const QString &response, bool success) {
  if (!success) return;
  if (response == cur_destinations) return;
  cur_destinations = response;
  refresh();
}

void MapSettings::refresh() {
  bool has_home = false, has_work = false;
  auto destinations = std::vector<NavDestination*>();

  auto destinations_str = cur_destinations.trimmed();
  if (!destinations_str.isEmpty()) {
    QJsonDocument doc = QJsonDocument::fromJson(destinations_str.toUtf8());
    if (doc.isNull()) {
      qDebug() << "JSON Parse failed on navigation locations" << cur_destinations;
      return;
    }

    for (auto el : doc.array()) {
      auto destination = new NavDestination(el.toObject());

      // add home and work later if they are missing
      if (destination->isFavorite()) {
        if (destination->label == NAV_FAVORITE_LABEL_HOME) has_home = true;
        else if (destination->label == NAV_FAVORITE_LABEL_WORK) has_work = true;
      }

      // skip current destination
      if (current_destination && *destination == *current_destination) continue;

      destinations.push_back(destination);
    }
  }

  clearLayout(destinations_layout);

  // TODO: sort: home, work, favorites, recents
  // add favorites before recents
  for (auto &save_type : {NAV_TYPE_FAVORITE, NAV_TYPE_RECENT}) {
    for (auto destination : destinations) {
      if (destination->type != save_type) continue;

      auto widget = new DestinationWidget(this);
      widget->set(destination, false);

      QObject::connect(widget, &ClickableWidget::clicked, [=]() {
        navigateTo(destination->toJson());
        emit closeSettings();
      });

      destinations_layout->addWidget(widget);
    }
  }

  // add home and work if missing
  if (!has_home) {
    auto widget = new DestinationWidget(this);
    widget->unset(NAV_FAVORITE_LABEL_HOME);
    destinations_layout->insertWidget(0, widget);
  }
  if (!has_work) {
    auto widget = new DestinationWidget(this);
    widget->unset(NAV_FAVORITE_LABEL_WORK);
    destinations_layout->insertWidget(1, widget);
  }

  destinations_layout->addStretch();
  repaint();
}

void MapSettings::navigateTo(const QJsonObject &place) {
  QJsonDocument doc(place);
  params.put("NavDestination", doc.toJson().toStdString());
  updateCurrentRoute();
}
