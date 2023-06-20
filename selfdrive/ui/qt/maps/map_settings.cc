#include "map_settings.h"

#include <vector>
#include <QDebug>

#include "common/util.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

static QString shorten(const QString &str, int max_len) {
  return str.size() > max_len ? str.left(max_len).trimmed() + "…" : str;
}

DestinationWidget::DestinationWidget(QWidget *parent) : ClickableWidget(parent) {
  setContentsMargins(0, 0, 0, 0);

  auto *frame = new QHBoxLayout(this);
  frame->setContentsMargins(32, 24, 32, 24);
  frame->setSpacing(32);

  icon = new QLabel(this);
  icon->setFixedSize(96, 96);
  frame->addWidget(icon);

  auto *inner_frame = new QVBoxLayout;
  inner_frame->setContentsMargins(0, 0, 0, 0);
  inner_frame->setSpacing(0);
  {
    title = new ElidedLabel(this);
    inner_frame->addWidget(title);

    subtitle = new ElidedLabel(this);
    subtitle->setStyleSheet("color: #A0A0A0;");
    inner_frame->addWidget(subtitle);
  }
  frame->addLayout(inner_frame, 1);

  action = new QLabel(this);
  action->setObjectName("action");
  action->setStyleSheet("font-size: 60px; font-weight: 600; border: none;");
  frame->addWidget(action);

  setFixedHeight(164);
  setStyleSheet(R"(
    /* default styles */
    DestinationWidget {
      background-color: #292929;
      border: 1px solid #4DFFFFFF;
      border-radius: 10px;
    }
    DestinationWidget QLabel {
      background-color: transparent;
      color: #FFFFFF;
      font-size: 48px;
      font-weight: 400;
    }

    /* on press */
    DestinationWidget:pressed {
      background-color: #3B3B3B;
    }
    DestinationWidget:pressed #action {
      color: #A0A0A0;
    }

    /* current destination */
    DestinationWidget[current=true] {
      background-color: #162440;
      border: 1px solid #80FFFFFF;
    }

    /* no saved destination */
    DestinationWidget[set=false] QLabel {
      color: #80FFFFFF;
    }
  )");
}

void DestinationWidget::set(NavDestination *destination, bool current) {
  setProperty("current", current);
  setProperty("set", true);

  auto title_text = destination->name();
  auto subtitle_text = destination->details();
  auto icon_pixmap = icons().recent;

  if (destination->isFavorite()) {
    if (destination->label() == NAV_FAVORITE_LABEL_HOME) {
      title_text = tr("Home");
      icon_pixmap = icons().home;
    } else if (destination->label() == NAV_FAVORITE_LABEL_WORK) {
      title_text = tr("Work");
      icon_pixmap = icons().work;
    } else {
      icon_pixmap = icons().favorite;
    }
  }

  if (current) {
    icon_pixmap = icons().directions;
  }

  // TODO: onroad and offroad have different dimensions
  title->setText(shorten(title_text, 25));
  subtitle->setText(shorten(subtitle_text, 25));
  subtitle->setVisible(true);
  icon->setPixmap(icon_pixmap);

  // TODO: use pixmap
  action->setText(current ? "×" : "→");
  action->setVisible(true);

  setStyleSheet(styleSheet());
}

void DestinationWidget::unset(const QString &label, bool current) {
  setProperty("current", current);
  setProperty("set", false);

  if (label.isEmpty()) {
    icon->setPixmap(icons().directions);
    title->setText(tr("No destination set"));
  } else {
    QString title_text = label == NAV_FAVORITE_LABEL_HOME ? tr("home") : tr("work");
    icon->setPixmap(label == NAV_FAVORITE_LABEL_HOME ? icons().home : icons().work);
    title->setText(tr("No %1 location set").arg(title_text));
  }

  subtitle->setVisible(false);
  action->setVisible(false);

  setStyleSheet(styleSheet());
}

MapSettings::MapSettings(bool closeable, QWidget *parent)
    : QFrame(parent), current_destination(nullptr) {
  QSize icon_size(100, 100);
  close_icon = loadPixmap("../assets/icons/close.svg", icon_size);

  setContentsMargins(0, 0, 0, 0);

  auto *frame = new QVBoxLayout(this);
  frame->setContentsMargins(40, 40, 40, 0);
  frame->setSpacing(0);

  auto *heading_frame = new QHBoxLayout;
  heading_frame->setContentsMargins(0, 0, 0, 0);
  heading_frame->setSpacing(32);
  {
    if (closeable) {
      auto *close_btn = new QPushButton("×");
      close_btn->setStyleSheet(R"(
        QPushButton {
          color: #FFFFFF;
          font-size: 100px;
          padding-bottom: 14px;
          border 1px grey solid;
          border-radius: 70px;
          background-color: #292929;
          font-weight: 500;
        }
        QPushButton:pressed {
          background-color: #3B3B3B;
        }
      )");
      close_btn->setFixedSize(140, 140);
      QObject::connect(close_btn, &QPushButton::clicked,
                       [=]() { emit closeSettings(); });
      // TODO: read map_on_left from ui state
      heading_frame->addWidget(close_btn);
    }

    auto *heading = new QVBoxLayout;
    heading->setContentsMargins(0, 0, 0, 0);
    heading->setSpacing(16);
    {
      auto *title = new QLabel(tr("comma navigation"), this);
      title->setStyleSheet(
          "color: #FFFFFF; font-size: 56px; font-weight: 500;");
      heading->addWidget(title);

      auto *subtitle = new QLabel(tr("manage at connect.comma.ai"), this);
      subtitle->setStyleSheet("color: #A0A0A0; font-size: 48px; font-weight: 500;");
      heading->addWidget(subtitle);
    }
    heading_frame->addLayout(heading, 1);
  }
  frame->addLayout(heading_frame);
  frame->addSpacing(32);

  current_widget = new DestinationWidget(this);
  QObject::connect(current_widget, &ClickableWidget::clicked, [=]() {
    if (!current_destination) return;
    params.remove("NavDestination");
    updateCurrentRoute();
  });
  frame->addWidget(current_widget);
  frame->addSpacing(32);
  frame->addWidget(horizontal_line());

  QWidget *destinations_container = new QWidget(this);
  destinations_layout = new QVBoxLayout(destinations_container);
  destinations_layout->setContentsMargins(0, 32, 0, 32);
  destinations_layout->setSpacing(20);
  ScrollView *destinations_scroller = new ScrollView(destinations_container, this);
  frame->addWidget(destinations_scroller);

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
        qDebug() << "/next" << success << resp;
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
      qWarning() << "JSON Parse failed on NavDestination" << dest;
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
      qWarning() << "JSON Parse failed on navigation locations" << cur_destinations;
      return;
    }

    for (auto el : doc.array()) {
      auto destination = new NavDestination(el.toObject());

      // add home and work later if they are missing
      if (destination->isFavorite()) {
        if (destination->label() == NAV_FAVORITE_LABEL_HOME) has_home = true;
        else if (destination->label() == NAV_FAVORITE_LABEL_WORK) has_work = true;
      }

      // skip current destination
      if (current_destination && *destination == *current_destination) continue;

      destinations.push_back(destination);
    }
  }

  // TODO: should we build a new layout and swap it in?
  clearLayout(destinations_layout);

  // Sort in order: HOME, WORK, and then descending-alphabetical FAVORITES, RECENTS
  std::sort(destinations.begin(), destinations.end(), [](const NavDestination *a, const NavDestination *b) {
    if (a->isFavorite() && b->isFavorite()) {
      if (a->label() == NAV_FAVORITE_LABEL_HOME) return true;
      else if (b->label() == NAV_FAVORITE_LABEL_HOME) return false;
      else if (a->label() == NAV_FAVORITE_LABEL_WORK) return true;
      else if (b->label() == NAV_FAVORITE_LABEL_WORK) return false;
      else if (a->label() != b->label()) return a->label() < b->label();
    }
    else if (a->isFavorite()) return true;
    else if (b->isFavorite()) return false;
    return a->name() < b->name();
  });

  for (auto destination : destinations) {
    auto widget = new DestinationWidget(this);
    widget->set(destination, false);

    QObject::connect(widget, &ClickableWidget::clicked, [=]() {
      navigateTo(destination->toJson());
      emit closeSettings();
    });

    destinations_layout->addWidget(widget);
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
    // TODO: refactor to remove this hack
    int index = !has_home || (current_destination && current_destination->isFavorite() && current_destination->label() == NAV_FAVORITE_LABEL_HOME) ? 0 : 1;
    destinations_layout->insertWidget(index, widget);
  }

  destinations_layout->addStretch();
  repaint();
}

void MapSettings::navigateTo(const QJsonObject &place) {
  QJsonDocument doc(place);
  params.put("NavDestination", doc.toJson().toStdString());
  updateCurrentRoute();
}
