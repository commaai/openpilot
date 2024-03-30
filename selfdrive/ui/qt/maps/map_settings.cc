#include "selfdrive/ui/qt/maps/map_settings.h"

#include <utility>

#include <QApplication>
#include <QDebug>

#include "common/util.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

static void swap(QJsonValueRef v1, QJsonValueRef v2) { std::swap(v1, v2); }

static bool locationEqual(const QJsonValue &v1, const QJsonValue &v2) {
  return v1["latitude"] == v2["latitude"] && v1["longitude"] == v2["longitude"];
}

static qint64 convertTimestampToEpoch(const QString &timestamp) {
  QDateTime dt = QDateTime::fromString(timestamp, Qt::ISODate);
  return dt.isValid() ? dt.toSecsSinceEpoch() : 0;
}

MapSettings::MapSettings(bool closeable, QWidget *parent) : QFrame(parent) {
  setContentsMargins(0, 0, 0, 0);
  setAttribute(Qt::WA_NoMousePropagation);

  auto *frame = new QVBoxLayout(this);
  frame->setContentsMargins(40, 40, 40, 0);
  frame->setSpacing(0);

  auto *heading_frame = new QHBoxLayout;
  heading_frame->setContentsMargins(0, 0, 0, 0);
  heading_frame->setSpacing(32);
  {
    if (closeable) {
      auto *close_btn = new QPushButton("←");
      close_btn->setStyleSheet(R"(
        QPushButton {
          color: #FFFFFF;
          font-size: 100px;
          padding-bottom: 8px;
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
      QObject::connect(close_btn, &QPushButton::clicked, [=]() { emit closeSettings(); });
      // TODO: read map_on_left from ui state
      heading_frame->addWidget(close_btn);
    }

    auto *heading = new QVBoxLayout;
    heading->setContentsMargins(0, 0, 0, 0);
    heading->setSpacing(16);
    {
      auto *title = new QLabel(tr("NAVIGATION"), this);
      title->setStyleSheet("color: #FFFFFF; font-size: 54px; font-weight: 600;");
      heading->addWidget(title);

      auto *subtitle = new QLabel(tr("Manage at connect.comma.ai"), this);
      subtitle->setStyleSheet("color: #A0A0A0; font-size: 40px; font-weight: 300;");
      heading->addWidget(subtitle);
    }
    heading_frame->addLayout(heading, 1);
  }
  frame->addLayout(heading_frame);
  frame->addSpacing(32);

  current_widget = new DestinationWidget(this);
  QObject::connect(current_widget, &DestinationWidget::actionClicked,
                   []() { NavManager::instance()->setCurrentDestination({}); });
  frame->addWidget(current_widget);
  frame->addSpacing(32);

  QWidget *destinations_container = new QWidget(this);
  destinations_layout = new QVBoxLayout(destinations_container);
  destinations_layout->setContentsMargins(0, 32, 0, 32);
  destinations_layout->setSpacing(20);
  destinations_layout->addWidget(home_widget = new DestinationWidget(this));
  destinations_layout->addWidget(work_widget = new DestinationWidget(this));
  QObject::connect(home_widget, &DestinationWidget::navigateTo, this, &MapSettings::navigateTo);
  QObject::connect(work_widget, &DestinationWidget::navigateTo, this, &MapSettings::navigateTo);
  destinations_layout->addStretch();

  ScrollView *destinations_scroller = new ScrollView(destinations_container, this);
  destinations_scroller->setFrameShape(QFrame::NoFrame);
  frame->addWidget(destinations_scroller);

  setStyleSheet("MapSettings { background-color: #333333; }");
  QObject::connect(NavManager::instance(), &NavManager::updated, this, &MapSettings::refresh);
}

void MapSettings::showEvent(QShowEvent *event) {
  refresh();
}

void MapSettings::refresh() {
  if (!isVisible()) return;

  setUpdatesEnabled(false);

  auto get_w = [this](int i) {
    auto w = i < widgets.size() ? widgets[i] : widgets.emplace_back(new DestinationWidget);
    if (!w->parentWidget()) {
      destinations_layout->insertWidget(destinations_layout->count() - 1, w);
      QObject::connect(w, &DestinationWidget::navigateTo, this, &MapSettings::navigateTo);
    }
    return w;
  };

  const auto current_dest = NavManager::instance()->currentDestination();
  if (!current_dest.isEmpty()) {
    current_widget->set(current_dest, true);
  } else {
    current_widget->unset("", true);
  }
  home_widget->unset(NAV_FAVORITE_LABEL_HOME);
  work_widget->unset(NAV_FAVORITE_LABEL_WORK);

  int n = 0;
  for (auto location : NavManager::instance()->currentLocations()) {
    DestinationWidget *w = nullptr;
    auto dest = location.toObject();
    if (dest["save_type"].toString() == NAV_TYPE_FAVORITE) {
      auto label = dest["label"].toString();
      if (label == NAV_FAVORITE_LABEL_HOME) w = home_widget;
      if (label == NAV_FAVORITE_LABEL_WORK) w = work_widget;
    }
    w = w ? w : get_w(n++);
    w->set(dest, false);
    w->setVisible(!locationEqual(dest, current_dest));
  }
  for (; n < widgets.size(); ++n) widgets[n]->setVisible(false);

  setUpdatesEnabled(true);
}

void MapSettings::navigateTo(const QJsonObject &place) {
  NavManager::instance()->setCurrentDestination(place);
  emit closeSettings();
}

DestinationWidget::DestinationWidget(QWidget *parent) : QPushButton(parent) {
  setContentsMargins(0, 0, 0, 0);

  auto *frame = new QHBoxLayout(this);
  frame->setContentsMargins(32, 24, 32, 24);
  frame->setSpacing(32);

  icon = new QLabel(this);
  icon->setAlignment(Qt::AlignCenter);
  icon->setFixedSize(96, 96);
  icon->setObjectName("icon");
  frame->addWidget(icon);

  auto *inner_frame = new QVBoxLayout;
  inner_frame->setContentsMargins(0, 0, 0, 0);
  inner_frame->setSpacing(0);
  {
    title = new ElidedLabel(this);
    title->setAttribute(Qt::WA_TransparentForMouseEvents);
    inner_frame->addWidget(title);

    subtitle = new ElidedLabel(this);
    subtitle->setAttribute(Qt::WA_TransparentForMouseEvents);
    subtitle->setObjectName("subtitle");
    inner_frame->addWidget(subtitle);
  }
  frame->addLayout(inner_frame, 1);

  action = new QPushButton(this);
  action->setFixedSize(96, 96);
  action->setObjectName("action");
  action->setStyleSheet("font-size: 65px; font-weight: 600;");
  QObject::connect(action, &QPushButton::clicked, this, &QPushButton::clicked);
  QObject::connect(action, &QPushButton::clicked, this,  &DestinationWidget::actionClicked);
  frame->addWidget(action);

  setFixedHeight(164);
  setStyleSheet(R"(
    DestinationWidget { background-color: #202123; border-radius: 10px; }
    QLabel { color: #FFFFFF; font-size: 48px; font-weight: 400; }
    #icon { background-color: #3B4356; border-radius: 48px; }
    #subtitle { color: #9BA0A5; }
    #action { border: none; border-radius: 48px; color: #FFFFFF; padding-bottom: 4px; }

    /* current destination */
    [current="true"] { background-color: #E8E8E8; }
    [current="true"] QLabel { color: #000000; }
    [current="true"] #icon { background-color: #42906B; }
    [current="true"] #subtitle { color: #333333; }
    [current="true"] #action { color: #202123; }

    /* no saved destination */
    [set="false"] QLabel { color: #9BA0A5; }
    [current="true"][set="false"] QLabel { color: #A0000000; }

    /* pressed */
    [current="false"]:pressed { background-color: #18191B; }
    [current="true"] #action:pressed { background-color: #D6D6D6; }
  )");
  QObject::connect(this, &QPushButton::clicked, [this]() { if (!dest.isEmpty()) emit navigateTo(dest); });
}

void DestinationWidget::set(const QJsonObject &destination, bool current) {
  if (dest == destination) return;

  dest = destination;
  setProperty("current", current);
  setProperty("set", true);

  auto icon_pixmap = current ? icons().directions : icons().recent;
  auto title_text = destination["place_name"].toString();
  auto subtitle_text = destination["place_details"].toString();

  if (destination["save_type"] == NAV_TYPE_FAVORITE) {
    if (destination["label"] == NAV_FAVORITE_LABEL_HOME) {
      icon_pixmap = icons().home;
      subtitle_text = title_text + ", " + subtitle_text;
      title_text = tr("Home");
    } else if (destination["label"] == NAV_FAVORITE_LABEL_WORK) {
      icon_pixmap = icons().work;
      subtitle_text = title_text + ", " + subtitle_text;
      title_text = tr("Work");
    } else {
      icon_pixmap = icons().favorite;
    }
  }

  icon->setPixmap(icon_pixmap);

  title->setText(title_text);
  subtitle->setText(subtitle_text);
  subtitle->setVisible(true);

  // TODO: use pixmap
  action->setAttribute(Qt::WA_TransparentForMouseEvents, !current);
  action->setText(current ? "×" : "→");
  action->setVisible(true);

  setStyleSheet(styleSheet());
}

void DestinationWidget::unset(const QString &label, bool current) {
  dest = {};
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
  setVisible(true);
}

// singleton NavManager

NavManager *NavManager::instance() {
  static NavManager *request = new NavManager(qApp);
  return request;
}

NavManager::NavManager(QObject *parent) : QObject(parent) {
  locations = QJsonDocument::fromJson(params.get("NavPastDestinations").c_str()).array();
  current_dest = QJsonDocument::fromJson(params.get("NavDestination").c_str()).object();
  if (auto dongle_id = getDongleId()) {
    {
      // Fetch favorite and recent locations
      QString url = CommaApi::BASE_URL + "/v1/navigation/" + *dongle_id + "/locations";
      RequestRepeater *repeater = new RequestRepeater(this, url, "ApiCache_NavDestinations", 30, true);
      QObject::connect(repeater, &RequestRepeater::requestDone, this, &NavManager::parseLocationsResponse);
    }
    {
      auto param_watcher = new ParamWatcher(this);
      QObject::connect(param_watcher, &ParamWatcher::paramChanged, this, &NavManager::updated);

      // Destination set while offline
      QString url = CommaApi::BASE_URL + "/v1/navigation/" + *dongle_id + "/next";
      HttpRequest *deleter = new HttpRequest(this);
      RequestRepeater *repeater = new RequestRepeater(this, url, "", 10, true);
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

        // athena can set destination at any time
        param_watcher->addParam("NavDestination");
        current_dest = QJsonDocument::fromJson(params.get("NavDestination").c_str()).object();
        emit updated();
      });
    }
  }
}

void NavManager::parseLocationsResponse(const QString &response, bool success) {
  if (!success || response == prev_response) return;

  prev_response = response;
  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qWarning() << "JSON Parse failed on navigation locations" << response;
    return;
  }

  // set last activity time.
  auto remote_locations = doc.array();
  for (QJsonValueRef loc : remote_locations) {
    auto obj = loc.toObject();
    auto serverTime = convertTimestampToEpoch(obj["modified"].toString());
    obj.insert("time", qMax(serverTime, getLastActivity(obj)));
    loc = obj;
  }

  locations = remote_locations;
  sortLocations();
  emit updated();
}

void NavManager::sortLocations() {
  // Sort: alphabetical FAVORITES, and then most recent.
  // We don't need to care about the ordering of HOME and WORK. DestinationWidget always displays them at the top.
  std::stable_sort(locations.begin(), locations.end(), [](const QJsonValue &a, const QJsonValue &b) {
    if (a["save_type"] == NAV_TYPE_FAVORITE || b["save_type"] == NAV_TYPE_FAVORITE) {
      return (std::tuple(a["save_type"].toString(), a["place_name"].toString()) <
              std::tuple(b["save_type"].toString(), b["place_name"].toString()));
    } else {
      return a["time"].toVariant().toLongLong() > b["time"].toVariant().toLongLong();
    }
  });

  write_param_future = std::async(std::launch::async, [destinations = QJsonArray(locations)]() {
    Params().put("NavPastDestinations", QJsonDocument(destinations).toJson().toStdString());
  });
}

qint64 NavManager::getLastActivity(const QJsonObject &loc) const {
  qint64 last_activity = 0;
  auto it = std::find_if(locations.begin(), locations.end(),
                         [&loc](const QJsonValue &l) { return locationEqual(loc, l); });
  if (it != locations.end()) {
    auto tm = it->toObject().value("time");
    if (!tm.isUndefined() && !tm.isNull()) {
      last_activity = tm.toVariant().toLongLong();
    }
  }
  return last_activity;
}

void NavManager::setCurrentDestination(const QJsonObject &loc) {
  current_dest = loc;
  if (!current_dest.isEmpty()) {
    current_dest["time"] = QDateTime::currentSecsSinceEpoch();
    auto it = std::find_if(locations.begin(), locations.end(),
                           [&loc](const QJsonValue &l) { return locationEqual(loc, l); });
    if (it != locations.end()) {
      *it = current_dest;
      sortLocations();
    }
    params.put("NavDestination", QJsonDocument(current_dest).toJson().toStdString());
  } else {
    params.remove("NavDestination");
  }
  emit updated();
}
