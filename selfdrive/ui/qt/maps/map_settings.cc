#include "map_settings.h"

#include <QDebug>

#include "common/util.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

static QString shorten(const QString &str, int max_len) {
  return str.size() > max_len ? str.left(max_len).trimmed() + "…" : str;
}

MapSettings::MapSettings(QWidget* parent) : QFrame(parent) {
  setContentsMargins(36, 36, 36, 36);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(20);

  // Home & Work layout
  QHBoxLayout *home_work_layout = new QHBoxLayout;
  {
    // Home
    home_button = new QPushButton;
    home_button->setIconSize(QSize(MAP_PANEL_ICON_SIZE, MAP_PANEL_ICON_SIZE));
    home_address = new QLabel;
    home_address->setWordWrap(true);

    QHBoxLayout *home_layout = new QHBoxLayout;
    home_layout->addWidget(home_button);
    home_layout->addSpacing(30);
    home_layout->addWidget(home_address);
    home_layout->addStretch();

    // Work
    work_button = new QPushButton;
    work_button->setIconSize(QSize(MAP_PANEL_ICON_SIZE, MAP_PANEL_ICON_SIZE));
    work_address = new QLabel;
    work_address->setWordWrap(true);

    QHBoxLayout *work_layout = new QHBoxLayout;
    work_layout->addWidget(work_button);
    work_layout->addSpacing(30);
    work_layout->addWidget(work_address);
    work_layout->addStretch();

    home_work_layout->addLayout(home_layout, 1);
    home_work_layout->addSpacing(50);
    home_work_layout->addLayout(work_layout, 1);
  }

  main_layout->addLayout(home_work_layout);
  main_layout->addWidget(horizontal_line());

  // Current route
  current_widget = new QWidget(this);
  {
    QVBoxLayout *current_layout = new QVBoxLayout(current_widget);
    current_layout->setContentsMargins(0, 0, 0, 0);
    current_layout->setSpacing(20);

    QHBoxLayout *heading_layout = new QHBoxLayout;
    heading_layout->setContentsMargins(0, 0, 0, 0);
    heading_layout->setSpacing(0);

    QLabel *title = new QLabel(tr("Current Destination"));
    title->setStyleSheet("font-size: 50px");
    heading_layout->addWidget(title);

    heading_layout->addStretch();

    QPushButton *clear_button = new QPushButton(tr("CLEAR"));
    clear_button->setStyleSheet(R"(
      QPushButton {
        border-radius: 10px;
        padding: 24px 40px;
        border-radius: 40px;
        font-size: 32px;
        font-weight: 500;
        color: #E4E4E4;
        background-color: #1AC4C4C4;
      }
      QPushButton:pressed {
        background-color: #33C4C4C4;
      }
    )");
    QObject::connect(clear_button, &QPushButton::clicked, [=]() {
      params.remove("NavDestination");
      updateCurrentRoute();
    });
    heading_layout->addWidget(clear_button);

    current_layout->addLayout(heading_layout);

    current_route = new QLabel("");
    current_route->setStyleSheet(R"(
      QLabel {
        font-size: 40px;
        color: #9C9C9C;
        font-weight: 400;
      }
    )");
    current_layout->addWidget(current_route);

    current_layout->addSpacing(10);
    current_layout->addWidget(horizontal_line());
  }
  main_layout->addWidget(current_widget);

  // Recents
  QLabel *recents_title = new QLabel(tr("Recent Destinations"));
  recents_title->setStyleSheet("font-size: 50px");
  main_layout->addWidget(recents_title);

  recent_layout = new QVBoxLayout;
  QWidget *recent_widget = new LayoutWidget(recent_layout, this);
  ScrollView *recent_scroller = new ScrollView(recent_widget, this);
  main_layout->addWidget(recent_scroller);


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

  clear();
  refresh();  // TODO: remove this

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
}

void MapSettings::showEvent(QShowEvent *event) {
  qDebug() << "MapSettings" << width() << height();
  updateCurrentRoute();
  refresh();
}

void MapSettings::clear() {
  home_button->setIcon(QPixmap("../assets/navigation/home_inactive.png"));
  home_address->setStyleSheet(R"(font-size: 40px; color: grey;)");
  home_address->setText(tr("No home\nlocation set"));
  home_button->disconnect();

  work_button->setIcon(QPixmap("../assets/navigation/work_inactive.png"));
  work_address->setStyleSheet(R"(font-size: 40px; color: grey;)");
  work_address->setText(tr("No work\nlocation set"));
  work_button->disconnect();

  clearLayout(recent_layout);
}

void MapSettings::updateCurrentRoute() {
  auto dest = QString::fromStdString(params.get("NavDestination"));
  QJsonDocument doc = QJsonDocument::fromJson(dest.trimmed().toUtf8());
  if (dest.size() && !doc.isNull()) {
    auto name = doc["place_name"].toString();
    auto details = doc["place_details"].toString();
    current_route->setText(shorten(name + " " + details, 42));
  }
  current_widget->setVisible(dest.size() && !doc.isNull());
}

void MapSettings::parseResponse(const QString &response, bool success) {
  if (!success) return;

  cur_destinations = response;
  if (isVisible()) {
    refresh();
  }
}

void MapSettings::refresh() {
  if (cur_destinations == prev_destinations) return;

  QJsonDocument doc = QJsonDocument::fromJson(cur_destinations.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on navigation locations";
    return;
  }

  prev_destinations = cur_destinations;
  clear();

  // add favorites before recents
  bool has_recents = false;
  for (auto &save_type: {NAV_TYPE_FAVORITE, NAV_TYPE_RECENT}) {
    for (auto location : doc.array()) {
      auto obj = location.toObject();

      auto type = obj["save_type"].toString();
      auto label = obj["label"].toString();
      auto name = obj["place_name"].toString();
      auto details = obj["place_details"].toString();

      if (type != save_type) continue;

      if (type == NAV_TYPE_FAVORITE && label == NAV_FAVORITE_LABEL_HOME) {
        home_address->setText(name);
        home_address->setStyleSheet(R"(font-size: 40px; color: white;)");
        home_button->setIcon(QPixmap("../assets/navigation/home.png"));
        QObject::connect(home_button, &QPushButton::clicked, [=]() {
          navigateTo(obj);
          emit closeSettings();
        });
      } else if (type == NAV_TYPE_FAVORITE && label == NAV_FAVORITE_LABEL_WORK) {
        work_address->setText(name);
        work_address->setStyleSheet(R"(font-size: 40px; color: white;)");
        work_button->setIcon(QPixmap("../assets/navigation/work.png"));
        QObject::connect(work_button, &QPushButton::clicked, [=]() {
          navigateTo(obj);
          emit closeSettings();
        });
      } else {
        ClickableWidget *widget = new ClickableWidget;
        QHBoxLayout *layout = new QHBoxLayout(widget);
        layout->setContentsMargins(15, 8, 40, 8);

        QLabel *star = new QLabel("★");
        auto sp = star->sizePolicy();
        sp.setRetainSizeWhenHidden(true);
        star->setSizePolicy(sp);

        star->setVisible(type == NAV_TYPE_FAVORITE);
        star->setStyleSheet(R"(font-size: 60px;)");
        layout->addWidget(star);
        layout->addSpacing(10);

        QLabel *recent_label = new QLabel(shorten(name + " " + details, 42));
        recent_label->setStyleSheet(R"(font-size: 32px;)");

        layout->addWidget(recent_label);
        layout->addStretch();

        QLabel *arrow = new QLabel("→");
        arrow->setStyleSheet(R"(font-size: 60px;)");
        layout->addWidget(arrow);

        widget->setStyleSheet(R"(
          .ClickableWidget {
            border-radius: 10px;
            border-width: 1px;
            border-style: solid;
            border-color: gray;
          }
          QWidget {
            background-color: #393939;
            color: #9c9c9c;
          }
        )");

        QObject::connect(widget, &ClickableWidget::clicked, [=]() {
          navigateTo(obj);
          emit closeSettings();
        });

        recent_layout->addWidget(widget);
        recent_layout->addSpacing(10);
        has_recents = true;
      }
    }
  }

  if (!has_recents) {
    QLabel *no_recents = new QLabel(tr("no recent destinations"));
    no_recents->setStyleSheet(R"(font-size: 40px; color: #9c9c9c)");
    recent_layout->addWidget(no_recents);
  }

  recent_layout->addStretch();
  repaint();
}

void MapSettings::navigateTo(const QJsonObject &place) {
  QJsonDocument doc(place);
  params.put("NavDestination", doc.toJson().toStdString());
  updateCurrentRoute();
}
