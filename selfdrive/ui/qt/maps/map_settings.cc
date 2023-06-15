#include "map_settings.h"

#include <QDebug>

#include "common/util.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

static QString shorten(const QString &str, int max_len) {
  return str.size() > max_len ? str.left(max_len).trimmed() + "…" : str;
}

MapSettings::DestinationWidget::DestinationWidget(QWidget *parent) : QPushButton(parent) {
  setContentsMargins(0, 0, 0, 0);

  auto *frame = new QHBoxLayout(this);
  frame->setContentsMargins(32, 24, 32, 24);
  frame->setSpacing(40);

  icon = new QLabel(this);
  icon->setFixedSize(72, 72);
  frame->addWidget(icon);

  auto *inner_frame = new QVBoxLayout;
  inner_frame->setContentsMargins(0, 0, 0, 0);
  inner_frame->setSpacing(0);

  title = new QLabel(this);
  inner_frame->addWidget(title);

  subtitle = new QLabel(this);
  subtitle->setStyleSheet("color: #A0A0A0;");
  inner_frame->addWidget(subtitle);
  frame->addLayout(inner_frame);

  action = new QLabel(this);
  action->setObjectName("action");
  action->setStyleSheet("font-size: 60px; font-weight: 600; border: none;");
  frame->addWidget(action);

  setStyleSheet(R"(
    DestinationWidget {
      background-color: #292929;
      border: 1px solid #4DFFFFFF;
      border-radius: 10px;
      color: #FFFFFF;
      font-size: 40px;
      font-weight: 400;
    }
    DestinationWidget[current="true"] {
      border: 1px solid #80FFFFFF;
    }
    DestinationWidget:pressed {
      background-color: #3B3B3B;
    }
    DestinationWidget[current~="true"]:disabled {
      color: #808080;
    }

    #action {
      font-size: 60px;
    }
    DestinationWidget:pressed #action {
      color: #A0A0A0;
    }

    QPushButton {
      border: none;
    }
  )");
}

void MapSettings::DestinationWidget::set(NavDestination *destination,
                                         bool current) {
  setProperty("current", current);

  auto title_text = destination->name;
  auto subtitle_text = destination->details;
  auto icon_pixmap = NAV_ICON_RECENT;

  if (destination->type == NAV_TYPE_FAVORITE) {
    title_text = destination->label;
    subtitle_text = destination->name + " " + destination->details;
    if (destination->label == NAV_FAVORITE_LABEL_HOME) {
      icon_pixmap = NAV_ICON_HOME;
    } else if (destination->label == NAV_FAVORITE_LABEL_WORK) {
      icon_pixmap = NAV_ICON_WORK;
    } else {
      icon_pixmap = NAV_ICON_FAVORITE;
    }
  }

  // TODO: shorten
  title->setText(title_text);
  subtitle->setText(subtitle_text);
  subtitle->setVisible(true);

  // TODO: use pixmap
  action->setText(current ? "×" : "→");
  action->setVisible(true);
}

void MapSettings::DestinationWidget::clear(const QString &label) {
  title->setText(tr("No %1 location set").arg(label));
  subtitle->setVisible(false);
  action->setVisible(false);
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
  title->setStyleSheet("color: #FFFFFF; font-size: 48px; font-weight: 500;");
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

  current_container = new QWidget(this);
  auto *current_layout = new QVBoxLayout(current_container);
  current_layout->setContentsMargins(0, 0, 0, 0);
  current_layout->setSpacing(16);

  auto *current_title = new QLabel(tr("current destination"), this);
  current_title->setStyleSheet("color: #A0A0A0; font-size: 40px; font-weight: 500;");
  current_layout->addWidget(current_title);

  current_widget = new DestinationWidget(this);
  current_widget->setDisabled(true);
  current_layout->addWidget(current_widget);

  // QObject::connect(clear_button, &QPushButton::clicked, [=]() {
  //   params.remove("NavDestination");
  //   updateCurrentRoute();
  // });

  current_layout->addWidget(horizontal_line());

  // destinations_list = new QVBoxLayout;
  // QWidget *recent_widget = new LayoutWidget(recent_layout, this);
  // ScrollView *recent_scroller = new ScrollView(recent_widget, this);
  // frame->addWidget(recent_scroller);


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
  current_container->setVisible(false);
  clearLayout(recent_layout);
}

void MapSettings::updateCurrentRoute() {
  auto dest = QString::fromStdString(params.get("NavDestination"));
  QJsonDocument doc = QJsonDocument::fromJson(dest.trimmed().toUtf8());
  auto visible = dest.size() && !doc.isNull();
  if (visible) {
    current_destination = new NavDestination(doc.object());
    current_widget->set(current_destination, true);
  }
  current_container->setVisible(visible);
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
  for (auto &save_type : {NAV_TYPE_FAVORITE, NAV_TYPE_RECENT}) {
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
