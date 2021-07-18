#include "map_settings.h"

#include <QDebug>

#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/scrollview.h"

static QString shorten(const QString &str, int max_len) {
  return str.size() > max_len ? str.left(max_len).trimmed() + "…" : str;
}

MapPanel::MapPanel(QWidget* parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  const int icon_size = 200;

  // Home
  QHBoxLayout *home_layout = new QHBoxLayout;
  home_button = new QPushButton;
  home_button->setIconSize(QSize(icon_size, icon_size));
  home_layout->addWidget(home_button);

  home_address = new QLabel;
  home_address->setWordWrap(true);
  home_layout->addSpacing(30);
  home_layout->addWidget(home_address);
  home_layout->addStretch();

  // Work
  QHBoxLayout *work_layout = new QHBoxLayout;
  work_button = new QPushButton;
  work_button->setIconSize(QSize(icon_size, icon_size));
  work_layout->addWidget(work_button);

  work_address = new QLabel;
  work_address->setWordWrap(true);
  work_layout->addSpacing(30);
  work_layout->addWidget(work_address);
  work_layout->addStretch();

  // Home & Work layout
  QHBoxLayout *home_work_layout = new QHBoxLayout;
  home_work_layout->addLayout(home_layout, 1);
  home_work_layout->addSpacing(50);
  home_work_layout->addLayout(work_layout, 1);

  main_layout->addLayout(home_work_layout);
  main_layout->addSpacing(20);
  main_layout->addWidget(horizontal_line());
  main_layout->addSpacing(20);

  // Recents
  recent_layout = new QVBoxLayout;
  QWidget *recent_widget = new LayoutWidget(recent_layout, this);
  ScrollView *recent_scroller = new ScrollView(recent_widget, this);
  main_layout->addWidget(recent_scroller, 1);

  clear();

  std::string dongle_id = params.get("DongleId");
  if (util::is_valid_dongle_id(dongle_id)) {
    // Fetch favorite and recent locations
    {
      std::string url = "https://api.commadotai.com/v1/navigation/" + dongle_id + "/locations";
      RequestRepeater* repeater = new RequestRepeater(this, QString::fromStdString(url), "ApiCache_NavDestinations", 30);
      QObject::connect(repeater, &RequestRepeater::receivedResponse, this, &MapPanel::parseResponse);
    }

    // Destination set while offline
    {
      std::string url = "https://api.commadotai.com/v1/navigation/" + dongle_id + "/next";
      RequestRepeater* repeater = new RequestRepeater(this, QString::fromStdString(url), "", 10, true);

      QObject::connect(repeater, &RequestRepeater::receivedResponse, [](QString resp) {
        auto params = Params();
        if (resp != "null") {
          if (params.get("NavDestination").empty()) {
            qWarning() << "Setting NavDestination from /next" << resp;
            params.put("NavDestination", resp.toStdString());
          } else {
            qWarning() << "Got location from /next, but NavDestination already set";
          }
        }
      });
    }
  }
}

void MapPanel::clear() {
  home_button->setIcon(QPixmap("../assets/navigation/home_inactive.png"));
  home_address->setStyleSheet(R"(font-size: 50px; color: grey;)");
  home_address->setText("No home\nlocation set");
  home_button->disconnect();

  work_button->setIcon(QPixmap("../assets/navigation/work_inactive.png"));
  work_address->setStyleSheet(R"(font-size: 50px; color: grey;)");
  work_address->setText("No work\nlocation set");
  work_button->disconnect();

  clearLayout(recent_layout);
}


void MapPanel::parseResponse(const QString &response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed on navigation locations";
    return;
  }

  clear();

  bool has_recents = false;
  for (auto &save_type: {"favorite", "recent"}) {
    for (auto location : doc.array()) {
      auto obj = location.toObject();

      auto type = obj["save_type"].toString();
      auto label = obj["label"].toString();
      auto name = obj["place_name"].toString();
      auto details = obj["place_details"].toString();

      if (type != save_type) continue;

      if (type == "favorite" && label == "home") {
        home_address->setText(name);
        home_address->setStyleSheet(R"(font-size: 50px; color: white;)");
        home_button->setIcon(QPixmap("../assets/navigation/home.png"));
        QObject::connect(home_button, &QPushButton::clicked, [=]() {
          navigateTo(obj);
          emit closeSettings();
        });
      } else if (type == "favorite" && label == "work") {
        work_address->setText(name);
        work_address->setStyleSheet(R"(font-size: 50px; color: white;)");
        work_button->setIcon(QPixmap("../assets/navigation/work.png"));
        QObject::connect(work_button, &QPushButton::clicked, [=]() {
          navigateTo(obj);
          emit closeSettings();
        });
      } else {
        ClickableWidget *widget = new ClickableWidget;
        QHBoxLayout *layout = new QHBoxLayout(widget);
        layout->setContentsMargins(15, 14, 40, 14);

        QLabel *star = new QLabel("★");
        auto sp = star->sizePolicy();
        sp.setRetainSizeWhenHidden(true);
        star->setSizePolicy(sp);

        star->setVisible(type == "favorite");
        star->setStyleSheet(R"(font-size: 60px;)");
        layout->addWidget(star);
        layout->addSpacing(10);


        QLabel *recent_label = new QLabel(shorten(name + " " + details, 45));
        recent_label->setStyleSheet(R"(font-size: 50px;)");

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
    QLabel *no_recents = new QLabel("no recent destinations");
    no_recents->setStyleSheet(R"(font-size: 50px; color: #9c9c9c)");
    recent_layout->addWidget(no_recents);
  }

  recent_layout->addStretch();
}

void MapPanel::navigateTo(const QJsonObject &place) {
  QJsonDocument doc(place);
  params.put("NavDestination", doc.toJson().toStdString());
}
