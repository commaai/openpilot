#include "map_settings.h"

#include <QDebug>

#include "selfdrive/ui/qt/request_repeater.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/common/util.h"

static QString shorten(const QString &str, int max_len) {
  return str.size() > max_len ? str.left(max_len).trimmed() + "…" : str;
}

MapPanel::MapPanel(QWidget* parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  Params params = Params();

  // Home
  QHBoxLayout *home_layout = new QHBoxLayout;
  home_button = new QPushButton;
  home_button->setIconSize(QSize(200, 200));
  home_layout->addWidget(home_button);

  home_address = new QLabel;
  home_address->setWordWrap(true);
  home_layout->addSpacing(30);
  home_layout->addWidget(home_address);
  home_layout->addStretch();

  // Work
  QHBoxLayout *work_layout = new QHBoxLayout;
  work_button = new QPushButton;
  work_button->setIconSize(QSize(200, 200));
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
  main_layout->addSpacing(50);
  main_layout->addWidget(horizontal_line());
  main_layout->addSpacing(50);

  // Recents
  QLabel *recent = new QLabel("Recent");
  recent->setStyleSheet(R"(font-size: 55px;)");
  main_layout->addWidget(recent);

  main_layout->addSpacing(20);

  recent_layout = new QVBoxLayout;
  main_layout->addLayout(recent_layout);

  // Settings
  main_layout->addSpacing(50);
  main_layout->addWidget(horizontal_line());
  main_layout->addWidget(new ParamControl("NavSettingTime24h",
                                    "Show ETA in 24h format",
                                    "Use 24h format instead of am/pm",
                                    "",
                                    this));
  main_layout->addStretch();

  clear();

  std::string dongle_id = Params().get("DongleId");
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
      RequestRepeater* repeater = new RequestRepeater(this, QString::fromStdString(url), "", 10);

      QObject::connect(repeater, &RequestRepeater::receivedResponse, [](QString resp) {
        auto params = Params();
        if (resp != "null" && params.get("NavDestination").empty()) {
          params.put("NavDestination", resp.toStdString());
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
  for (auto location : doc.array()) {
    auto obj = location.toObject();

    auto type = obj["save_type"].toString();
    auto label = obj["label"].toString();
    auto name = obj["place_name"].toString();
    auto details = shorten(obj["place_details"].toString(), 30);

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
      layout->setContentsMargins(40, 10, 40, 10);

      QLabel *recent_label = new QLabel(name + " " + details);
      recent_label->setStyleSheet(R"(font-size: 50px; color: #9c9c9c)");

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

  if (!has_recents) {
    QLabel *no_recents = new QLabel("  no recent destinations");
    no_recents->setStyleSheet(R"(font-size: 50px; color: #9c9c9c)");
    recent_layout->addWidget(no_recents);
  }
}

void MapPanel::navigateTo(const QJsonObject &place) {
  QJsonDocument doc(place);
  Params().put("NavDestination", doc.toJson().toStdString());
}
