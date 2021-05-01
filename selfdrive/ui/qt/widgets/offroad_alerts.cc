#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonObject>
#include <QJsonDocument>

#include "offroad_alerts.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/common/util.h"

void Alert::refresh() {
  if (alerts.empty()) {
    QString json = QString::fromStdString(util::read_file("../controls/lib/alerts_offroad.json"));
    QJsonObject obj = QJsonDocument::fromJson(json.toUtf8()).object();
    for (auto &k : obj.keys()) {
      auto &a = alerts.emplace_back();
      a.key = k.toStdString();
      a.severity = obj[k].toObject()["severity"].toInt();
      a.label = nullptr;
    }
  }

  alertCount = 0;
  updateAvailable = params.getBool("UpdateAvailable");
  for (auto &a : alerts) {
    auto bytes = params.get(a.key);
    if (!bytes.empty()) {
      QJsonDocument doc_par = QJsonDocument::fromJson(QByteArray(bytes.data(), bytes.size()));
      QJsonObject obj = doc_par.object();
      if (a.label == nullptr) {
        a.label = new QLabel();
        a.label->setStyleSheet("background-color: " + QString(a.severity ? "#E22C2C" : "#292929"));
        a.label->setMargin(60);
        a.label->setWordWrap(true);
      }
      a.label->setText(obj.value("text").toString());
      alertCount++;
    } else {
      if (a.label != nullptr) {
        delete a.label;
        a.label = nullptr;
      }
    }
  }
}

OffroadAlert::OffroadAlert(const Alert &alert, QWidget* parent) : QFrame(parent) {
  QVBoxLayout *layout = new QVBoxLayout();
  layout->setMargin(50);
  layout->setSpacing(30);

  QWidget *alerts_widget = new QWidget;
  QVBoxLayout *vbMain = new QVBoxLayout();
  alerts_widget->setLayout(vbMain);

  alerts_layout = new QVBoxLayout;
  alerts_layout->setMargin(0);
  alerts_layout->setSpacing(30);
  alerts_widget->setLayout(alerts_layout);
  alerts_widget->setStyleSheet("background-color: transparent;");

  vbMain->addLayout(alerts_layout);  
  vbMain->addStretch(1);

  // release notes
  releaseNotes.setWordWrap(true);
  releaseNotes.setVisible(false);
  releaseNotes.setStyleSheet("font-size: 48px;");
  releaseNotes.setAlignment(Qt::AlignTop);

  releaseNotesScroll = new ScrollView(&releaseNotes, this);
  layout->addWidget(releaseNotesScroll);

  alertsScroll = new ScrollView(alerts_widget, this);
  layout->addWidget(alertsScroll);

  // bottom footer, dismiss + reboot buttons
  QHBoxLayout *footer_layout = new QHBoxLayout();
  layout->addLayout(footer_layout);

  QPushButton *dismiss_btn = new QPushButton("Dismiss");
  dismiss_btn->setFixedSize(400, 125);
  footer_layout->addWidget(dismiss_btn, 0, Qt::AlignBottom | Qt::AlignLeft);
  QObject::connect(dismiss_btn, &QPushButton::released, this, &OffroadAlert::closeAlerts);

  rebootBtn.setText("Reboot and Update");
  rebootBtn.setFixedSize(600, 125);
  rebootBtn.setVisible(false);
  footer_layout->addWidget(&rebootBtn, 0, Qt::AlignBottom | Qt::AlignRight);
  QObject::connect(&rebootBtn, &QPushButton::released, [=]() { Hardware::reboot(); });

  setLayout(layout);
  setStyleSheet(R"(
    * {
      font-size: 48px;
      color: white;
    }
    QFrame {
      border-radius: 30px;
      background-color: #393939;
    }
    QPushButton {
      color: black;
      font-weight: 500;
      border-radius: 30px;
      background-color: white;
    }
  )");

  updateAlerts(alert);
}

void OffroadAlert::updateAlerts(const Alert &alert) {
  for (const auto &a : alert.alerts) {
    if (a.label != nullptr && a.label->parent() == nullptr) {
      alerts_layout->addWidget(a.label);
    }
  }

  rebootBtn.setVisible(alert.updateAvailable);
  releaseNotesScroll->setVisible(alert.updateAvailable);
  if (alert.updateAvailable) {
    releaseNotes.setText(QString::fromStdString(Params().get("ReleaseNotes")));
  }
  alertsScroll->setVisible(!alert.updateAvailable);
}
