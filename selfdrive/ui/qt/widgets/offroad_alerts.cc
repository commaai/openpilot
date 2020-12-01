#include <QWidget>
#include <QLabel>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDebug>

#include "offroad_alerts.hpp"

#include "common/params.h"


void cleanLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()){
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      cleanLayout(childLayout);
    }
    delete item;
  }
}

QString vectorToQString(std::vector<char> v){
  return QString::fromStdString(std::string(v.begin(), v.end()));
}

OffroadAlert::OffroadAlert(QWidget* parent){
  vlayout = new QVBoxLayout;
  refresh();
  setLayout(vlayout);
}

void OffroadAlert::refresh(){
  cleanLayout(vlayout);
  parse_alerts();

  bool updateAvailable = false;
  std::vector<char> bytes = Params().read_db_bytes("UpdateAvailable");
  if (bytes.size() && bytes[0] == '1'){
    updateAvailable = true;
  }
  show_alert = updateAvailable || alerts.size() ;
   
  if (updateAvailable){
    //If there is update available, don't show alerts
    alerts.clear();

    QFrame *f = new QFrame();

    QVBoxLayout *update_layout = new QVBoxLayout;
    update_layout->addWidget(new QLabel("Update available"));

    std::vector<char> release_notes_bytes = Params().read_db_bytes("ReleaseNotes");
    QString releaseNotes = vectorToQString(release_notes_bytes);
    QLabel *notes_label = new QLabel(releaseNotes);
    notes_label->setWordWrap(true);
    update_layout->addSpacing(20);
    update_layout->addWidget(notes_label);
    update_layout->addSpacing(20);

    QPushButton *update_button = new QPushButton("Reboot and Update");
    update_layout->addWidget(update_button);
    update_layout->setMargin(10);
#ifdef __aarch64__
    QObject::connect(update_button, &QPushButton::released,[=]() {std::system("sudo reboot");});
#endif

    f->setLayout(update_layout);
    f->setStyleSheet(R"(
      .QFrame{
        border-radius: 30px;
        border: 2px solid white;
        background-color: #114267;
      }
      QLabel{
        font-size: 60px;
        background-color: #114267;
      }
    )");

    vlayout->addWidget(f);
    vlayout->addSpacing(60);
  }else{
    vlayout->addSpacing(60);

    for (auto alert : alerts){
      QLabel *l = new QLabel(alert.text);
      l->setWordWrap(true);
      l->setMargin(60);
      
      if (alert.severity){
        l->setStyleSheet(R"(
          QLabel {
            font-size: 40px;
            font-weight: bold;
            border-radius: 30px;
            background-color: #971b1c;
            border-style: solid;
            border-width: 2px;
            border-color: white;
          }
        )");//red rounded rectange with white surround
      }else{
        l->setStyleSheet(R"(
          QLabel {
            font-size: 40px;
            font-weight: bold;
            border-radius: 30px;
            background-color: #114267;
            border-style: solid;
            border-width: 2px;
            border-color: white;
          }
        )");//blue rounded rectange with white surround
      }

      vlayout->addWidget(l);
      vlayout->addSpacing(20);
    }

    //Pad the vlayout
    for (int i = alerts.size(); i < 4; i++){
      QWidget *w = new QWidget();
      vlayout->addWidget(w);
      vlayout->addSpacing(50);
    }
  }

  QPushButton *hide_alerts_button = new QPushButton(updateAvailable ? "Later" : "Hide alerts");
  vlayout->addWidget(hide_alerts_button);
  QObject::connect(hide_alerts_button, SIGNAL(released()), this, SIGNAL(closeAlerts()));
}

void OffroadAlert::parse_alerts(){
  alerts.clear();
  //We launch in selfdrive/ui
  QFile inFile("../controls/lib/alerts_offroad.json");
  inFile.open(QIODevice::ReadOnly | QIODevice::Text);
  QByteArray data = inFile.readAll();
  inFile.close();

  QJsonDocument doc = QJsonDocument::fromJson(data);
  if (doc.isNull()) {
    qDebug() << "Parse failed";
  }
  
  QJsonObject json = doc.object();
  for (const QString& key : json.keys()) {
    std::vector<char> bytes = Params().read_db_bytes(key.toStdString().c_str());
    
    if (bytes.size()){
      QJsonDocument doc_par = QJsonDocument::fromJson(QByteArray(bytes.data(), bytes.size()));
      QJsonObject obj = doc_par.object();
      Alert alert = {obj.value("text").toString(), obj.value("severity").toInt()};
      alerts.push_back(alert);
    }
  }
}
