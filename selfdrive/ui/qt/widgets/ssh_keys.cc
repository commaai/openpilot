#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QState>
#include <QStateMachine>
#include <QNetworkReply>

#include "widgets/ssh_keys.hpp"
#include "widgets/input.hpp"
#include "common/params.h"


SshControl::SshControl() : AbstractControl("SSH Keys", "", "") {
  // setup widget
  btn.setFixedSize(200, 80);
  btn.setStyleSheet(R"(
    padding: 0;
    border-radius: 40px;
    font-size: 30px;
    font-weight: 500;
    color: #E4E4E4;
    background-color: #393939;
  )");
  hlayout->addWidget(&btn);

  QObject::connect(&btn, &QPushButton::released, [=]() {
    if (btn.text() == "ADD") {
      username = InputDialog::getText("Enter your GitHub username", 1);
      if (username.length() > 0) {
        getUserKeys(username);
      }
      btn.setEnabled(false);
    } else {
      Params().delete_db_value("GithubSshKeys");
      refresh();
    }
  });

  // setup networking
  manager = new QNetworkAccessManager(this);
  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(5000);
  connect(networkTimer, SIGNAL(timeout()), this, SLOT(timeout()));

  refresh();

  // TODO: add desription through AbstractControl
  //QLabel* wallOfText = new QLabel("Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own. A Comma employee will NEVER ask you to add their GitHub username.");
}

void SshControl::refresh() {
  QString param = QString::fromStdString(Params().get("GithubSshKeys"));
  if (param.length()) {
    btn.setText("REMOVE");
  } else {
    btn.setText("ADD");
  }
  btn.setEnabled(true);
}

void SshControl::getUserKeys(QString username){
  QString url = "https://github.com/" + username + ".keys";
  aborted = false;
  reply = manager->get(QNetworkRequest(QUrl(url)));
  connect(reply, SIGNAL(finished()), this, SLOT(parseResponse()));
  networkTimer->start();
}

void SshControl::timeout(){
  aborted = true;
  reply->abort();
}

void SshControl::parseResponse(){
  if (!aborted) {
    networkTimer->stop();
    QString response = reply->readAll();
    if (reply->error() == QNetworkReply::NoError && response.length()) {
      Params().write_db_value("GithubSshKeys", response.toStdString());
    } else if(reply->error() == QNetworkReply::NoError){
      //emit failedResponse("Username " + usernameGitHub + " has no keys on GitHub");
    } else {
      //emit failedResponse("Username " + usernameGitHub + " doesn't exist");
    }
  } else {
    emit failedResponse("Request timed out");
  }

  refresh();
  reply->deleteLater();
  reply = nullptr;
}
