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
  // init widget
  btn.setText("EDIT");
  btn.setStyleSheet(R"(
    padding: 0;
    border-radius: 40px;
    font-size: 30px;
    font-weight: 500;
    color: #E4E4E4;
    background-color: #393939;
  )");
  btn.setFixedSize(200, 80);
  hlayout->addWidget(&btn);

  // setup networking
  manager = new QNetworkAccessManager(this);
  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(5000);
  connect(networkTimer, SIGNAL(timeout()), this, SLOT(timeout()));

  // TODO: add desription
  //QLabel* wallOfText = new QLabel("Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own. A Comma employee will NEVER ask you to add their GitHub username.");
}

void SshControl::checkForSSHKey(){
  QString param = QString::fromStdString(Params().get("GithubSshKeys"));
  if (param.length()) {
    emit SSHAdded();
  } else {
    emit NoSSHAdded();
  }
}

void SshControl::getSSHKeys(QString username){
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
      emit gotSSHKeys();
    } else if(reply->error() == QNetworkReply::NoError){
      //emit failedResponse("Username " + usernameGitHub + " has no keys on GitHub");
    } else {
      //emit failedResponse("Username " + usernameGitHub + " doesn't exist");
    }
  } else {
    emit failedResponse("Request timed out");
  }

  reply->deleteLater();
  reply = nullptr;
}
