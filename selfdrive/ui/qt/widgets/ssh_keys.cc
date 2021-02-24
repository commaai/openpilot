#include <QLabel>
#include <QPushButton>
#include <QState>
#include <QStateMachine>
#include <QNetworkReply>

#include "common/params.h"
#include "widgets/ssh_keys.hpp"
#include "widgets/input_field.hpp"

SSH::SSH(QWidget* parent) : QWidget(parent){
  // init variables
  manager = new QNetworkAccessManager(this);
  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(5000);
  connect(networkTimer, SIGNAL(timeout()), this, SLOT(timeout()));

  // Layout on entering
  QVBoxLayout* main_layout = new QVBoxLayout;
  main_layout->setMargin(50);

  QPushButton* exitButton = new QPushButton("BACK", this);
  exitButton->setFixedSize(500, 100);
  main_layout->addWidget(exitButton, 0, Qt::AlignLeft | Qt::AlignTop);
  connect(exitButton, SIGNAL(released()), this, SIGNAL(closeSSHSettings()));

  QLabel* wallOfText = new QLabel("Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own.");
  wallOfText->setAlignment(Qt::AlignHCenter);
  wallOfText->setWordWrap(true);
  wallOfText->setStyleSheet(R"(font-size: 60px;)");
  main_layout->addWidget(wallOfText, 0);

  QPushButton* actionButton = new QPushButton;
  actionButton->setFixedHeight(100);
  main_layout->addWidget(actionButton, 0, Qt::AlignBottom);

  setStyleSheet(R"(
    QPushButton {
      font-size: 60px;
      margin: 0px;
      padding: 15px;
      border-radius: 25px;
      color: #dddddd;
      background-color: #444444;
    }
  )");
  setLayout(main_layout);

  // Initialize the state machine and states
  QStateMachine* state = new QStateMachine(this);
  QState* initialState = new QState(); //State when entering the widget
  QState* initialStateNoGithub = new QState(); //Starting state, key not connected
  QState* initialStateConnected = new QState(); //Starting state, ssh connected
  QState* removeSSH_State = new QState(); // State when user wants to remove the SSH keys
  QState* loadingState = new QState(); // State while waiting for the network response

  // Adding states to the state machine and adding the transitions
  state->addState(initialState);
  connect(initialState, &QState::entered, [=](){
    checkForSSHKey();
  });
  initialState->addTransition(this, &SSH::NoSSHAdded, initialStateNoGithub);
  initialState->addTransition(this, &SSH::SSHAdded, initialStateConnected);

  state->addState(initialStateConnected);
  connect(initialStateConnected, &QState::entered, [=](){
    actionButton->setText("Clear SSH keys");
    actionButton->setStyleSheet(R"(background-color: #750c0c;)");
  });
  initialStateConnected->addTransition(actionButton, &QPushButton::released, removeSSH_State);

  state->addState(removeSSH_State);
  connect(removeSSH_State, &QState::entered, [=](){
    Params().delete_db_value("GithubSshKeys");
  });
  removeSSH_State->addTransition(removeSSH_State, &QState::entered, initialState);

  state->addState(initialStateNoGithub);
  connect(initialStateNoGithub, &QState::entered, [=](){
    actionButton->setText("Link GitHub SSH keys");
    actionButton->setStyleSheet(R"(background-color: #444444;)");
  });
  initialStateNoGithub->addTransition(actionButton, &QPushButton::released, loadingState);

  state->addState(loadingState);
  connect(loadingState, &QState::entered, [=](){
    QString user = InputDialog::getText("Enter your GitHub username");
    if (user.size()) {
      getSSHKeys(user);
    }
  });
  connect(this, &SSH::failedResponse, [=](QString message){
    QString user = InputDialog::getText(message);
    if (user.size()) {
      getSSHKeys(user);
    }
  });
  loadingState->addTransition(loadingState, &QState::entered, initialState);
  loadingState->addTransition(this, &SSH::failedResponse, initialState);
  loadingState->addTransition(this, &SSH::gotSSHKeys, initialState);

  state->setInitialState(initialState);
  state->start();
}

void SSH::checkForSSHKey(){
  QString SSHKey = QString::fromStdString(Params().get("GithubSshKeys"));
  if (SSHKey.length()) {
    emit SSHAdded();
  } else {
    emit NoSSHAdded();
  }
}

void SSH::getSSHKeys(QString username){
  QString url = "https://github.com/" + username + ".keys";
  aborted = false;
  reply = manager->get(QNetworkRequest(QUrl(url)));
  connect(reply, SIGNAL(finished()), this, SLOT(parseResponse()));
  networkTimer->start();
}

void SSH::timeout(){
  aborted = true;
  reply->abort();
}

void SSH::parseResponse(){
  if (!aborted) {
    networkTimer->stop();
    QString response = reply->readAll();
    if (reply->error() == QNetworkReply::NoError && response.length()) {
      Params().write_db_value("GithubSshKeys", response.toStdString());
      emit gotSSHKeys();
    } else {
      emit failedResponse("Username doesn't exist");
    }
  } else {
    emit failedResponse("Request timed out");
  }
  reply->deleteLater();
  reply = NULL;
}
