#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QState>
#include <QStateMachine>
#include <QNetworkReply>

#include "widgets/ssh_keys.hpp"
#include "widgets/input.hpp"
#include "common/params.h"

QWidget* layout_to_widget(QLayout* l){
  QWidget* q = new QWidget;
  q->setLayout(l);
  return q;
}

SSH::SSH(QWidget* parent) : QWidget(parent){
  // init variables
  manager = new QNetworkAccessManager(this);
  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(5000);
  connect(networkTimer, SIGNAL(timeout()), this, SLOT(timeout()));

  dialog = new InputDialog("");
  // Construct the layouts to display
  slayout = new QStackedLayout(this); // Initial screen, input, waiting for response

  //Layout on entering
  QVBoxLayout* initialLayout = new QVBoxLayout;
  initialLayout->setContentsMargins(80, 80, 80, 80);

  QHBoxLayout* header = new QHBoxLayout;
  QPushButton* exitButton = new QPushButton("BACK", this);
  exitButton->setFixedSize(500, 100);
  header->addWidget(exitButton, 0, Qt::AlignLeft | Qt::AlignTop);
  initialLayout->addWidget(layout_to_widget(header));

  QLabel* title = new QLabel("Authorize SSH keys");
  title->setStyleSheet(R"(font-size: 75px;)");
  header->addWidget(title, 0, Qt::AlignRight | Qt::AlignTop);

  QLabel* wallOfText = new QLabel("Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own. A Comma employee will NEVER ask you to add their GitHub username.");
  wallOfText->setWordWrap(true);
  wallOfText->setStyleSheet(R"(font-size: 60px;)");
  initialLayout->addWidget(wallOfText, 0);

  QPushButton* actionButton = new QPushButton;
  actionButton->setFixedHeight(100);
  initialLayout->addWidget(actionButton, 0, Qt::AlignBottom);

  slayout->addWidget(layout_to_widget(initialLayout));

  QLabel* loading = new QLabel("Loading SSH keys from GitHub.");
  slayout->addWidget(loading);

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
  setLayout(slayout);


  //Initialize the state machine and states
  QStateMachine* state = new QStateMachine(this);
  QState* initialState = new QState(); //State when entering the widget
  QState* initialStateNoGithub = new QState(); //Starting state, key not connected
  QState* initialStateConnected = new QState(); //Starting state, ssh connected
  QState* quitState = new QState(); // State when exiting the widget
  QState* removeSSH_State = new QState(); // State when user wants to remove the SSH keys
  QState* defaultInputFieldState = new QState(); // State when we want the user to give us the username
  QState* loadingState = new QState(); // State while waiting for the network response


  // Adding states to the state machine and adding the transitions
  state->addState(initialState);
  connect(initialState, &QState::entered, [=](){checkForSSHKey(); slayout->setCurrentIndex(0);});
  initialState->addTransition(this, &SSH::NoSSHAdded, initialStateNoGithub);
  initialState->addTransition(this, &SSH::SSHAdded, initialStateConnected);


  state->addState(quitState);
  connect(quitState, &QState::entered, [=](){emit closeSSHSettings();});
  quitState->addTransition(quitState, &QState::entered, initialState);

  state->addState(initialStateConnected);
  connect(initialStateConnected, &QState::entered, [=](){actionButton->setText("Remove GitHub SSH keys"); actionButton->setStyleSheet(R"(background-color: #750c0c;)");});
  initialStateConnected->addTransition(exitButton, &QPushButton::released, quitState);
  initialStateConnected->addTransition(actionButton, &QPushButton::released, removeSSH_State);

  state->addState(removeSSH_State);
  connect(removeSSH_State, &QState::entered, [=](){Params().delete_db_value("GithubSshKeys");});
  removeSSH_State->addTransition(removeSSH_State, &QState::entered, initialState);

  state->addState(initialStateNoGithub);
  connect(initialStateNoGithub, &QState::entered, [=](){actionButton->setText("Link GitHub SSH keys"); actionButton->setStyleSheet(R"(background-color: #444444;)");});
  initialStateNoGithub->addTransition(exitButton, &QPushButton::released, quitState);
  initialStateNoGithub->addTransition(actionButton, &QPushButton::released, defaultInputFieldState);
  connect(actionButton, &QPushButton::released, [=](){dialog->setMessage("Enter your GitHub username");});

  state->addState(defaultInputFieldState);
  connect(defaultInputFieldState, &QState::entered, [=](){dialog->show();});
  connect(dialog, &InputDialog::emitText, [=](QString a){usernameGitHub = a;});
  defaultInputFieldState->addTransition(dialog, &InputDialog::cancel, initialState);
  defaultInputFieldState->addTransition(dialog, &InputDialog::emitText, loadingState);


  state->addState(loadingState);
  connect(loadingState, &QState::entered, [=](){slayout->setCurrentIndex(1); getSSHKeys();});
  connect(this, &SSH::failedResponse, [=](QString message){dialog->setMessage(message);});
  loadingState->addTransition(this, &SSH::failedResponse, defaultInputFieldState);
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

void SSH::getSSHKeys(){
  QString url = "https://github.com/" + usernameGitHub + ".keys";
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
    } else if(reply->error() == QNetworkReply::NoError){
      emit failedResponse("Username " + usernameGitHub + " has no keys on GitHub");
    } else {
      emit failedResponse("Username " + usernameGitHub + " doesn't exist");
    }
  }else{
    emit failedResponse("Request timed out");
  }
  reply->deleteLater();
  reply = nullptr;
}
