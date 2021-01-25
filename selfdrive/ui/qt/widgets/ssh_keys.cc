#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QState>
#include <QStateMachine>

#include "widgets/ssh_keys.hpp"
#include "widgets/input_field.hpp"
#include "common/params.h"

QWidget* layout_to_widget(QLayout* l){
  QWidget* q = new QWidget;
  q->setLayout(l);
  return q;
}

SSH::SSH(QWidget* parent) : QWidget(parent){
  networkAccessManager = new QNetworkAccessManager(this);
  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(5000);

  //Initialize the state machine and states
  QStateMachine* state = new QStateMachine(this);
  QState* initialState = new QState(); //State when entering the widget
  QState* initialStateNoGithub = new QState(); //Starting state, key not connected
  QState* initialStateConnected = new QState(); //Starting state, ssh connected
  QState* quitState = new QState(); // State when exiting the widget
  QState* removeSSH_State = new QState(); // State when user wants to remove the SSH keys 
  QState* defaultInputFieldState = new QState(); // State when we want the user to give us the username 
  QState* loadingState = new QState(); // State while waiting for the network response
  QState* noConnectionInputFieldState = new QState(); // State when the connection timed out
  QState* noUsernameInputFieldState = new QState(); // State when the username doesn't exist

  // Construct the layouts to display
  slayout = new QStackedLayout(this); // Initial screen, input, waiting for response

  //Layout on entering
  QVBoxLayout* initialLayout = new QVBoxLayout;
  initialLayout->setContentsMargins(80, 80, 80, 80);
  QHBoxLayout* header = new QHBoxLayout;
  header->addWidget(new QLabel("Authorized SSH keys"), 0, Qt::AlignLeft);

  QPushButton* exitButton = new QPushButton("CANCEL", this);

  header->addWidget(exitButton, 0, Qt::AlignRight);
  initialLayout->addWidget(layout_to_widget(header));

  QLabel* wallOfText = new QLabel("Warning: This grants SSH access to all public keys in your GitHub settings. \nNever enter a GitHub username other than your own. \nA Comma employee will NEVER ask you to add their GitHub username.");
  wallOfText->setWordWrap(true);
  initialLayout->addWidget(wallOfText);

  QHBoxLayout* currentAccount = new QHBoxLayout;
  currentAccount->addWidget(new QLabel("GitHub account"), 0, Qt::AlignLeft);
  initialLayout->addWidget(layout_to_widget(currentAccount));

  QPushButton* actionButton = new QPushButton;
  initialLayout->addWidget(actionButton);

  slayout->addWidget(layout_to_widget(initialLayout));

  InputField* input = new InputField;
  slayout->addWidget(input);

  QLabel* loading = new QLabel("Loading SSH keys from GitHub.");
  slayout->addWidget(loading);

  setLayout(slayout);

  // Adding states to the state machine and adding the transitions
  state->addState(initialState);
  connect(initialState, &QState::entered, [=](){checkForSSHKey(); slayout->setCurrentIndex(0);});
  initialState->addTransition(this, &SSH::NoSSHAdded, initialStateNoGithub);
  initialState->addTransition(this, &SSH::SSHAdded, initialStateConnected);
  
  
  state->addState(initialStateNoGithub);
  connect(initialStateNoGithub, &QState::entered, [=](){actionButton->setText("Link GitHub SSH keys");});
  initialStateNoGithub->addTransition(exitButton, &QPushButton::released, quitState);
  initialStateNoGithub->addTransition(actionButton, &QPushButton::released, defaultInputFieldState);

  state->addState(initialStateConnected);
  connect(initialStateConnected, &QState::entered, [=](){actionButton->setText("Remove Github SSH keys");});
  initialStateConnected->addTransition(exitButton, &QPushButton::released, quitState);
  initialStateConnected->addTransition(actionButton, &QPushButton::released, removeSSH_State);

  state->addState(quitState);
  connect(quitState, &QState::entered, [=](){emit closeSSHSettings();});
  quitState->addTransition(quitState, &QState::entered, initialState);

  state->addState(removeSSH_State);
  connect(initialStateNoGithub, &QState::entered, [=](){Params().delete_db_value("GithubSshKeys");});
  removeSSH_State->addTransition(removeSSH_State, &QState::entered, initialState);

  state->addState(defaultInputFieldState);
  connect(defaultInputFieldState, &QState::entered, [=](){input->setPromptText("Enter your GitHub username"); slayout->setCurrentIndex(1);});
  connect(input, &InputField::emitText, [=](QString a){usernameGitHub = a;}); // Store the string the user provided
  defaultInputFieldState->addTransition(input, &InputField::cancel, initialState);
  defaultInputFieldState->addTransition(input, &InputField::emitText, loadingState);

  state->addState(loadingState);
  connect(loadingState, &QState::entered, [=](){slayout->setCurrentIndex(2); getSSHKeys();});

  state->addState(noConnectionInputFieldState);

  state->addState(noUsernameInputFieldState);



  state->setInitialState(initialState);
  state->start();
}

void SSH::checkForSSHKey(){
  QString SSHKey = QString::fromStdString(Params().get("GithubSshKeys"));
  if (SSHKey.length()) {
    qDebug()<<"SSHAdded";
    emit SSHAdded();
  } else {
    qDebug()<<"NoSSHAdded";
    emit NoSSHAdded();
  }
}

void SSH::getSSHKeys(){
  QString url = "https://github.com/" + usernameGitHub +".keys";
  manager->get(QNetworkRequest(QUrl(url)));

  qDebug()<<"Getting SSH keys";
}