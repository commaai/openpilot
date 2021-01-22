#include <QDebug>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QState>
#include <QStateMachine>

#include "widgets/ssh_keys.hpp"
QWidget* layout_to_widget(QLayout* l){
  QWidget* q = new QWidget;
  q->setLayout(l);
  return q;
}

SSH::SSH(QWidget* parent) : QWidget(parent){
    QStateMachine* state = new QStateMachine(this);
    QState* initialState = new QState(); //State when entering the widget
    QState* quitState = new QState(); // State when exiting the widget
    QState* removeSSH_State = new QState(); // State when user wants to remove the SSH keys 
    QState* defaultInputFieldState = new QState(); // State when we want the user to give us the username 
    QState* noConnectionInputFieldState = new QState(); // State when the connection timed out
    QState* noUserNameInputFieldState = new QState(); // State when the username doesn't exist 
    QState* loadingState = new QState(); // State while waiting for the network response
    
    // Construct the layouts to display
    slayout = new QStackedLayout(this);

    QVBoxLayout* initialLayout = new QVBoxLayout;
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


    slayout->addWidget(layout_to_widget(initialLayout));
    setLayout(slayout);

    // Adding states to the state machine and adding the transitions
    state->addState(initialState);
    initialState->addTransition(exitButton, &QPushButton::released, quitState);

    state->addState(quitState);
    connect(quitState, &QState::entered, [=](){emit closeSSHSettings();});
    quitState->addTransition(quitState, &QState::entered, initialState);

    state->addState(removeSSH_State);

    state->addState(defaultInputFieldState);

    state->addState(noConnectionInputFieldState);

    state->addState(noUserNameInputFieldState);
    
    state->addState(loadingState);
    
    state->setInitialState(initialState);
    state->start();
}
