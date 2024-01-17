#pragma once

#include "selfdrive/ui/ui.h"
#include <QLabel>
#include <QPropertyAnimation>

class AssistantOverlay : public QLabel {
    Q_OBJECT

public:
    explicit AssistantOverlay(QWidget *parent = nullptr);

    void updateText(const QString &newText);
    void animateShow();
    void animateHide();

private:
    QTimer *updateTimer;
    QPropertyAnimation *showAnimation;
    QPropertyAnimation *hideAnimation;


private slots:
    void updateState(const UIState &s);
};


