#include "assistant.h"

AssistantOverlay::AssistantOverlay(QWidget *parent) : QLabel(parent) {

    setStyleSheet("QLabel {"
                    "  background-color: #373737;"
                    "  border-radius: 20px;"
                    "  font-family: 'Inter';"
                    "  font-size: 60px;"
                    "  color: white;"  // Text color
                    "}");

    // Set up the animations
    showAnimation = new QPropertyAnimation(this, "geometry");
    showAnimation->setDuration(250); // Duration in milliseconds

    hideAnimation = new QPropertyAnimation(this, "geometry");
    hideAnimation->setDuration(250);

    int height = 100; // Fixed height
    setGeometry(0, 0, 0, height);

    QObject::connect(uiState(), &UIState::uiUpdate, this, &AssistantOverlay::updateState);
    hide();
}

void AssistantOverlay::animateShow() {
    parentCenterX = parentWidget()->width() / 2;
    finalWidth = parentWidget()->width() * 0.5;
    startX = parentCenterX - finalWidth / 2;
    QRect startRect(parentCenterX, 0, 0, height()); // Centered, zero width
    QRect endRect(startX, 0, finalWidth, height()); // Adjusted x, final width
    showAnimation->setStartValue(startRect);
    showAnimation->setEndValue(endRect);
    show();
    showAnimation->start();
}

void AssistantOverlay::animateHide() {
    parentCenterX = parentWidget()->width() / 2;
    finalWidth = parentWidget()->width() * 0.5;
    startX = parentCenterX - finalWidth / 2;
    QRect startRect(startX, 0, finalWidth, height()); // Adjusted x, final width
    QRect endRect(parentCenterX, 0, 0, height()); // Centered, zero width
    hideAnimation->setStartValue(startRect);
    hideAnimation->setEndValue(endRect);
    hideAnimation->start();
    connect(hideAnimation, &QPropertyAnimation::finished, this, &AssistantOverlay::hide);
}

void AssistantOverlay::updateState(const UIState &s) {
  const SubMaster &sm = *(s.sm);

  if (sm.updated("speechToText")) {
    if (!this->isVisible()) {this->animateShow();}
    this->setText(QString::fromStdString(sm["speechToText"].getSpeechToText().getResult()));
    // Calculate the width of the new text and set the alignment
    this->setAlignment(QFontMetrics(this->font()).horizontalAdvance(this->text()) > this->finalWidth ? Qt::AlignRight : Qt::AlignCenter);
    if (sm["speechToText"].getSpeechToText().getFinalResultReady()) {
        QTimer::singleShot(8000, this, &AssistantOverlay::animateHide);
    }
  }
}
