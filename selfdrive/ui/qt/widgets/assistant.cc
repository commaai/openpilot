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

  hideTimer = new QTimer(this);
  connect(hideTimer, &QTimer::timeout, this, &AssistantOverlay::animateHide);

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
  hideTimer->stop();
}

void AssistantOverlay::updateText(QString text) {
  this->setText(text);
  this->setAlignment(QFontMetrics(this->font()).horizontalAdvance(text) > this->finalWidth ? Qt::AlignRight : Qt::AlignCenter);
}


void AssistantOverlay::updateState(const UIState &s) {
  const SubMaster &sm = *(s.sm);
  if (!sm.updated("speechToText")) return;

  static cereal::SpeechToText::State current_state = cereal::SpeechToText::State::NONE;
  cereal::SpeechToText::State requested_state = sm["speechToText"].getSpeechToText().getState();

  // Check for valid state transition
  if (current_state == cereal::SpeechToText::State::BEGIN ||
     (current_state == cereal::SpeechToText::State::NONE &&
     (requested_state == cereal::SpeechToText::State::EMPTY ||
      requested_state == cereal::SpeechToText::State::FINAL ||
      requested_state == cereal::SpeechToText::State::NONE)) ||
      requested_state == cereal::SpeechToText::State::BEGIN) {

    current_state = requested_state;  // Update state

    // Handle UI updates
    switch (current_state) {
      case cereal::SpeechToText::State::BEGIN:
        if (!hideTimer->isActive()) animateShow();
        updateText("Hello, I'm listening");
        hideTimer->start(30000);
        break;
      case cereal::SpeechToText::State::EMPTY:
        updateText("Sorry, I didn't catch that");
        hideTimer->start(8000);
        break;
      case cereal::SpeechToText::State::ERROR:
        updateText("Sorry, an error occorred");
        hideTimer->start(8000);
        break;
      case cereal::SpeechToText::State::FINAL:
      case cereal::SpeechToText::State::NONE:
        updateText(QString::fromStdString(sm["speechToText"].getSpeechToText().getTranscript()));
        hideTimer->start(requested_state == cereal::SpeechToText::State::FINAL ? 8000 : 30000);
        break;
    }
  }
}