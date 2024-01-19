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
  static bool show_allowed = false;
  static bool visable = false;
  if (!sm.updated("speechToText")) {
    return; // Early exit if speechToText is not updated
  }
  // Should probably refactor to a switch statement but its working.
  if (cereal::SpeechToText::State::BEGIN == sm["speechToText"].getSpeechToText().getState()) {
    show_allowed = true;
    this->animateShow();
    updateText("Hello, I'm listening");
    if (hideTimer->isActive()) {
      hideTimer->stop();
    }
    visable = true; // Require begin state or not valid to show
  } else if (!sm["speechToText"].getValid()){ // show if not valid and show set the error text, then lock out and hide until next begin
    if (!visable && show_allowed) {this->animateShow(); visable = true;}
    updateText("Sorry, an error occorred");
    hideTimer->start(8000);
    show_allowed = false;
  } else if (cereal::SpeechToText::State::EMPTY == sm["speechToText"].getSpeechToText().getState()){
    updateText("Sorry, I didn't catch that");
    hideTimer->start(8000);
    visable = false;
    show_allowed = false;
  } else if (show_allowed){ // Interim and Final Results
    updateText(QString::fromStdString(sm["speechToText"].getSpeechToText().getResult()));
    if (sm["speechToText"].getSpeechToText().getFinalResultReady()) {
      hideTimer->start(8000);
      visable = false;
    }
  } else { //shouldn't get here unless I missed something
    qWarning() << "AssistantOverlay in bad state";
  }
}
