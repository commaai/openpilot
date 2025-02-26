#include "selfdrive/ui/qt/offroad/firehose.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/offroad/settings.h"

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QScrollArea>
#include <QStackedLayout>
#include <QProgressBar>

FirehosePanel::FirehosePanel(SettingsWindow *parent) : QWidget((QWidget*)parent) {
  layout = new QVBoxLayout(this);
  layout->setContentsMargins(40, 40, 40, 40);
  layout->setSpacing(20);

  // header
  QLabel *title = new QLabel(tr("🔥 Firehose Mode 🔥"));
  title->setStyleSheet("font-size: 100px; font-weight: 500; font-family: 'Noto Color Emoji';");
  layout->addWidget(title, 0, Qt::AlignCenter);

  // Create a container for the content
  QFrame *content = new QFrame();
  content->setStyleSheet("background-color: #292929; border-radius: 15px; padding: 20px;");
  QVBoxLayout *content_layout = new QVBoxLayout(content);
  content_layout->setSpacing(20);

  // Top description
  QLabel *description = new QLabel(tr("openpilot learns to drive by watching humans, like you, drive.\n\nFirehose Mode allows you to maximize your training data uploads to improve openpilot's driving models. More data means bigger models, so Experimental mode will get better!"));
  description->setStyleSheet("font-size: 45px; padding-bottom: 20px;");
  description->setWordWrap(true);
  content_layout->addWidget(description);

  // Add a separator
  QFrame *line = new QFrame();
  line->setFrameShape(QFrame::HLine);
  line->setFrameShadow(QFrame::Sunken);
  line->setStyleSheet("background-color: #444444; margin-top: 5px; margin-bottom: 5px;");
  content_layout->addWidget(line);

  // Enable firehose toggle - using ParamControl with no description
  enable_firehose = new ParamControl(
    "FirehoseMode",
    tr("Start Firehose Mode"), 
    "", // No description for the toggle
    ""  // No icon
  );
  
  // Connect to toggle state changes
  QObject::connect(enable_firehose, &ParamControl::toggleFlipped, [=](bool state) {
    updateFirehoseState(state);
  });
  
  content_layout->addWidget(enable_firehose);

  // Create progress bar container (always visible)
  progress_container = new QFrame();
  QHBoxLayout *progress_layout = new QHBoxLayout(progress_container);
  progress_layout->setContentsMargins(10, 0, 10, 10);
  progress_layout->setSpacing(20);
  
  // Progress bar
  progress_bar = new QProgressBar();
  progress_bar->setRange(0, 100);
  progress_bar->setValue(0);
  progress_bar->setTextVisible(false);
  progress_bar->setStyleSheet(R"(
    QProgressBar {
      background-color: #444444;
      border-radius: 10px;
      height: 20px;
    }
    QProgressBar::chunk {
      background-color: #3498db;
      border-radius: 10px;
    }
  )");
  progress_bar->setFixedHeight(40);
  
  // Progress text
  progress_text = new QLabel("0%");
  progress_text->setStyleSheet("font-size: 40px; font-weight: bold; color: white;");
  
  progress_layout->addWidget(progress_bar, 1);
  progress_layout->addWidget(progress_text);
  
  content_layout->addWidget(progress_container);
  
  // Add a separator before detailed instructions
  QFrame *line2 = new QFrame();
  line2->setFrameShape(QFrame::HLine);
  line2->setFrameShadow(QFrame::Sunken);
  line2->setStyleSheet("background-color: #444444; margin-top: 10px; margin-bottom: 10px;");
  content_layout->addWidget(line2);
  
  // Detailed instructions at the bottom
  detailed_instructions = new QLabel(tr(
    "Follow these steps to get your device ready:<br>"
    "  1. Bring your device inside and connect to a good USB-C adapter<br>"
    "  2. Connect to Wi-Fi<br>"
    "  3. Enable the toggle<br>"
    "  4. Leave it connected until complete<br>"
    "<br>"
    "The toggle turns off once you restart your device. Repeat at least once a week for maximum effectiveness."
    "<br><br><b>FAQ</b><br>"
    "<i>Does it matter how or where I drive?</i> Nope, just drive as you normally would.<br>"
    "<i>What's a good USB-C adapter?</i> Any fast phone or laptop charger should be fine.<br>"
  ));
  detailed_instructions->setStyleSheet("font-size: 40px; padding: 20px; color: #E4E4E4;");
  detailed_instructions->setWordWrap(true);
  content_layout->addWidget(detailed_instructions);

  // Add content to main layout
  layout->addWidget(content, 1);

  // Connect to UI state changes
  QObject::connect(uiState(), &UIState::offroadTransition, this, &FirehosePanel::updateToggles);
  
  // Initialize state based on params
  bool firehose_enabled = params.getBool("FirehoseMode");
  updateFirehoseState(firehose_enabled);
}

void FirehosePanel::updateFirehoseState(bool enabled) {
  // Update progress bar with current value
  if (enabled) {
    // For demonstration purposes, set a random progress value
    int progress = 35; // Replace with actual progress calculation
    progress_bar->setValue(progress);
    progress_text->setText(QString("%1%").arg(progress));
  } else {
    // Reset progress when disabled
    progress_bar->setValue(0);
    progress_text->setText("0%");
  }
}

void FirehosePanel::showEvent(QShowEvent *event) {
  updateToggles(offroad);
}

void FirehosePanel::updateToggles(bool _offroad) {
  // Update UI elements based on offroad status
  enable_firehose->setEnabled(_offroad);
  
  offroad = _offroad;
}
