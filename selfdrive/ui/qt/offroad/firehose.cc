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
#include <QJsonDocument>
#include <QJsonObject>

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
  QLabel *description = new QLabel(tr("openpilot learns to drive by watching humans, like you, drive.\n\nFirehose Mode allows you to maximize your training data uploads to improve openpilot's driving models. More data means bigger models, which means better Experimental Mode."));
  description->setStyleSheet("font-size: 45px; padding-bottom: 20px;");
  description->setWordWrap(true);
  content_layout->addWidget(description);

  // Add a separator
  QFrame *line = new QFrame();
  line->setFrameShape(QFrame::HLine);
  line->setFrameShadow(QFrame::Sunken);
  line->setStyleSheet("background-color: #444444; margin-top: 5px; margin-bottom: 5px;");
  content_layout->addWidget(line);

  QLabel *toggle_label = new QLabel(tr("Firehose Mode: ACTIVE"));
  toggle_label->setStyleSheet("font-size: 60px; font-weight: bold; color: white;");
  content_layout->addWidget(toggle_label);

  // Add contribution label
  contribution_label = new QLabel("0 minutes");
  contribution_label->setStyleSheet("font-size: 40px; margin-top: 10px; margin-bottom: 10px;");
  contribution_label->setWordWrap(true);
  content_layout->addWidget(contribution_label);

  // Create progress bar container
  progress_container = new QFrame();
  progress_container->hide();
  QHBoxLayout *progress_layout = new QHBoxLayout(progress_container);
  progress_layout->setContentsMargins(10, 0, 10, 10);
  progress_layout->setSpacing(20);

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
  progress_text = new QLabel(tr("0%"));
  progress_text->setStyleSheet("font-size: 40px; font-weight: bold; color: white;");

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
    "For maximum effectiveness, bring your device inside and connect to a good USB-C adapter and Wi-Fi weekly.<br>"
    "<br>"
    "Firehose Mode can also work while you're driving if connected to a hotspot or unlimited SIM card.<br>"
    "<br>"
    "<br><br><b>Frequently Asked Questions</b><br><br>"
    "<i>Does it matter how or where I drive?</i> Nope, just drive as you normally would.<br><br>"
    "<i>What's a good USB-C adapter?</i> Any fast phone or laptop charger should be fine.<br><br>"
    "<i>Does it matter which software I run?</i> Yes, only upstream openpilot (and particular forks) are able to be used for training."
  ));
  detailed_instructions->setStyleSheet("font-size: 40px; color: #E4E4E4;");
  detailed_instructions->setWordWrap(true);
  content_layout->addWidget(detailed_instructions);

  layout->addWidget(content, 1);

  // Set up the API request for firehose stats
  const QString dongle_id = QString::fromStdString(Params().get("DongleId"));
  firehose_stats = new RequestRepeater(this, CommaApi::BASE_URL + "/v1/devices/" + dongle_id + "/firehose_stats",
                                       "ApiCache_FirehoseStats", 30, true);
  QObject::connect(firehose_stats, &RequestRepeater::requestDone, [=](const QString &response, bool success) {
    if (success) {
      QJsonDocument doc = QJsonDocument::fromJson(response.toUtf8());
      QJsonObject json = doc.object();
      int count = json["firehose"].toInt();
      contribution_label->setText(tr("%1 %2 of your driving are in the training dataset so far.").arg(count).arg(count == 1 ? "minute" : "minutes"));
      contribution_label->show();
    }
  });
}