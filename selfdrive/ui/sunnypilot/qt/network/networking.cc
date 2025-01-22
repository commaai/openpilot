#include "selfdrive/ui/sunnypilot/qt/network/networking.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QStackedLayout>

NetworkingSP::NetworkingSP(QWidget *parent) : Networking(parent) {
  auto vlayout = wifiScreen->findChild<QVBoxLayout*>();
  auto hlayout = new QHBoxLayout();
  
  // Create and setup scan button
  auto scanButton = new QPushButton(tr("Scan"));
  scanButton->setObjectName("scan_btn");
  scanButton->setFixedSize(400, 100);

  connect(wifi, &WifiManager::refreshSignal, this, [=]() { scanButton->setText(tr("Scan")); scanButton->setEnabled(true); });
  connect(scanButton, &QPushButton::clicked, [=]() { scanButton->setText(tr("Scanning...")); scanButton->setEnabled(false); wifi->requestScan(); });

  hlayout->addWidget(scanButton);
  hlayout->addStretch(1);

  // Look for an existing Advanced button
  QPushButton* existingAdvanced = wifiScreen->findChild<QPushButton*>("advanced_btn");
  if (existingAdvanced) {
    hlayout->addWidget(existingAdvanced);
  }

  // Insert our new layout at the top of vlayout
  vlayout->setMargin(40);
  vlayout->insertLayout(0, hlayout);

  // Add our scan button to the existing style selectors
  auto newStyleSheet = styleSheet().replace(
    ", #advanced_btn ",
    ", #advanced_btn, #scan_btn "
  ).replace(
    ", #advanced_btn:pressed",
    ", #advanced_btn:pressed, #scan_btn:pressed"
  ).append(R"(
    #scan_btn:disabled {
      background-color: #121212;
      color: #33FFFFFF;
    }
  )");
  setStyleSheet(newStyleSheet);
}
