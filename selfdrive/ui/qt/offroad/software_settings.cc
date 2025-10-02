#include "selfdrive/ui/qt/offroad/settings.h"

#include <cassert>
#include <cmath>
#include <string>

#include <QDebug>
#include <QLabel>
#include <QApplication>
#include <QScreen>

#include "common/params.h"
#include "common/util.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/widgets/input.h"
#include "system/hardware/hw.h"


void SoftwarePanel::checkForUpdates() {
  std::system("pkill -SIGUSR1 -f system.updated.updated");
}

SoftwarePanel::SoftwarePanel(QWidget* parent) : ListWidget(parent) {
  onroadLbl = new QLabel(tr("Updates are only downloaded while the car is off."));
  onroadLbl->setStyleSheet("font-size: 50px; font-weight: 400; text-align: left; padding-top: 30px; padding-bottom: 30px;");
  addItem(onroadLbl);

  // current version
  versionLbl = new LabelControl(tr("Current Version"), "");
  addItem(versionLbl);

  // download update btn
  downloadBtn = new ButtonControl(tr("Download"), tr("CHECK"));
  connect(downloadBtn, &ButtonControl::clicked, [=]() {
    downloadBtn->setEnabled(false);
    if (downloadBtn->text() == tr("CHECK")) {
      checkForUpdates();
    } else {
      std::system("pkill -SIGHUP -f system.updated.updated");
    }
  });
  addItem(downloadBtn);

  // install update btn
  installBtn = new ButtonControl(tr("Install Update"), tr("INSTALL"));
  connect(installBtn, &ButtonControl::clicked, [=]() {
    installBtn->setEnabled(false);
    params.putBool("DoReboot", true);
  });
  addItem(installBtn);

  // branch selecting
  targetBranchBtn = new ButtonControl(tr("Target Branch"), tr("SELECT"));
  connect(targetBranchBtn, &ButtonControl::clicked, [=]() {
    auto current = params.get("GitBranch");
    QStringList branches = QString::fromStdString(params.get("UpdaterAvailableBranches")).split(",");
    for (QString b : {current.c_str(), "devel-staging", "devel", "nightly", "nightly-dev", "master"}) {
      auto i = branches.indexOf(b);
      if (i >= 0) {
        branches.removeAt(i);
        branches.insert(0, b);
      }
    }

    QString cur = QString::fromStdString(params.get("UpdaterTargetBranch"));
    QString selection = MultiOptionDialog::getSelection(tr("Select a branch"), branches, cur, this);
    if (!selection.isEmpty()) {
      params.put("UpdaterTargetBranch", selection.toStdString());
      targetBranchBtn->setValue(QString::fromStdString(params.get("UpdaterTargetBranch")));
      checkForUpdates();
    }
  });
  if (!params.getBool("IsTestedBranch")) {
    addItem(targetBranchBtn);
  }

  // uninstall button
  auto uninstallBtn = new ButtonControl(tr("Uninstall %1").arg(getBrand()), tr("UNINSTALL"));
  connect(uninstallBtn, &ButtonControl::clicked, [&]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to uninstall?"), tr("Uninstall"), this)) {
      params.putBool("DoUninstall", true);
    }
  });
  addItem(uninstallBtn);

  fs_watch = new ParamWatcher(this);
  QObject::connect(fs_watch, &ParamWatcher::paramChanged, [=](const QString &param_name, const QString &param_value) {
    updateLabels();
  });

  connect(uiState(), &UIState::offroadTransition, [=](bool offroad) {
    is_onroad = !offroad;
    updateLabels();
  });

  updateLabels();
}

void SoftwarePanel::paintEvent(QPaintEvent *event) {
  QWidget::paintEvent(event);
  if (qEnvironmentVariableIsSet("DEBUG_TEXT_COMPARE") && qgetenv("DEBUG_TEXT_COMPARE") == "1") {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    struct Sample { int size; const char *text; };
    const Sample samples[] = {{100, "HH"}, {80, "HH"}, {70, "HH"}};

    int left_x = 8;  // far left to avoid overlapping content
    int top_y = height() / 2 + 40;  // bottom half of screen
    int row_h = 120;

    for (int i = 0; i < 3; ++i) {
      int sz = samples[i].size;
      QString text = samples[i].text;
      int y = top_y + i * row_h;

      QFont f = font();
      f.setPixelSize(sz);
      p.setFont(f);
      QFontMetrics fm(f);
      QRect br = fm.boundingRect(text);

      int tx = left_x + 12;
      int ty = y + fm.ascent(); // baseline
      p.setPen(Qt::white);
      p.drawText(tx, ty, text);

      // Green: full font metrics box (top at ascent, height = fm.height())
      p.setPen(QPen(Qt::green, 1));
      QRect r(tx, ty - fm.ascent(), br.width(), fm.height());
      p.drawRect(r);

      // Tight bounding rect of the actual glyph (optional for reference)

      // Red: exactly font-size pixels tall, starting at top of the green box
      p.setPen(QPen(Qt::red, 2));
      p.drawLine(QPoint(left_x, r.top()), QPoint(left_x, r.top() + sz));
    }

    // Metrics readout
    QFont mf = font();
    mf.setPixelSize(18);
    p.setFont(mf);
    auto scr = QApplication::primaryScreen();
    QString scale_line = QString("QT_SCALE_FACTOR=%1 DPR=%2 logicalDPI=%3 physicalDPI=%4")
                           .arg(QString::fromUtf8(qgetenv("QT_SCALE_FACTOR")))
                           .arg(scr ? scr->devicePixelRatio() : 0.0)
                           .arg(scr ? scr->logicalDotsPerInch() : 0.0)
                           .arg(scr ? scr->physicalDotsPerInch() : 0.0);
    p.setPen(Qt::white);
    p.drawText(left_x, top_y + 3 * row_h + 40, scale_line);

    // Per-size metrics (height and tight height)
    int metrics_y = top_y + 3 * row_h + 70;
    const int line_step = 22;
    for (int i = 0; i < 3; ++i) {
      int sz = samples[i].size;
      QFont f = font();
      f.setPixelSize(sz);
      QFontMetrics fm(f);
      QRect tr = fm.tightBoundingRect("HH");
      QString mline = QString("sz=%1 fm.height=%2 tight.h=%3 ascent=%4 descent=%5")
                        .arg(sz)
                        .arg(fm.height())
                        .arg(tr.height())
                        .arg(fm.ascent())
                        .arg(fm.descent());
      p.drawText(left_x, metrics_y + i * line_step, mline);
    }
  }
}

void SoftwarePanel::showEvent(QShowEvent *event) {
  // nice for testing on PC
  installBtn->setEnabled(true);

  updateLabels();
}

void SoftwarePanel::updateLabels() {
  // add these back in case the files got removed
  fs_watch->addParam("LastUpdateTime");
  fs_watch->addParam("UpdateFailedCount");
  fs_watch->addParam("UpdaterState");
  fs_watch->addParam("UpdateAvailable");

  if (!isVisible()) {
    return;
  }

  // updater only runs offroad
  onroadLbl->setVisible(is_onroad);
  downloadBtn->setVisible(!is_onroad);

  // download update
  QString updater_state = QString::fromStdString(params.get("UpdaterState"));
  bool failed = std::atoi(params.get("UpdateFailedCount").c_str()) > 0;
  if (updater_state != "idle") {
    downloadBtn->setEnabled(false);
    downloadBtn->setValue(updater_state);
  } else {
    if (failed) {
      downloadBtn->setText(tr("CHECK"));
      downloadBtn->setValue(tr("failed to check for update"));
    } else if (params.getBool("UpdaterFetchAvailable")) {
      downloadBtn->setText(tr("DOWNLOAD"));
      downloadBtn->setValue(tr("update available"));
    } else {
      QString lastUpdate = tr("never");
      auto tm = params.get("LastUpdateTime");
      if (!tm.empty()) {
        lastUpdate = timeAgo(QDateTime::fromString(QString::fromStdString(tm + "Z"), Qt::ISODate));
      }
      downloadBtn->setText(tr("CHECK"));
      downloadBtn->setValue(tr("up to date, last checked %1").arg(lastUpdate));
    }
    downloadBtn->setEnabled(true);
  }
  targetBranchBtn->setValue(QString::fromStdString(params.get("UpdaterTargetBranch")));

  // current + new versions
  versionLbl->setText(QString::fromStdString(params.get("UpdaterCurrentDescription")));
  versionLbl->setDescription(QString::fromStdString(params.get("UpdaterCurrentReleaseNotes")));

  installBtn->setVisible(!is_onroad && params.getBool("UpdateAvailable"));
  installBtn->setValue(QString::fromStdString(params.get("UpdaterNewDescription")));
  installBtn->setDescription(QString::fromStdString(params.get("UpdaterNewReleaseNotes")));

  update();
}
