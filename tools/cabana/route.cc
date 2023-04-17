#include "tools/cabana/route.h"

#include <QButtonGroup>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QTabWidget>

#include "common/prefix.h"
#include "tools/cabana/streams/livestream.h"
// #include "tools/cabana/streams/devicestream.h"
// #include "tools/cabana/streams/pandastream.h"
#include "tools/cabana/streams/replaystream.h"

static std::unique_ptr<OpenpilotPrefix> op_prefix;

// StreamDialog

StreamDialog::StreamDialog(AbstractStream **stream, QWidget *parent) : QDialog(parent) {
  assert(*stream == nullptr);

  setWindowTitle(tr("Open stream"));
  QFormLayout *main_layout = new QFormLayout(this);
  QTabWidget *tab = new QTabWidget(this);
  tab->addTab(route_widget = new OpenRouteWidget(stream, this), tr("Route"));
  tab->addTab(panda_widget = new OpenPandaWidget(stream, this), tr("Panda"));
  tab->addTab(device_widget = new OpenDeviceWidget(stream, this), tr("Device"));
  main_layout->addWidget(tab);

  btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
  main_layout->addWidget(btn_box);

  QObject::connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(btn_box, &QDialogButtonBox::accepted, [=]() {
    if (((AbstractOpenWidget *)tab->currentWidget())->open()) {
      accept();
    }
  });
}

// OpenRouteDiloag

OpenRouteDialog::OpenRouteDialog(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Open Route"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addWidget(route_widget = new OpenRouteWidget(&can, this));
  auto btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
  main_layout->addWidget(btn_box);

  QObject::connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(btn_box, &QDialogButtonBox::accepted, [=]() {
    if (route_widget->open()) {
      accept();
    }
  });
}

// OpenPandaWidget

OpenDeviceWidget::OpenDeviceWidget(AbstractStream **stream, QWidget *parent) : AbstractOpenWidget(stream, parent) {
  QRadioButton *msgq = new QRadioButton(tr("MSGQ"));
  QRadioButton *zmq = new QRadioButton(tr("ZMQ"));
  ip_address = new QLineEdit(this);
  ip_address->setPlaceholderText(tr("Enter the IpAddress"));
  QString ip_range = "(?:[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])";
  QString pattern("^" + ip_range + "\\." + ip_range + "\\." + ip_range + "\\." + ip_range + "$");
  QRegularExpression re(pattern);
  ip_address->setValidator(new QRegularExpressionValidator(re, this));

  group = new QButtonGroup(this);
  group->addButton(msgq, 0);
  group->addButton(zmq, 1);

  QFormLayout *form_layout = new QFormLayout(this);
  form_layout->addRow(msgq);
  form_layout->addRow(zmq, ip_address);
  QObject::connect(group, qOverload<QAbstractButton *, bool>(&QButtonGroup::buttonToggled), [=](QAbstractButton *button, bool checked) {
    ip_address->setEnabled(button == zmq && checked);
  });
  zmq->setChecked(true);
}

bool OpenDeviceWidget::open() {
  QString ip = ip_address->text().isEmpty() ? "127.0.0.1" : ip_address->text();
  *stream = new LiveStream(qApp, group->checkedId() == 0 ? "" : ip);
  return true;
}

// OpenPandaWidget

OpenPandaWidget::OpenPandaWidget(AbstractStream **stream, QWidget *parent) : AbstractOpenWidget(stream, parent) {
  QFormLayout *form_layout = new QFormLayout(this);
  serial_edit = new QLineEdit(this);
  serial_edit->setPlaceholderText(tr("Leave empty to use default serial"));
  form_layout->addRow(tr("Serial"), serial_edit);
}

bool OpenPandaWidget::open() {
  // PandaStreamConfig config = {.serial = serial_edit->text()};
  // stream.reset(new PandaStream(qApp, config));
  return true;
}

// OpenRouteWidget

OpenRouteWidget::OpenRouteWidget(AbstractStream **stream, QWidget *parent) : AbstractOpenWidget(stream, parent) {
  // TODO: get route list from api.comma.ai
  QGridLayout *grid_layout = new QGridLayout();
  grid_layout->addWidget(new QLabel(tr("Name")), 0, 0);
  grid_layout->addWidget(route_edit = new QLineEdit(this), 0, 1);
  route_edit->setPlaceholderText(tr("Enter remote route name or click browse to select a local route"));
  auto file_btn = new QPushButton(tr("Browse..."), this);
  grid_layout->addWidget(file_btn, 0, 2);

  grid_layout->addWidget(new QLabel(tr("Video")), 1, 0);
  grid_layout->addWidget(choose_video_cb = new QComboBox(this), 1, 1);
  QString items[] = {tr("No Video"), tr("Road Camera"), tr("Wide Road Camera"), tr("Driver Camera"), tr("QCamera")};
  for (int i = 0; i < std::size(items); ++i) {
    choose_video_cb->addItem(items[i]);
  }
  choose_video_cb->setCurrentIndex(1);  // default is road camera;

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(grid_layout);
  main_layout->addStretch(0);
  setMinimumSize({550, 120});

  QObject::connect(file_btn, &QPushButton::clicked, [=]() {
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Local Route"), settings.last_route_dir);
    if (!dir.isEmpty()) {
      route_edit->setText(dir);
      settings.last_route_dir = QFileInfo(dir).absolutePath();
    }
  });
}

bool OpenRouteWidget::open() {
  QString route = route_edit->text();
  QString data_dir;
  if (int idx = route.lastIndexOf('/'); idx != -1) {
    data_dir = route.mid(0, idx + 1);
    route = route.mid(idx + 1);
  }

  failed_to_load = true;
  bool is_valid_format = Route::parseRoute(route).str.size() > 0;
  if (!is_valid_format) {
    QMessageBox::warning(nullptr, tr("Warning"), tr("Invalid route format: '%1'").arg(route));
  } else {
    // TODO: Remove when OpenpilotPrefix supports ZMQ
#ifndef __APPLE__
    op_prefix.reset(new OpenpilotPrefix());
#endif
    uint32_t flags[] = {REPLAY_FLAG_NO_VIPC, REPLAY_FLAG_NONE, REPLAY_FLAG_ECAM, REPLAY_FLAG_DCAM, REPLAY_FLAG_QCAMERA};
    ReplayStream *replay_stream = *stream ? (ReplayStream *)*stream : new ReplayStream(qApp);
    failed_to_load = !replay_stream->loadRoute(route, data_dir, flags[choose_video_cb->currentIndex()]);
    if (failed_to_load) {
      if (replay_stream != *stream) {
        delete replay_stream;
      }
      QMessageBox::warning(nullptr, tr("Warning"), tr("Failed to load route: '%1'").arg(route));
    } else {
      *stream = replay_stream;
    }
  }
  return !failed_to_load;
}
