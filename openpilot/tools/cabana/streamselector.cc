#include "tools/cabana/streamselector.h"

#include <filesystem>
#include <fstream>

#include <QFileDialog>
#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QTimer>

#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/routesdialog.h"
#include "tools/cabana/utils/qtutil.h"

// OpenReplayWidget

OpenReplayWidget::OpenReplayWidget(QWidget *parent) : AbstractOpenStreamWidget(parent) {
  QGridLayout *grid_layout = new QGridLayout(this);
  grid_layout->addWidget(new QLabel(tr("Route")), 0, 0);
  grid_layout->addWidget(route_edit = new QLineEdit(this), 0, 1);
  route_edit->setPlaceholderText(tr("Enter route name or browse for local/remote route"));
  auto browse_remote_btn = new QPushButton(tr("Remote route..."), this);
  grid_layout->addWidget(browse_remote_btn, 0, 2);
  auto browse_local_btn = new QPushButton(tr("Local route..."), this);
  grid_layout->addWidget(browse_local_btn, 0, 3);

  QHBoxLayout *camera_layout = new QHBoxLayout();
  for (auto c : {tr("Road camera"), tr("Driver camera"), tr("Wide road camera")})
    camera_layout->addWidget(cameras.emplace_back(new QCheckBox(c, this)));
  cameras[0]->setChecked(true);
  camera_layout->addStretch(1);
  grid_layout->addItem(camera_layout, 1, 1);

  setMinimumWidth(550);
  QObject::connect(browse_local_btn, &QPushButton::clicked, [=]() {
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Local Route"), QString::fromStdString(settings.last_route_dir));
    if (!dir.isEmpty()) {
      route_edit->setText(dir);
      settings.last_route_dir = std::filesystem::absolute(dir.toStdString()).parent_path().string();
    }
  });
  QObject::connect(browse_remote_btn, &QPushButton::clicked, [this]() {
    RoutesDialog route_dlg(this);
    if (route_dlg.exec()) {
      route_edit->setText(QString::fromStdString(route_dlg.route()));
    }
  });
}

AbstractStream *OpenReplayWidget::open() {
  QString route = route_edit->text();
  QString data_dir;
  if (int idx = route.lastIndexOf('/'); idx != -1 && util::file_exists(route.toStdString())) {
    data_dir = route.mid(0, idx + 1);
    route = route.mid(idx + 1);
  }

  bool is_valid_format = Route::parseRoute(route.toStdString()).str.size() > 0;
  if (!is_valid_format) {
    QMessageBox::warning(nullptr, tr("Warning"), tr("Invalid route format: '%1'").arg(route));
  } else {
    auto replay_stream = std::make_unique<ReplayStream>();
    Connection err = replay_stream->error.connect([](const std::string &msg) {
      QMessageBox::warning(nullptr, tr("Error"), QString::fromStdString(msg));
    });
    uint32_t flags = REPLAY_FLAG_NONE;
    if (cameras[1]->isChecked()) flags |= REPLAY_FLAG_CABIN_CAMERA;
    if (cameras[2]->isChecked()) flags |= REPLAY_FLAG_WIDE_ROAD;
    if (flags == REPLAY_FLAG_NONE && !cameras[0]->isChecked()) flags = REPLAY_FLAG_NO_VIPC;

    if (replay_stream->loadRoute(route.toStdString(), data_dir.toStdString(), flags)) {
      return replay_stream.release();
    }
  }
  return nullptr;
}

// OpenPandaWidget

static const uint32_t speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U};
static const uint32_t data_speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U, 2000U, 5000U};

OpenPandaWidget::OpenPandaWidget(QWidget *parent) : AbstractOpenStreamWidget(parent) {
  form_layout = new QFormLayout(this);
  if (can && dynamic_cast<PandaStream *>(can) != nullptr) {
    form_layout->addWidget(new QLabel(tr("Already connected to %1.").arg(QString::fromStdString(can->routeName()))));
    form_layout->addWidget(new QLabel("Close the current connection via [File menu -> Close Stream] before connecting to another Panda."));
    QTimer::singleShot(0, [this]() { emit enableOpenButton(false); });
    return;
  }

  QHBoxLayout *serial_layout = new QHBoxLayout();
  serial_layout->addWidget(serial_edit = new QComboBox());

  QPushButton *refresh = new QPushButton(tr("Refresh"));
  refresh->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
  serial_layout->addWidget(refresh);
  form_layout->addRow(tr("Serial"), serial_layout);

  QObject::connect(refresh, &QPushButton::clicked, this, &OpenPandaWidget::refreshSerials);
  QObject::connect(serial_edit, &QComboBox::currentTextChanged, this, &OpenPandaWidget::buildConfigForm);

  // Populate serials
  refreshSerials();
  buildConfigForm();
}

void OpenPandaWidget::refreshSerials() {
  serial_edit->clear();
  for (auto serial : Panda::list()) {
    serial_edit->addItem(QString::fromStdString(serial));
  }
}

void OpenPandaWidget::buildConfigForm() {
  for (int i = form_layout->rowCount() - 1; i > 0; --i) {
    form_layout->removeRow(i);
  }

  QString serial = serial_edit->currentText();
  bool has_fd = false;
  bool has_panda = !serial.isEmpty();
  if (has_panda) {
    try {
      Panda panda(serial.toStdString());
      has_fd = (panda.hw_type == cereal::PandaState::PandaType::RED_PANDA) || (panda.hw_type == cereal::PandaState::PandaType::RED_PANDA_V2);
    } catch (const std::exception& e) {
      fprintf(stderr, "failed to open panda %s\n", serial.toUtf8().constData());
      has_panda = false;
    }
  }

  if (has_panda) {
    config.serial = serial.toStdString();
    config.bus_config.resize(3);
    for (int i = 0; i < config.bus_config.size(); i++) {
      QHBoxLayout *bus_layout = new QHBoxLayout;

      // CAN Speed
      bus_layout->addWidget(new QLabel(tr("CAN Speed (kbps):")));
      QComboBox *can_speed = new QComboBox;
      for (int j = 0; j < std::size(speeds); j++) {
        can_speed->addItem(QString::number(speeds[j]));

        if (data_speeds[j] == config.bus_config[i].can_speed_kbps) {
          can_speed->setCurrentIndex(j);
        }
      }
      QObject::connect(can_speed, qOverload<int>(&QComboBox::currentIndexChanged), [=](int index) {config.bus_config[i].can_speed_kbps = speeds[index];});
      bus_layout->addWidget(can_speed);

      // CAN-FD Speed
      if (has_fd) {
        QCheckBox *enable_fd = new QCheckBox("CAN-FD");
        bus_layout->addWidget(enable_fd);
        bus_layout->addWidget(new QLabel(tr("Data Speed (kbps):")));
        QComboBox *data_speed = new QComboBox;
        for (int j = 0; j < std::size(data_speeds); j++) {
          data_speed->addItem(QString::number(data_speeds[j]));

          if (data_speeds[j] == config.bus_config[i].data_speed_kbps) {
            data_speed->setCurrentIndex(j);
          }
        }

        data_speed->setEnabled(false);
        bus_layout->addWidget(data_speed);

        QObject::connect(data_speed, qOverload<int>(&QComboBox::currentIndexChanged), [=](int index) {config.bus_config[i].data_speed_kbps = data_speeds[index];});
        QObject::connect(enable_fd, &QCheckBox::stateChanged, data_speed, &QComboBox::setEnabled);
        QObject::connect(enable_fd, &QCheckBox::stateChanged, [=](int state) {config.bus_config[i].can_fd = (bool)state;});
      }

      form_layout->addRow(tr("Bus %1:").arg(i), bus_layout);
    }
  } else {
    config.serial = "";
    form_layout->addWidget(new QLabel(tr("No panda found")));
  }
}

AbstractStream *OpenPandaWidget::open() {
  try {
    return new PandaStream(config);
  } catch (std::exception &e) {
    QMessageBox::warning(nullptr, tr("Warning"), tr("Failed to connect to panda: '%1'").arg(e.what()));
    return nullptr;
  }
}

// OpenDeviceWidget

OpenDeviceWidget::OpenDeviceWidget(QWidget *parent) : AbstractOpenStreamWidget(parent) {
  QRadioButton *msgq = new QRadioButton(tr("MSGQ"));
  QRadioButton *zmq = new QRadioButton(tr("ZMQ"));
  ip_address = new QLineEdit(this);
  ip_address->setPlaceholderText(tr("Enter device Ip Address"));
  ip_address->setValidator(new IpAddressValidator(this));

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

AbstractStream *OpenDeviceWidget::open() {
  std::string ip = ip_address->text().isEmpty() ? "127.0.0.1" : ip_address->text().toStdString();
  bool msgq = group->checkedId() == 0;
  return new DeviceStream(msgq ? "" : ip);
}

#ifdef __linux__
// OpenSocketCanWidget

OpenSocketCanWidget::OpenSocketCanWidget(QWidget *parent) : AbstractOpenStreamWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addStretch(1);

  QFormLayout *form_layout = new QFormLayout();

  QHBoxLayout *device_layout = new QHBoxLayout();
  device_edit = new QComboBox();
  device_edit->setFixedWidth(300);
  device_layout->addWidget(device_edit);

  QPushButton *refresh = new QPushButton(tr("Refresh"));
  refresh->setFixedWidth(100);
  device_layout->addWidget(refresh);
  form_layout->addRow(tr("Device"), device_layout);
  main_layout->addLayout(form_layout);

  main_layout->addStretch(1);

  QObject::connect(refresh, &QPushButton::clicked, this, &OpenSocketCanWidget::refreshDevices);
  QObject::connect(device_edit, &QComboBox::currentTextChanged, this, [=]{ config.device = device_edit->currentText().toStdString(); });

  // Populate devices
  refreshDevices();
}

void OpenSocketCanWidget::refreshDevices() {
  device_edit->clear();
  // Scan /sys/class/net/ for CAN interfaces (type 280 = ARPHRD_CAN)
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator("/sys/class/net", ec)) {
    std::ifstream type_file(entry.path() / "type");
    int type = 0;
    if (type_file >> type && type == 280) {
      device_edit->addItem(QString::fromStdString(entry.path().filename().string()));
    }
  }
}

AbstractStream *OpenSocketCanWidget::open() {
  try {
    return new SocketCanStream(config);
  } catch (std::exception &e) {
    QMessageBox::warning(nullptr, tr("Warning"), tr("Failed to connect to SocketCAN device: '%1'").arg(e.what()));
    return nullptr;
  }
}
#endif

// StreamSelector

StreamSelector::StreamSelector(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Open stream"));
  QVBoxLayout *layout = new QVBoxLayout(this);
  tab = new QTabWidget(this);
  layout->addWidget(tab);

  QHBoxLayout *dbc_layout = new QHBoxLayout();
  dbc_file = new QLineEdit(this);
  dbc_file->setReadOnly(true);
  dbc_file->setPlaceholderText(tr("Choose a dbc file to open"));
  QPushButton *file_btn = new QPushButton(tr("Browse..."));
  dbc_layout->addWidget(new QLabel(tr("dbc File")));
  dbc_layout->addWidget(dbc_file);
  dbc_layout->addWidget(file_btn);
  layout->addLayout(dbc_layout);

  QFrame *line = new QFrame(this);
  line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
  layout->addWidget(line);

  btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
  layout->addWidget(btn_box);

  addStreamWidget(new OpenReplayWidget, tr("&Replay"));
  addStreamWidget(new OpenPandaWidget, tr("&Panda"));
#ifdef __linux__
  if (SocketCanStream::available()) {
    addStreamWidget(new OpenSocketCanWidget, tr("&SocketCAN"));
  }
#endif
  addStreamWidget(new OpenDeviceWidget, tr("&Device"));

  QObject::connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(btn_box, &QDialogButtonBox::accepted, [=]() {
    setEnabled(false);
    if (stream_ = ((AbstractOpenStreamWidget *)tab->currentWidget())->open(); stream_) {
      accept();
    }
    setEnabled(true);
  });
  QObject::connect(file_btn, &QPushButton::clicked, [this]() {
    QString fn = QFileDialog::getOpenFileName(this, tr("Open File"), QString::fromStdString(settings.last_dir), "DBC (*.dbc)");
    if (!fn.isEmpty()) {
      dbc_file->setText(fn);
      settings.last_dir = std::filesystem::absolute(fn.toStdString()).parent_path().string();
    }
  });
}

void StreamSelector::addStreamWidget(AbstractOpenStreamWidget *w, const QString &title) {
  tab->addTab(w, title);
  auto open_btn = btn_box->button(QDialogButtonBox::Open);
  QObject::connect(w, &AbstractOpenStreamWidget::enableOpenButton, open_btn, &QPushButton::setEnabled);
}
