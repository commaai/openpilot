#include "tools/cabana/streams/devicestream.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

#include "openpilot/cereal/services.h"

#include <QButtonGroup>
#include <QFileInfo>
#include <QFormLayout>
#include <QMessageBox>
#include <QRadioButton>
#include <QThread>

#include "tools/cabana/utils/util.h"

// DeviceStream

DeviceStream::DeviceStream(QObject *parent, QString address) : zmq_address(address), LiveStream(parent) {
}

DeviceStream::~DeviceStream() {
  stopBridge();
}

void DeviceStream::stopBridge() {
  if (bridge_pid <= 0) return;

  ::kill(bridge_pid, SIGTERM);
  for (int i = 0; i < 30; ++i) {
    int status = 0;
    pid_t r = ::waitpid(bridge_pid, &status, WNOHANG);
    if (r == bridge_pid || (r < 0 && errno == ECHILD)) {
      bridge_pid = -1;
      return;
    }
    usleep(100000);  // 100ms, up to ~3s
  }
  ::kill(bridge_pid, SIGKILL);
  ::waitpid(bridge_pid, nullptr, 0);
  bridge_pid = -1;
}

void DeviceStream::start() {
  if (!zmq_address.isEmpty()) {
    stopBridge();
    QString bridge_path = QFileInfo(QCoreApplication::applicationDirPath() +
                                    "/../../openpilot/cereal/messaging/bridge").absoluteFilePath();
    const std::string path = bridge_path.toStdString();
    const std::string addr = zmq_address.toStdString();
    const char *can_filter = "/\"can/\"";

    pid_t pid = ::fork();
    if (pid == 0) {
      execl(path.c_str(), path.c_str(), addr.c_str(), can_filter, static_cast<char *>(nullptr));
      _exit(127);
    }
    if (pid < 0) {
      QMessageBox::warning(nullptr, tr("Error"),
                           tr("Failed to start bridge: %1").arg(QString::fromLocal8Bit(strerror(errno))));
      return;
    }
    bridge_pid = pid;
  }

  LiveStream::start();
}

void DeviceStream::streamThread() {
  zmq_address.isEmpty() ? unsetenv("ZMQ") : setenv("ZMQ", "1", 1);

  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can", "127.0.0.1", false, true, services.at("can").queue_size));
  assert(sock != NULL);
  // run as fast as messages come in
  while (!QThread::currentThread()->isInterruptionRequested()) {
    std::unique_ptr<Message> msg(sock->receive(true));
    if (!msg) {
      QThread::msleep(50);
      continue;
    }
    handleEvent(kj::ArrayPtr<capnp::word>((capnp::word*)msg->getData(), msg->getSize() / sizeof(capnp::word)));
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
  QString ip = ip_address->text().isEmpty() ? "127.0.0.1" : ip_address->text();
  bool msgq = group->checkedId() == 0;
  return new DeviceStream(qApp, msgq ? "" : ip);
}
