#include "tools/cabana/streams/devicestream.h"

#include <memory>
#include <string>

#include "cereal/services.h"

#include <QButtonGroup>
#include <QFileInfo>
#include <QFormLayout>
#include <QMessageBox>
#include <QRadioButton>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QThread>

// DeviceStream

DeviceStream::DeviceStream(QObject *parent, QString address) : zmq_address(address), LiveStream(parent) {
}

DeviceStream::~DeviceStream() {
  if (!bridge_process)
    return;

  bridge_process->terminate();
  if (!bridge_process->waitForFinished(3000)) {
    bridge_process->kill();
    bridge_process->waitForFinished();
  }
}

void DeviceStream::start() {
  if (!zmq_address.isEmpty()) {
    bridge_process = new QProcess(this);
    QString bridge_path = QCoreApplication::applicationDirPath() + "/../../cereal/messaging/bridge";
    bridge_process->start(QFileInfo(bridge_path).absoluteFilePath(), QStringList { zmq_address, "/\"can/\"" });

    if (!bridge_process->waitForStarted()) {
      QMessageBox::warning(nullptr, tr("Error"), tr("Failed to start bridge: %1").arg(bridge_process->errorString()));
      return;
    }
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

AbstractStream *OpenDeviceWidget::open() {
  QString ip = ip_address->text().isEmpty() ? "127.0.0.1" : ip_address->text();
  bool msgq = group->checkedId() == 0;
  return new DeviceStream(qApp, msgq ? "" : ip);
}
