#include <QApplication>
#include <QCommandLineParser>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimer>

#include "tools/cabana/mainwin.h"
#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/pandastream.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/streams/socketcanstream.h"

// Load fingerprint to DBC mapping
static QJsonDocument loadFingerprintMapping() {
  QFile json_file(QCoreApplication::applicationDirPath() + "/dbc/car_fingerprint_to_dbc.json");
  if (json_file.open(QIODevice::ReadOnly)) {
    return QJsonDocument::fromJson(json_file.readAll());
  }
  return QJsonDocument();
}

// Dump signals to JSON for pycabana validation
static int dumpSignals(ReplayStream *stream, const QString &dbc_file, const QString &output_file) {
  // Use QEventLoop to properly wait for segment loading
  QEventLoop loop;
  QTimer timeout;
  timeout.setSingleShot(true);

  // Connect to eventsMerged signal from the stream
  bool data_ready = false;
  QObject::connect(stream, &AbstractStream::eventsMerged, [&]() {
    if (!stream->allEvents().empty()) {
      data_ready = true;
      loop.quit();
    }
  });

  // Timeout after 120 seconds (network download can be slow)
  QObject::connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
  timeout.start(120000);

  // Progress timer to show we're still running
  QTimer progress;
  int elapsed = 0;
  QObject::connect(&progress, &QTimer::timeout, [&]() {
    elapsed += 5;
    qInfo() << "Still waiting for data..." << elapsed << "seconds elapsed";
  });
  progress.start(5000);

  // Start the replay to trigger segment loading
  qInfo() << "Starting replay and waiting for data...";
  stream->start();

  // Run the event loop until data is ready or timeout
  loop.exec();

  if (!data_ready) {
    qCritical() << "Timeout waiting for segment data";
    return 1;
  }
  qInfo() << "Data loaded, processing" << stream->allEvents().size() << "events...";

  // Determine DBC file - use provided or infer from fingerprint
  QString actual_dbc = dbc_file;
  if (actual_dbc.isEmpty()) {
    QString fingerprint = stream->carFingerprint();
    if (fingerprint.isEmpty()) {
      qCritical() << "No --dbc provided and no car fingerprint in route";
      return 1;
    }
    auto fingerprint_map = loadFingerprintMapping();
    if (!fingerprint_map.object().contains(fingerprint)) {
      qCritical() << "No --dbc provided and unknown fingerprint:" << fingerprint;
      qCritical() << "Use --dbc to specify DBC file manually";
      return 1;
    }
    actual_dbc = fingerprint_map[fingerprint].toString();
    qInfo() << "Inferred DBC from fingerprint" << fingerprint << "->" << actual_dbc;
  }

  // Load DBC
  QString dbc_path = actual_dbc;
  if (!dbc_path.endsWith(".dbc")) {
    dbc_path += ".dbc";
  }
  if (!QFile::exists(dbc_path)) {
    dbc_path = QString("%1/%2").arg(OPENDBC_FILE_PATH, dbc_path);
  }
  if (!QFile::exists(dbc_path)) {
    qCritical() << "DBC file not found:" << actual_dbc;
    return 1;
  }
  dbc()->open(SOURCE_ALL, dbc_path);

  QJsonArray events_json;
  int count = 0;
  const int max_events = 10000;  // Limit for reasonable file size

  for (auto e : stream->allEvents()) {
    if (count >= max_events) break;

    MessageId msg_id = {.source = e->src, .address = e->address};
    auto msg = dbc()->msg(msg_id);
    if (!msg) continue;

    QJsonObject event_obj;
    event_obj["time"] = stream->toSeconds(e->mono_time);
    event_obj["address"] = QString("0x%1").arg(e->address, 0, 16);
    event_obj["bus"] = e->src;
    event_obj["data"] = QString(QByteArray::fromRawData((const char *)e->dat, e->size).toHex().toUpper());

    QJsonObject signals_obj;
    for (auto s : msg->sigs) {
      double value = 0;
      if (s->getValue(e->dat, e->size, &value)) {
        signals_obj[s->name] = value;
      }
    }
    event_obj["signals"] = signals_obj;
    events_json.append(event_obj);
    count++;
  }

  QJsonObject root;
  root["route"] = stream->routeName();
  root["dbc"] = actual_dbc;
  root["fingerprint"] = stream->carFingerprint();
  root["event_count"] = count;
  root["events"] = events_json;

  QFile out(output_file.isEmpty() ? "/dev/stdout" : output_file);
  if (!out.open(QIODevice::WriteOnly)) {
    qCritical() << "Cannot open output file:" << output_file;
    return 1;
  }
  out.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
  qInfo() << "Wrote" << count << "events to" << (output_file.isEmpty() ? "stdout" : output_file);
  return 0;
}

int main(int argc, char *argv[]) {
  QCoreApplication::setApplicationName("Cabana");
  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
  initApp(argc, argv, false);

  // Pre-parse to check for --dump (need different app type)
  bool dump_mode = false;
  for (int i = 1; i < argc; i++) {
    if (QString(argv[i]) == "--dump") {
      dump_mode = true;
      break;
    }
  }

  // Use QCoreApplication for dump mode (no GUI needed)
  QScopedPointer<QCoreApplication> app(dump_mode ?
    new QCoreApplication(argc, argv) :
    new QApplication(argc, argv));

  if (!dump_mode) {
    static_cast<QApplication*>(app.data())->setApplicationDisplayName("Cabana");
    static_cast<QApplication*>(app.data())->setWindowIcon(QIcon(":cabana-icon.png"));
    utils::setTheme(settings.theme);
  }

  UnixSignalHandler signalHandler;

  QCommandLineParser cmd_parser;
  cmd_parser.addHelpOption();
  cmd_parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  cmd_parser.addOption({"demo", "use a demo route instead of providing your own"});
  cmd_parser.addOption({"auto", "Auto load the route from the best available source (no video): internal, openpilotci, comma_api, car_segments, testing_closet"});
  cmd_parser.addOption({"qcam", "load qcamera"});
  cmd_parser.addOption({"ecam", "load wide road camera"});
  cmd_parser.addOption({"dcam", "load driver camera"});
  cmd_parser.addOption({"msgq", "read can messages from the msgq"});
  cmd_parser.addOption({"panda", "read can messages from panda"});
  cmd_parser.addOption({"panda-serial", "read can messages from panda with given serial", "panda-serial"});
  if (SocketCanStream::available()) {
    cmd_parser.addOption({"socketcan", "read can messages from given SocketCAN device", "socketcan"});
  }
  cmd_parser.addOption({"zmq", "read can messages from zmq at the specified ip-address", "ip-address"});
  cmd_parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  cmd_parser.addOption({"no-vipc", "do not output video"});
  cmd_parser.addOption({"dbc", "dbc file to open", "dbc"});
  cmd_parser.addOption({"dump", "dump decoded signals to JSON (infers DBC from fingerprint, or use --dbc)"});
  cmd_parser.addOption({"dump-output", "output file for --dump (default: stdout)", "file"});
  cmd_parser.process(*app);

  AbstractStream *stream = nullptr;

  if (cmd_parser.isSet("msgq")) {
    stream = new DeviceStream(app.data());
  } else if (cmd_parser.isSet("zmq")) {
    stream = new DeviceStream(app.data(), cmd_parser.value("zmq"));
  } else if (cmd_parser.isSet("panda") || cmd_parser.isSet("panda-serial")) {
    try {
      stream = new PandaStream(app.data(), {.serial = cmd_parser.value("panda-serial")});
    } catch (std::exception &e) {
      qWarning() << e.what();
      return 0;
    }
  } else if (SocketCanStream::available() && cmd_parser.isSet("socketcan")) {
    stream = new SocketCanStream(app.data(), {.device = cmd_parser.value("socketcan")});
  } else {
    uint32_t replay_flags = REPLAY_FLAG_NONE;
    if (cmd_parser.isSet("ecam")) replay_flags |= REPLAY_FLAG_ECAM;
    if (cmd_parser.isSet("qcam")) replay_flags |= REPLAY_FLAG_QCAMERA;
    if (cmd_parser.isSet("dcam")) replay_flags |= REPLAY_FLAG_DCAM;
    if (cmd_parser.isSet("no-vipc")) replay_flags |= REPLAY_FLAG_NO_VIPC;

    const QStringList args = cmd_parser.positionalArguments();
    QString route;
    if (args.size() > 0) {
      route = args.first();
    } else if (cmd_parser.isSet("demo")) {
      route = DEMO_ROUTE;
    }
    if (!route.isEmpty()) {
      auto replay_stream = std::make_unique<ReplayStream>(app.data());
      bool auto_source = cmd_parser.isSet("auto");
      if (!replay_stream->loadRoute(route, cmd_parser.value("data_dir"), replay_flags, auto_source)) {
        return 0;
      }
      // Handle --dump mode
      if (cmd_parser.isSet("dump")) {
        return dumpSignals(replay_stream.get(), cmd_parser.value("dbc"), cmd_parser.value("dump-output"));
      }

      stream = replay_stream.release();
    }
  }

  // Dump mode only works with replay
  if (cmd_parser.isSet("dump")) {
    qCritical() << "--dump only works with replay routes";
    return 1;
  }

  MainWindow w(stream, cmd_parser.value("dbc"));
  return app->exec();
}
