#include <QApplication>
#include <QCommandLineParser>

#include "common/prefix.h"
#include "tools/replay/consoleui.h"
#include "tools/replay/replay.h"

int main(int argc, char *argv[]) {
  QCoreApplication app(argc, argv);

  const std::tuple<QString, REPLAY_FLAGS, QString> flags[] = {
      {"dcam", REPLAY_FLAG_DCAM, "load driver camera"},
      {"ecam", REPLAY_FLAG_ECAM, "load wide road camera"},
      {"no-loop", REPLAY_FLAG_NO_LOOP, "stop at the end of the route"},
      {"no-cache", REPLAY_FLAG_NO_FILE_CACHE, "turn off local cache"},
      {"qcam", REPLAY_FLAG_QCAMERA, "load qcamera"},
      {"no-hw-decoder", REPLAY_FLAG_NO_HW_DECODER, "disable HW video decoding"},
      {"no-vipc", REPLAY_FLAG_NO_VIPC, "do not output video"},
  };

  QCommandLineParser parser;
  parser.setApplicationDescription("Mock openpilot components by publishing logged messages.");
  parser.addHelpOption();
  parser.addPositionalArgument("route", "the drive to replay. find your drives at connect.comma.ai");
  parser.addOption({{"a", "allow"}, "whitelist of services to send", "allow"});
  parser.addOption({{"b", "block"}, "blacklist of services to send", "block"});
  parser.addOption({{"c", "cache"}, "cache <n> segments in memory. default is 5", "n"});
  parser.addOption({{"s", "start"}, "start from <seconds>", "seconds"});
  parser.addOption({"demo", "use a demo route instead of providing your own"});
  parser.addOption({"data_dir", "local directory with routes", "data_dir"});
  parser.addOption({"prefix", "set OPENPILOT_PREFIX", "prefix"});
  for (auto &[name, _, desc] : flags) {
    parser.addOption({name, desc});
  }

  parser.process(app);
  const QStringList args = parser.positionalArguments();
  if (args.empty() && !parser.isSet("demo")) {
    parser.showHelp();
  }

  const QString route = args.empty() ? DEMO_ROUTE : args.first();
  QStringList allow = parser.value("allow").isEmpty() ? QStringList{} : parser.value("allow").split(",");
  QStringList block = parser.value("block").isEmpty() ? QStringList{} : parser.value("block").split(",");

  uint32_t replay_flags = REPLAY_FLAG_NONE;
  for (const auto &[name, flag, _] : flags) {
    if (parser.isSet(name)) {
      replay_flags |= flag;
    }
  }

  std::unique_ptr<OpenpilotPrefix> op_prefix;
  auto prefix = parser.value("prefix");
  if (!prefix.isEmpty()) {
    op_prefix.reset(new OpenpilotPrefix(prefix.toStdString()));
  }

  Replay *replay = new Replay(route, allow, block, nullptr, replay_flags, parser.value("data_dir"), &app);
  if (!parser.value("c").isEmpty()) {
    replay->setSegmentCacheLimit(parser.value("c").toInt());
  }
  if (!replay->load()) {
    return 0;
  }

  ConsoleUI console_ui(replay);
  replay->start(parser.value("start").toInt());
  return app.exec();
}
