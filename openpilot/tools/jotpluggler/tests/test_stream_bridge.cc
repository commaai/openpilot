#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "tools/jotpluggler/app.h"
#include "tools/jotpluggler/common.h"
#include "tools/jotpluggler/stream_bridge.h"

#include "openpilot/cereal/messaging/messaging.h"
#include "openpilot/cereal/services.h"
#include "openpilot/common/util.h"
#include "msgq_repo/msgq/event.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <thread>
#include <sys/stat.h>
#include <sys/wait.h>
using namespace std::chrono_literals;
namespace {
template <typename Predicate>
bool wait_until(std::chrono::milliseconds timeout, Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (predicate()) return true;
    std::this_thread::sleep_for(10ms);
  } while (std::chrono::steady_clock::now() < deadline);
  return predicate();
}
std::optional<std::string> env_value(const char *name) {
  const char *value = std::getenv(name);
  return value ? std::make_optional<std::string>(value) : std::nullopt;
}
void restore_env(const char *name, const std::optional<std::string> &value) { value ? setenv(name, value->c_str(), 1) : unsetenv(name); }
std::unique_ptr<PubSocket> publisher(Context *context) {
  return std::unique_ptr<PubSocket>(PubSocket::create(context, "deviceState", true, services.at("deviceState").queue_size));
}
int send_state(PubSocket *socket, bool started) {
  MessageBuilder message;
  message.initEvent().initDeviceState().setStarted(started);
  auto bytes = message.toBytes();
  return socket->send(reinterpret_cast<char *>(bytes.begin()), bytes.size());
}
std::string take_error(StreamPoller *poller) { std::string error; poller->consume(nullptr, &error); return error; }
bool take_started(StreamPoller *poller) {
  StreamExtractBatch batch;
  if (!poller->consume(&batch, nullptr)) return false;
  for (const RouteSeries &series : batch.series) {
    if (series.path == "/deviceState/started" &&
        std::any_of(series.values.begin(), series.values.end(), [](double value) { return value > 0.5; })) return true;
  }
  return false;
}
StreamSourceConfig local_source() { return {.kind = StreamSourceKind::CerealLocal, .address = "127.0.0.1"}; }
StreamSourceConfig remote_source(std::string address) { return {.kind = StreamSourceKind::CerealRemote, .address = std::move(address)}; }
}  // namespace
TEST_CASE("stream bridge owns its prefix and child lifecycle") {
  const auto original = env_value("OPENPILOT_PREFIX");
  ScopedMsgqPrefix prefix;
  StreamBridgeProcess process(20ms);
  REQUIRE_THROWS_WITH(process.start("/does/not/exist"), Catch::Matchers::Contains("Failed to start"));
  struct stat info = {};
  REQUIRE(stat(prefix.path().c_str(), &info) == 0);
  REQUIRE((info.st_mode & 0777) == 0700);
  const auto pid_file = prefix.path() / "child.pid";
  prefix.activate();
  process.start("/bin/sh", {"-c", "test \"$OPENPILOT_PREFIX\" = \"$1\" || exit 1; echo $$ > \"$2\"; trap '' TERM; while :; do :; done",
                                    "bridge", prefix.prefix(), pid_file.string()});
  prefix.restore();
  REQUIRE(env_value("OPENPILOT_PREFIX") == original);
  REQUIRE(wait_until(1s, [&] { return std::filesystem::exists(pid_file); }));
  const pid_t child_pid = std::stoi(util::read_file(pid_file.string()));
  REQUIRE_THROWS(process.start("/usr/bin/false"));
  process.stop();
  REQUIRE_FALSE(process.owned());
  errno = 0;
  REQUIRE((kill(child_pid, 0) == -1 && errno == ESRCH));
  int status = 0;
  REQUIRE((waitpid(child_pid, &status, WNOHANG) == -1 && errno == ECHILD));
  process.stop();
}
TEST_CASE("bridge failures are safe and a StreamPoller can restart") {
  const auto original = env_value("OPENPILOT_PREFIX");
  ScopedMsgqPrefix diagnostics;
  ScopedMsgqPrefix missing;
  const auto log = diagnostics.path() / "bridge.stderr";
  missing.activate();
  std::filesystem::remove_all(missing.path());
  StreamBridgeProcess bridge;
  std::string bridge_error;
  try {
    bridge.start("/bin/sh", {"-c", "exec \"$1\" \"$2\" \"$3\" >/dev/null 2>\"$4\"", "capture",
      (repo_root() / "openpilot/cereal/messaging/bridge").string(), "127.0.0.2", stream_bridge_whitelist({"deviceState"}), log.string()});
    wait_until(1s, [&] {
      try { bridge.check_running(); } catch (const std::exception &err) { bridge_error = err.what(); return true; }
      return false;
    });
  } catch (const std::exception &err) { bridge_error = err.what(); }
  missing.restore();
  bridge.stop();
  REQUIRE_THAT(bridge_error, Catch::Matchers::Contains("status 1"));
  REQUIRE(util::read_file(log.string()).find("MSGQ publisher") != std::string::npos);
  StreamPoller poller;
  ScopedMsgqPrefix absent;
  absent.activate();
  std::filesystem::remove_all(absent.path());
  poller.start(local_source(), 5.0, "");
  REQUIRE(wait_until(2s, [&] { return !poller.snapshot().active; }));
  absent.restore();
  REQUIRE_THAT(take_error(&poller), Catch::Matchers::Contains("Failed to connect cereal service"));
  poller.start(remote_source("not a valid ZMQ host"), 5.0, "");
  REQUIRE(wait_until(2s, [&] { return !poller.snapshot().active; }));
  REQUIRE(poller.consume(nullptr, nullptr));
  REQUIRE(env_value("OPENPILOT_PREFIX") == original);
  ScopedMsgqPrefix local;
  local.activate();
  const auto previous_fake = env_value("CEREAL_FAKE");
  const auto previous_fake_prefix = env_value("CEREAL_FAKE_PREFIX");
  setenv("CEREAL_FAKE", "1", 1);
  setenv("CEREAL_FAKE_PREFIX", local.prefix().c_str(), 1);
  SocketEventHandle fake_event("deviceState", local.prefix());
  Context context;
  auto pub = publisher(&context);
  REQUIRE(pub != nullptr);
  poller.start(local_source(), 5.0, "");
  const bool connected = wait_until(2s, [&] { return poller.snapshot().connected || !poller.snapshot().active; }) && poller.snapshot().connected;
  restore_env("CEREAL_FAKE", previous_fake);
  restore_env("CEREAL_FAKE_PREFIX", previous_fake_prefix);
  local.restore();
  REQUIRE(connected);
  REQUIRE(take_error(&poller).empty());
  fake_event.set_enabled(true);
  const bool fake_called = wait_until(2s, [&] { return fake_event.recv_called().peek(); });
  const int sent = send_state(pub.get(), true);
  fake_event.set_enabled(false);
  fake_event.recv_ready().set();
  REQUIRE(fake_called);
  REQUIRE(sent > 0);
  REQUIRE(wait_until(2s, [&] { return take_started(&poller); }));
  poller.stop();
}
TEST_CASE("concurrent setup is isolated and bridge death is surfaced") {
  const auto original = env_value("OPENPILOT_PREFIX");
  ScopedMsgqPrefix scratch;
  const auto script = scratch.path() / "short-bridge";
  const auto bridge_executable = repo_root() / "openpilot/cereal/messaging/bridge";
  const auto previous_override = env_value("JOTP_STREAM_BRIDGE");
  const auto previous_setup_test = env_value("JOTP_STREAM_SETUP_TEST");
  setenv("JOTP_STREAM_BRIDGE", bridge_executable.c_str(), 1);
  setenv("JOTP_STREAM_SETUP_TEST", scratch.path().c_str(), 1);
  ScopedMsgqPrefix caller;
  caller.activate();
  Context context;
  auto pub = publisher(&context);
  REQUIRE(pub != nullptr);
  StreamPoller remote, local;
  remote.start(remote_source("127.0.0.2"), 5.0, "");
  const bool remote_held = wait_until(2s, [&] { return std::filesystem::exists(scratch.path() / "remote-held"); });
  const bool prefix_held = remote_held && env_value("OPENPILOT_PREFIX") != std::make_optional(caller.prefix());
  if (remote_held) local.start(local_source(), 5.0, "");
  const bool local_blocked = remote_held && wait_until(1s, [&] { return std::filesystem::exists(scratch.path() / "local-blocked"); });
  std::ofstream(scratch.path() / "release").put('\n');
  const bool both_connected = local_blocked && wait_until(5s, [&] { return remote.snapshot().connected && local.snapshot().connected; });
  restore_env("JOTP_STREAM_BRIDGE", previous_override);
  restore_env("JOTP_STREAM_SETUP_TEST", previous_setup_test);
  REQUIRE((remote_held && prefix_held && local_blocked));
  REQUIRE(both_connected);
  REQUIRE(std::filesystem::exists(scratch.path() / "local-after"));
  REQUIRE(env_value("OPENPILOT_PREFIX") == std::make_optional(caller.prefix()));
  REQUIRE(wait_until(2s, [&] { return send_state(pub.get(), true) > 0 && take_started(&local); }));
  remote.stop();
  local.stop();
  caller.restore();
  REQUIRE(env_value("OPENPILOT_PREFIX") == original);
  const auto pid_file = scratch.path() / "dying.pid";
  std::ofstream(script) << "#!/bin/sh\necho $$ > \"$1\"\nexec " << shell_quote(bridge_executable.string()) << " 127.0.0.2 \"$2\"\n";
  REQUIRE(chmod(script.c_str(), 0700) == 0);
  setenv("JOTP_STREAM_BRIDGE", script.c_str(), 1);
  StreamPoller dying;
  dying.start(remote_source(pid_file.string()), 5.0, "");
  const bool connected = wait_until(15s, [&] { return (dying.snapshot().connected && std::filesystem::exists(pid_file)) || !dying.snapshot().active; }) && dying.snapshot().connected;
  restore_env("JOTP_STREAM_BRIDGE", previous_override);
  REQUIRE_THAT(connected ? "" : take_error(&dying), Catch::Matchers::Equals(""));
  REQUIRE(connected);
  REQUIRE(kill(std::stoi(util::read_file(pid_file.string())), SIGKILL) == 0);
  REQUIRE(wait_until(3s, [&] { return !dying.snapshot().active; }));
  REQUIRE_FALSE(dying.snapshot().connected);
  REQUIRE_THAT(take_error(&dying), Catch::Matchers::Contains("terminated by signal 9"));
  REQUIRE(env_value("OPENPILOT_PREFIX") == original);
}
TEST_CASE("remote StreamPoller receives device data without displacing caller publishers") {
  const auto original = env_value("OPENPILOT_PREFIX");
  std::filesystem::path caller_path, device_path;
  {
    ScopedMsgqPrefix caller;
    caller_path = caller.path();
    caller.activate();
    Context caller_context;
    auto caller_pub = publisher(&caller_context);
    REQUIRE(caller_pub != nullptr);
    ScopedMsgqPrefix device;
    device_path = device.path();
    device.activate();
    StreamBridgeProcess device_bridge;
    device_bridge.start(repo_root() / "openpilot/cereal/messaging/bridge");
    Context device_context;
    auto device_pub = publisher(&device_context);
    REQUIRE(device_pub != nullptr);
    device.restore();
    StreamPoller poller;
    poller.start(remote_source("127.0.0.1"), 5.0, "");
    REQUIRE((wait_until(3s, [&] { return poller.snapshot().connected || !poller.snapshot().active; }) && poller.snapshot().connected));
    REQUIRE(std::string(std::getenv("OPENPILOT_PREFIX")) == caller.prefix());
    REQUIRE(send_state(caller_pub.get(), false) > 0);
    REQUIRE(wait_until(8s, [&] { return send_state(device_pub.get(), true) > 0 && take_started(&poller); }));
    poller.stop();
    REQUIRE(send_state(caller_pub.get(), false) > 0);
    device_bridge.stop();
    caller.restore();
  }
  REQUIRE_FALSE(std::filesystem::exists(caller_path));
  REQUIRE_FALSE(std::filesystem::exists(device_path));
  REQUIRE(env_value("OPENPILOT_PREFIX") == original);
}
