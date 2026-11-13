#include <emscripten.h>

#include <cstdio>
#include <memory>
#include <string>

#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/ui/app.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/utils/util.h"
#include "tools/cabana/web/web.h"

// clang-format off
// a JWT arrives once as #jwt=<token> and is then remembered by the browser
EM_JS(char *, readApiToken, (), {
  const m = location.hash.match(/jwt=([^&]+)/);
  if (m) {
    localStorage.setItem("cabana_jwt", m[1]);
    history.replaceState(null, "", location.pathname + location.search);
  }
  return stringToNewUTF8(localStorage.getItem("cabana_jwt") || "");
});

EM_JS(char *, readQueryParam, (const char *name), {
  return stringToNewUTF8(new URLSearchParams(location.search).get(UTF8ToString(name)) || "");
});
// clang-format on

static std::string takeString(char *s) {
  std::string out(s);
  free(s);
  return out;
}

int main(int argc, char *argv[]) {
  web::setApiToken(takeString(readApiToken()));

  // ?route=<name> opens straight into a route, like the native command line
  const std::string route = takeString(readQueryParam("route"));
  const std::string dbc = takeString(readQueryParam("dbc"));
  StreamLoader loader;
  if (!route.empty()) {
    loader = [route]() -> std::unique_ptr<AbstractStream> {
      auto stream = std::make_unique<ReplayStream>();
      Connection err = stream->error.connect([](const std::string &msg) {
        fprintf(stderr, "%s\n", msg.c_str());
        utils::runOnMainThread([msg]() { MessageBox::warning("Error", msg); });
      });
      if (!stream->loadRoute(route, "", REPLAY_FLAG_QCAMERA)) return nullptr;
      return stream;
    };
  }
  return run(nullptr, std::move(loader), dbc);
}
