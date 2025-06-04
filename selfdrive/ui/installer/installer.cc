#include <array>
#include <cassert>
#include <fstream>
#include <map>

#include "common/swaglog.h"
#include "common/util.h"
#include "third_party/raylib/include/raylib.h"

int freshClone();
int cachedFetch(const std::string &cache);
int executeGitCommand(const std::string &cmd);

std::string get_str(std::string const s) {
  std::string::size_type pos = s.find('?');
  assert(pos != std::string::npos);
  return s.substr(0, pos);
}

// Leave some extra space for the fork installer
const std::string GIT_URL = get_str("https://github.com/commaai/openpilot.git" "?                                                                ");
const std::string BRANCH_STR = get_str(BRANCH "?                                                                ");

#define GIT_SSH_URL "git@github.com:commaai/openpilot.git"
#define CONTINUE_PATH "/data/continue.sh"

const std::string CACHE_PATH = "/data/openpilot.cache";

#define INSTALL_PATH "/data/openpilot"
#define TMP_INSTALL_PATH "/data/tmppilot"

extern const uint8_t str_continue[] asm("_binary_selfdrive_ui_installer_continue_openpilot_sh_start");
extern const uint8_t str_continue_end[] asm("_binary_selfdrive_ui_installer_continue_openpilot_sh_end");
extern const uint8_t inter_ttf[] asm("_binary_selfdrive_ui_installer_inter_ascii_ttf_start");
extern const uint8_t inter_ttf_end[] asm("_binary_selfdrive_ui_installer_inter_ascii_ttf_end");

Font font;

void run(const char* cmd) {
  int err = std::system(cmd);
  assert(err == 0);
}

void renderProgress(int progress) {
  BeginDrawing();
    ClearBackground(BLACK);
    DrawTextEx(font, "Installing...", (Vector2){150, 290}, 110, 0, WHITE);
    Rectangle bar = {150, 570, (float)GetScreenWidth() - 300, 72};
    DrawRectangleRec(bar, (Color){41, 41, 41, 255});
    progress = std::clamp(progress, 0, 100);
    bar.width *= progress / 100.0f;
    DrawRectangleRec(bar, (Color){70, 91, 234, 255});
    DrawTextEx(font, (std::to_string(progress) + "%").c_str(), (Vector2){150, 670}, 85, 0, WHITE);
  EndDrawing();
}

int doInstall() {
  // wait for valid time
  while (!util::system_time_valid()) {
    util::sleep_for(500);
    LOGD("Waiting for valid time");
  }

  // cleanup previous install attempts
  run("rm -rf " TMP_INSTALL_PATH " " INSTALL_PATH);

  // do the install
  if (util::file_exists(CACHE_PATH)) {
    return cachedFetch(CACHE_PATH);
  } else {
    return freshClone();
  }
}

int freshClone() {
  LOGD("Doing fresh clone");
  std::string cmd = util::string_format("git clone --progress %s -b %s --depth=1 --recurse-submodules %s 2>&1",
                                        GIT_URL.c_str(), BRANCH_STR.c_str(), TMP_INSTALL_PATH);
  return executeGitCommand(cmd);
}

int cachedFetch(const std::string &cache) {
  LOGD("Fetching with cache: %s", cache.c_str());

  run(util::string_format("cp -rp %s %s", cache.c_str(), TMP_INSTALL_PATH).c_str());
  run(util::string_format("cd %s && git remote set-branches --add origin %s", TMP_INSTALL_PATH, BRANCH_STR.c_str()).c_str());

  renderProgress(10);

  return executeGitCommand(util::string_format("cd %s && git fetch --progress origin %s 2>&1", TMP_INSTALL_PATH, BRANCH_STR.c_str()));
}

int executeGitCommand(const std::string &cmd) {
  static const std::array stages = {
    // prefix, weight in percentage
    std::pair{"Receiving objects: ", 91},
    std::pair{"Resolving deltas: ", 2},
    std::pair{"Updating files: ", 7},
  };

  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) return -1;

  char buffer[512];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    std::string line(buffer);
    int base = 0;
    for (const auto &[text, weight] : stages) {
      if (line.find(text) != std::string::npos) {
        size_t percentPos = line.find("%");
        if (percentPos != std::string::npos && percentPos >= 3) {
          int percent = std::stoi(line.substr(percentPos - 3, 3));
          int progress = base + int(percent / 100. * weight);
          renderProgress(progress);
        }
        break;
      }
      base += weight;
    }
  }
  return pclose(pipe);
}

void cloneFinished(int exitCode) {
  LOGD("git finished with %d", exitCode);
  assert(exitCode == 0);

  renderProgress(100);

  // ensure correct branch is checked out
  int err = chdir(TMP_INSTALL_PATH);
  assert(err == 0);
  run(("git checkout " + BRANCH_STR).c_str());
  run(("git reset --hard origin/" + BRANCH_STR).c_str());
  run("git submodule update --init");

  // move into place
  run("mv " TMP_INSTALL_PATH " " INSTALL_PATH);

#ifdef INTERNAL
  run("mkdir -p /data/params/d/");

  // https://github.com/commaci2.keys
  const std::string ssh_keys = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMX2kU8eBZyEWmbq0tjMPxksWWVuIV/5l64GabcYbdpI";
  std::map<std::string, std::string> params = {
    {"SshEnabled", "1"},
    {"RecordFrontLock", "1"},
    {"GithubSshKeys", ssh_keys},
  };
  for (const auto& [key, value] : params) {
    std::ofstream param;
    param.open("/data/params/d/" + key);
    param << value;
    param.close();
  }
  run("cd " INSTALL_PATH " && "
      "git remote set-url origin --push " GIT_SSH_URL " && "
      "git config --replace-all remote.origin.fetch \"+refs/heads/*:refs/remotes/origin/*\"");
#endif

  // write continue.sh
  FILE *of = fopen("/data/continue.sh.new", "wb");
  assert(of != NULL);

  size_t num = str_continue_end - str_continue;
  size_t num_written = fwrite(str_continue, 1, num, of);
  assert(num == num_written);
  fclose(of);

  run("chmod +x /data/continue.sh.new");
  run("mv /data/continue.sh.new " CONTINUE_PATH);

  // wait for the installed software's UI to take over
  util::sleep_for(60 * 1000);
}

int main(int argc, char *argv[]) {
  InitWindow(2160, 1080, "Installer");
  font = LoadFontFromMemory(".ttf", inter_ttf, inter_ttf_end - inter_ttf, 120, NULL, 0);
  SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
  renderProgress(0);
  int result = doInstall();
  cloneFinished(result);
  CloseWindow();
  UnloadFont(font);
  return 0;
}
