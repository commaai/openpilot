#include <array>
#include <cassert>
#include <fstream>
#include <map>

#include "common/swaglog.h"
#include "common/util.h"
#include "system/hardware/hw.h"
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

const std::string INSTALL_PATH = "/data/openpilot";
const std::string VALID_CACHE_PATH = "/data/.openpilot_cache";

#define TMP_INSTALL_PATH "/data/tmppilot"

const int FONT_SIZE = 120;

extern const uint8_t str_continue[] asm("_binary_selfdrive_ui_installer_continue_openpilot_sh_start");
extern const uint8_t str_continue_end[] asm("_binary_selfdrive_ui_installer_continue_openpilot_sh_end");
extern const uint8_t inter_ttf[] asm("_binary_selfdrive_ui_installer_inter_ascii_ttf_start");
extern const uint8_t inter_ttf_end[] asm("_binary_selfdrive_ui_installer_inter_ascii_ttf_end");
extern const uint8_t inter_light_ttf[] asm("_binary_selfdrive_assets_fonts_Inter_Light_ttf_start");
extern const uint8_t inter_light_ttf_end[] asm("_binary_selfdrive_assets_fonts_Inter_Light_ttf_end");
extern const uint8_t inter_bold_ttf[] asm("_binary_selfdrive_assets_fonts_Inter_Bold_ttf_start");
extern const uint8_t inter_bold_ttf_end[] asm("_binary_selfdrive_assets_fonts_Inter_Bold_ttf_end");

Font font_inter;
Font font_roman;
Font font_display;

const bool tici_device = Hardware::get_device_type() == cereal::InitData::DeviceType::TICI ||
                         Hardware::get_device_type() == cereal::InitData::DeviceType::TIZI;

std::vector<std::string> tici_prebuilt_branches = {"release3", "release-tizi", "release3-staging", "nightly", "nightly-dev"};
std::string migrated_branch;

void branchMigration() {
  migrated_branch = BRANCH_STR;
  cereal::InitData::DeviceType device_type = Hardware::get_device_type();
  if (device_type == cereal::InitData::DeviceType::TICI) {
    if (std::find(tici_prebuilt_branches.begin(), tici_prebuilt_branches.end(), BRANCH_STR) != tici_prebuilt_branches.end()) {
      migrated_branch = "release-tici";
    } else if (BRANCH_STR == "master") {
      migrated_branch = "master-tici";
    }
  } else if (device_type == cereal::InitData::DeviceType::TIZI) {
    if (BRANCH_STR == "release3") {
      migrated_branch = "release-tizi";
    } else if (BRANCH_STR == "release3-staging") {
      migrated_branch = "release-tizi-staging";
    }
  } else if (device_type == cereal::InitData::DeviceType::MICI) {
    if (BRANCH_STR == "release3") {
      migrated_branch = "release-mici";
    } else if (BRANCH_STR == "release3-staging") {
      migrated_branch = "release-mici-staging";
    }
  }
}

void run(const char* cmd) {
  int err = std::system(cmd);
  assert(err == 0);
}

void finishInstall() {
  BeginDrawing();
    ClearBackground(BLACK);
    if (tici_device) {
      const char *m = "Finishing install...";
      int text_width = MeasureText(m, FONT_SIZE);
      DrawTextEx(font_display, m, (Vector2){(float)(GetScreenWidth() - text_width)/2 + FONT_SIZE, (float)(GetScreenHeight() - FONT_SIZE)/2}, FONT_SIZE, 0, WHITE);
    } else {
      DrawTextEx(font_display, "finishing setup", (Vector2){8, 10}, 82, 0, WHITE);
    }
  EndDrawing();
  util::sleep_for(60 * 1000);
}

void renderProgress(int progress) {
  BeginDrawing();
    ClearBackground(BLACK);
    if (tici_device) {
      DrawTextEx(font_inter, "Installing...", (Vector2){150, 290}, 110, 0, WHITE);
      Rectangle bar = {150, 570, (float)GetScreenWidth() - 300, 72};
      DrawRectangleRec(bar, (Color){41, 41, 41, 255});
      progress = std::clamp(progress, 0, 100);
      bar.width *= progress / 100.0f;
      DrawRectangleRec(bar, (Color){70, 91, 234, 255});
      DrawTextEx(font_inter, (std::to_string(progress) + "%").c_str(), (Vector2){150, 670}, 85, 0, WHITE);
    } else {
      DrawTextEx(font_display, "installing", (Vector2){8, 10}, 82, 0, WHITE);
      const std::string percent_str = std::to_string(progress) + "%";
      DrawTextEx(font_roman, percent_str.c_str(), (Vector2){6, (float)(GetScreenHeight() - 128 + 18)}, 128, 0,
                 (Color){255, 255, 255, (unsigned char)(255 * 0.9 * 0.35)});
    }

  EndDrawing();
}

int doInstall() {
  // wait for valid time
  while (!util::system_time_valid()) {
    util::sleep_for(500);
    LOGD("Waiting for valid time");
  }

  // cleanup previous install attempts
  run("rm -rf " TMP_INSTALL_PATH);

  // do the install
  if (util::file_exists(INSTALL_PATH) && util::file_exists(VALID_CACHE_PATH)) {
    return cachedFetch(INSTALL_PATH);
  } else {
    return freshClone();
  }
}

int freshClone() {
  LOGD("Doing fresh clone");
  std::string cmd = util::string_format("git clone --progress %s -b %s --depth=1 --recurse-submodules %s 2>&1",
                                        GIT_URL.c_str(), migrated_branch.c_str(), TMP_INSTALL_PATH);
  return executeGitCommand(cmd);
}

int cachedFetch(const std::string &cache) {
  LOGD("Fetching with cache: %s", cache.c_str());

  run(util::string_format("cp -rp %s %s", cache.c_str(), TMP_INSTALL_PATH).c_str());
  run(util::string_format("cd %s && git remote set-branches --add origin %s", TMP_INSTALL_PATH, migrated_branch.c_str()).c_str());

  renderProgress(10);

  return executeGitCommand(util::string_format("cd %s && git fetch --progress origin %s 2>&1", TMP_INSTALL_PATH, migrated_branch.c_str()));
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
  run(("git checkout " + migrated_branch).c_str());
  run(("git reset --hard origin/" + migrated_branch).c_str());
  run("git submodule update --init");

  // move into place
  run(("rm -f " + VALID_CACHE_PATH).c_str());
  run(("rm -rf " + INSTALL_PATH).c_str());
  run(util::string_format("mv %s %s", TMP_INSTALL_PATH, INSTALL_PATH.c_str()).c_str());

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
  run(("cd " + INSTALL_PATH + " && "
      "git remote set-url origin --push " GIT_SSH_URL " && "
      "git config --replace-all remote.origin.fetch \"+refs/heads/*:refs/remotes/origin/*\"").c_str());
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
  finishInstall();
}

int main(int argc, char *argv[]) {
  if (tici_device) {
    InitWindow(2160, 1080, "Installer");
  } else {
    InitWindow(536, 240, "Installer");
  }

  font_inter = LoadFontFromMemory(".ttf", inter_ttf, inter_ttf_end - inter_ttf, FONT_SIZE, NULL, 0);
  font_roman = LoadFontFromMemory(".ttf", inter_light_ttf, inter_light_ttf_end - inter_light_ttf, FONT_SIZE, NULL, 0);
  font_display = LoadFontFromMemory(".ttf", inter_bold_ttf, inter_bold_ttf_end - inter_bold_ttf, FONT_SIZE, NULL, 0);
  SetTextureFilter(font_inter.texture, TEXTURE_FILTER_BILINEAR);
  SetTextureFilter(font_roman.texture, TEXTURE_FILTER_BILINEAR);
  SetTextureFilter(font_display.texture, TEXTURE_FILTER_BILINEAR);

  branchMigration();

  if (util::file_exists(CONTINUE_PATH)) {
    finishInstall();
  } else {
    renderProgress(0);
    int result = doInstall();
    cloneFinished(result);
  }

  CloseWindow();
  UnloadFont(font_inter);
  UnloadFont(font_roman);
  UnloadFont(font_display);
  return 0;
}
