#include "selfdrive/ui/raylib/setup.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "third_party/raylib/include/raylib.h"

#include <curl/curl.h>

#include "common/util.h"
#include "system/hardware/hw.h"
#include "selfdrive/ui/raylib/api.h"
#include "selfdrive/ui/raylib/window.h"
#include "selfdrive/ui/raylib/networking.h"
#include "selfdrive/ui/raylib/util.h"
#include "selfdrive/ui/raylib/widgets/input.h"

const std::string USER_AGENT = "AGNOSSetup-";
const char* OPENPILOT_URL = "https://openpilot.comma.ai";

bool is_elf(const char *fname) {
  FILE *fp = fopen(fname, "rb");
  if (fp == NULL) {
    return false;
  }
  char buf[4];
  size_t n = fread(buf, 1, 4, fp);
  fclose(fp);
  return n == 4 && buf[0] == 0x7f && buf[1] == 'E' && buf[2] == 'L' && buf[3] == 'F';
}

void Setup::download(const char* url) {
  // autocomplete incomplete urls
  std::string full_url = url;
  if (full_url.find("://") == std::string::npos) {
    full_url = "https://installer.comma.ai/" + full_url;
  }

  CURL *curl = curl_easy_init();
  if (!curl) {
    finished(url, "Something went wrong. Reboot the device.");
    return;
  }

  auto version = util::read_file("/VERSION");

  struct curl_slist *list = NULL;
  list = curl_slist_append(list, ("X-openpilot-serial: " + Hardware::get_serial()).c_str());

  char tmpfile[] = "/tmp/installer_XXXXXX";
  FILE *fp = fdopen(mkstemp(tmpfile), "wb");

  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, (USER_AGENT + version).c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

  int ret = curl_easy_perform(curl);
  long res_status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &res_status);

  if (ret != CURLE_OK || res_status != 200) {
    finished(url, "Ensure the entered URL is valid, and the device's internet connection is good.");
  } else if (!is_elf(tmpfile)) {
    finished(url, "No custom software found at this URL.");
  } else {
    rename(tmpfile, "/tmp/installer");

    FILE *fp_url = fopen("/tmp/installer_url", "w");
    fprintf(fp_url, "%s", url);
    fclose(fp_url);

    finished(url);
  }

  curl_slist_free_all(list);
  curl_easy_cleanup(curl);
  fclose(fp);
}

void Setup::low_voltage() {
  DrawText("WARNING: Low Voltage", 55, 144, 90, RED);
  DrawText("Power your device in a car with a harness or proceed at your own risk.", 55, 234, 80, WHITE);

  if (GuiButton((Rectangle){55, GetScreenHeight() - 160, (GetScreenWidth() - 160) / 2, 160}, "Power off")) {
    Hardware::poweroff();
  }
  if (GuiButton((Rectangle){GetScreenWidth() / 2 + 55, GetScreenHeight() - 160, (GetScreenWidth() - 160) / 2, 160}, "Continue")) {
    nextPage();
  }
}

void Setup::getting_started() {
  DrawText("Getting Started", 165, 280, 90, WHITE);
  DrawText("Before we get on the road, let's finish installation and cover some details.", 165, 370, 80, WHITE);

  if (GuiButton((Rectangle){GetScreenWidth() - 310, 0, 310, GetScreenHeight()}, "")) {
    nextPage();
  }
}

void Setup::network_setup() {
  DrawText("Connect to Wi-Fi", 55, 50, 90, WHITE);

  // TODO: Implement Networking widget in raylib

  if (GuiButton((Rectangle){55, GetScreenHeight() - 160, (GetScreenWidth() - 160) / 2, 160}, "Back")) {
    prevPage();
  }
  if (GuiButton((Rectangle){GetScreenWidth() / 2 + 55, GetScreenHeight() - 160, (GetScreenWidth() - 160) / 2, 160}, "Continue")) {
    nextPage();
  }
}

void Setup::software_selection() {
  DrawText("Choose Software to Install", 55, 50, 90, WHITE);

  // TODO: Implement radio buttons in raylib

  if (GuiButton((Rectangle){55, GetScreenHeight() - 160, (GetScreenWidth() - 160) / 2, 160}, "Back")) {
    prevPage();
  }
  if (GuiButton((Rectangle){GetScreenWidth() / 2 + 55, GetScreenHeight() - 160, (GetScreenWidth() - 160) / 2, 160}, "Continue")) {
    // TODO: Implement software selection logic
  }
}

void Setup::downloading() {
  DrawText("Downloading...", GetScreenWidth() / 2 - MeasureText("Downloading...", 90) / 2, GetScreenHeight() / 2 - 45, 90, WHITE);
}

void Setup::download_failed(const char* url, const char* error) {
  DrawText("Download Failed", 55, 185, 90, WHITE);
  DrawText(url, 55, 275, 64, WHITE);
  DrawText(error, 55, 339, 80, WHITE);

  if (GuiButton((Rectangle){55, GetScreenHeight() - 160, (GetScreenWidth() - 160) / 2, 160}, "Reboot device")) {
    Hardware::reboot();
  }
  if (GuiButton((Rectangle){GetScreenWidth() / 2 + 55, GetScreenHeight() - 160, (GetScreenWidth() - 160) / 2, 160}, "Start over")) {
    currentPage = 1;
  }
}

void Setup::prevPage() {
  if (currentPage > 0) currentPage--;
}

void Setup::nextPage() {
  currentPage++;
}

Setup::Setup() {
  InitWindow(1920, 1080, "Setup");
  SetTargetFPS(60);

  std::stringstream buffer;
  buffer << std::ifstream("/sys/class/hwmon/hwmon1/in1_input").rdbuf();
  float voltage = (float)std::atoi(buffer.str().c_str()) / 1000.;
  if (voltage < 7) {
    currentPage = 0;
  } else {
    currentPage = 1;
  }
}

void Setup::run() {
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    switch (currentPage) {
      case 0: low_voltage(); break;
      case 1: getting_started(); break;
      case 2: network_setup(); break;
      case 3: software_selection(); break;
      case 4: downloading(); break;
      case 5: download_failed(lastUrl.c_str(), lastError.c_str()); break;
    }

    EndDrawing();
  }

  CloseWindow();
}

void Setup::finished(const char* url, const char* error) {
  if (error[0] == '\0') {
    // hide setup on success
    CloseWindow();
  } else {
    lastUrl = url;
    lastError = error;
    currentPage = 5;
  }
}

int main() {
  Setup setup;
  setup.run();
  return 0;
}
