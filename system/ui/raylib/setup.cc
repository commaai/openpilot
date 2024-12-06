#include "system/ui/raylib/setup.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "third_party/raylib/include/raylib.h"

#include <curl/curl.h>

#include "common/util.h"
#include "system/hardware/hw.h"
// #include "selfdrive/ui/qt/api.h"
// #include "selfdrive/ui/raylib/networking.h"
#include "system/ui/raylib/util.h"

const std::string USER_AGENT = "AGNOSSetup-";
const char* OPENPILOT_URL = "https://openpilot.comma.ai";
typedef enum Screen { LOW_VOLTAGE, GETTING_STARTED, NETWORK_SETUP, SOFTWARE_SELECTION, DOWNLOADING, DOWNLOAD_FAILED } Screen;

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
  DrawText("WARNING: Low Voltage", 55, 144, 90, COLOR_RED);
  DrawText("Power your device in a car with a harness or proceed at your own risk.", 55, 234, 80, COLOR_WHITE);

  // Define button rectangles
  Rectangle powerOffBtn = {55.0f, static_cast<float>(GetScreenHeight() - 160),
                         static_cast<float>((GetScreenWidth() - 160) / 2), 160.0f};
  Rectangle continueBtn = {static_cast<float>(GetScreenWidth()) / 2 + 55.0f,
                         GetScreenHeight() - 160.0f,
                         static_cast<float>(GetScreenWidth() - 160) / 2, 160.0f};
  // Draw buttons
  DrawRectangleRec(powerOffBtn, CheckCollisionPointRec(GetMousePosition(), powerOffBtn) ? COLOR_DARKGRAY : COLOR_GRAY);
  DrawRectangleRec(continueBtn, CheckCollisionPointRec(GetMousePosition(), continueBtn) ? COLOR_DARKGRAY : COLOR_GRAY);

  // Draw button text
  DrawText("Power off", powerOffBtn.x + powerOffBtn.width/2 - MeasureText("Power off", 40)/2,
           powerOffBtn.y + powerOffBtn.height/2 - 20, 40, COLOR_WHITE);
  DrawText("Continue", continueBtn.x + continueBtn.width/2 - MeasureText("Continue", 40)/2,
           continueBtn.y + continueBtn.height/2 - 20, 40, COLOR_WHITE);

  // Handle button clicks
  if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    Vector2 mousePos = GetMousePosition();
    if (CheckCollisionPointRec(mousePos, powerOffBtn)) {
      Hardware::poweroff();
    } else if (CheckCollisionPointRec(mousePos, continueBtn)) {
      nextPage();
    }
  }
}

void Setup::getting_started() {
  // Draw header text
  DrawText("Getting Started", 55, 144, 90, COLOR_WHITE);
  DrawText("Before we get on the road, let's finish installation and cover some details.", 55, 234, 80, COLOR_WHITE);

  // Define continue button rectangle
  Rectangle continueBtn = {static_cast<float>(GetScreenWidth()) / 2 + 55.0f, GetScreenHeight() - 160.0f, (GetScreenWidth() - 160.0f) / 2, 160.0f};

  // Draw button
  DrawRectangleRec(continueBtn, CheckCollisionPointRec(GetMousePosition(), continueBtn) ? COLOR_DARKGRAY : COLOR_GRAY);

  // Draw button text
  DrawText("Continue", continueBtn.x + continueBtn.width/2 - MeasureText("Continue", 40)/2,
           continueBtn.y + continueBtn.height/2 - 20, 40, COLOR_WHITE);

  // Handle button click
  if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    Vector2 mousePos = GetMousePosition();
    if (CheckCollisionPointRec(mousePos, continueBtn)) {
      nextPage();
    }
  }
}

void Setup::network_setup() {
  DrawText("Connect to Wi-Fi", 55, 50, 90, COLOR_WHITE);

  // Draw network list container
  Rectangle networkList = {55.0f, 160.0f,
                         static_cast<float>(GetScreenWidth() - 110),
                         static_cast<float>(GetScreenHeight() - 380)};
  DrawRectangleRec(networkList, COLOR_DARKGRAY);

  // Example networks (replace with actual network scanning)
  const char* networks[] = {
    "WiFi Network 1",
    "Home Network",
    "Guest Network",
    "Office WiFi"
  };
  int networkCount = sizeof(networks) / sizeof(networks[0]);

  // Draw network items
  for (int i = 0; i < networkCount; i++) {
    Rectangle networkItem = {
      networkList.x + 10,
      networkList.y + 10 + (i * 60),
      networkList.width - 20,
      50
    };

    // Highlight on hover
    bool isHovered = CheckCollisionPointRec(GetMousePosition(), networkItem);
    DrawRectangleRec(networkItem, isHovered ? COLOR_GRAY : COLOR_DARKGRAY);

    // Draw network name
    DrawText(networks[i], networkItem.x + 20, networkItem.y + 15, 20, COLOR_WHITE);

    // Handle network selection
    if (isHovered && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
      // TODO: Connect to selected network
      // For now, just print selected network
      printf("Selected network: %s\n", networks[i]);
    }
  }

  // Navigation buttons
  Rectangle backBtn = {55, GetScreenHeight() - 160.0f, (GetScreenWidth() - 160.0f) / 2, 160.0f};
  Rectangle continueBtn = {static_cast<float>(GetScreenWidth()) / 2 + 55.0f, GetScreenHeight() - 160.0f, (GetScreenWidth() - 160.0f) / 2, 160.0f};

  // Draw buttons
  DrawRectangleRec(backBtn, CheckCollisionPointRec(GetMousePosition(), backBtn) ? COLOR_DARKGRAY : COLOR_GRAY);
  DrawRectangleRec(continueBtn, CheckCollisionPointRec(GetMousePosition(), continueBtn) ? COLOR_DARKGRAY : COLOR_GRAY);

  // Draw button text
  DrawText("Back", backBtn.x + backBtn.width/2 - MeasureText("Back", 40)/2,
           backBtn.y + backBtn.height/2 - 20, 40, COLOR_WHITE);
  DrawText("Continue", continueBtn.x + continueBtn.width/2 - MeasureText("Continue", 40)/2,
           continueBtn.y + continueBtn.height/2 - 20, 40, COLOR_WHITE);

  // Handle button clicks
  if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    Vector2 mousePos = GetMousePosition();
    if (CheckCollisionPointRec(mousePos, backBtn)) {
      prevPage();
    } else if (CheckCollisionPointRec(mousePos, continueBtn)) {
      nextPage();
    }
  }
}

void Setup::software_selection() {
  DrawText("Choose Software to Install", 55, 50, 90, COLOR_WHITE);

  // Navigation buttons
  Rectangle backBtn = {55, GetScreenHeight() - 160.0f, (GetScreenWidth() - 160.0f) / 2, 160.0f};
  Rectangle continueBtn = {static_cast<float>(GetScreenWidth()) / 2 + 55.0f, GetScreenHeight() - 160.0f, (GetScreenWidth() - 160.0f) / 2, 160.0f};

  // Draw buttons
  DrawRectangleRec(backBtn, CheckCollisionPointRec(GetMousePosition(), backBtn) ? COLOR_DARKGRAY : COLOR_GRAY);
  DrawRectangleRec(continueBtn, CheckCollisionPointRec(GetMousePosition(), continueBtn) ? COLOR_DARKGRAY : COLOR_GRAY);

  // Draw button text
  DrawText("Back", backBtn.x + backBtn.width/2 - MeasureText("Back", 40)/2,
           backBtn.y + backBtn.height/2 - 20, 40, COLOR_WHITE);
  DrawText("Continue", continueBtn.x + continueBtn.width/2 - MeasureText("Continue", 40)/2,
           continueBtn.y + continueBtn.height/2 - 20, 40, COLOR_WHITE);

  // Handle button clicks
  if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    Vector2 mousePos = GetMousePosition();
    if (CheckCollisionPointRec(mousePos, backBtn)) {
      prevPage();
    } else if (CheckCollisionPointRec(mousePos, continueBtn)) {
      // TODO: Implement software selection logic
    }
  }
}

void Setup::downloading() {
  DrawText("Downloading...", static_cast<float>(GetScreenWidth()) / 2 - MeasureText("Downloading...", 90) / 2, static_cast<float>(GetScreenHeight()) / 2 - 45, 90, COLOR_WHITE);
}

void Setup::download_failed(const char* url, const char* error) {
  DrawText("Download Failed", 55, 185, 90, COLOR_WHITE);
  DrawText(url, 55, 275, 64, COLOR_WHITE);
  DrawText(error, 55, 339, 80, COLOR_WHITE);

  // Define button rectangles
  Rectangle rebootBtn = {55, GetScreenHeight() - 160.0f, (GetScreenWidth() - 160.0f) / 2, 160.0f};
  Rectangle startOverBtn = {static_cast<float>(GetScreenWidth()) / 2 + 55.0f, GetScreenHeight() - 160.0f, (GetScreenWidth() - 160.0f) / 2, 160.0f};

  // Draw buttons
  DrawRectangleRec(rebootBtn, CheckCollisionPointRec(GetMousePosition(), rebootBtn) ? COLOR_DARKGRAY : COLOR_GRAY);
  DrawRectangleRec(startOverBtn, CheckCollisionPointRec(GetMousePosition(), startOverBtn) ? COLOR_DARKGRAY : COLOR_GRAY);

  // Draw button text
  DrawText("Reboot device", rebootBtn.x + rebootBtn.width/2 - MeasureText("Reboot device", 40)/2,
           rebootBtn.y + rebootBtn.height/2 - 20, 40, COLOR_WHITE);
  DrawText("Start over", startOverBtn.x + startOverBtn.width/2 - MeasureText("Start over", 40)/2,
           startOverBtn.y + startOverBtn.height/2 - 20, 40, COLOR_WHITE);

  // Handle button clicks
  if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    Vector2 mousePos = GetMousePosition();
    if (CheckCollisionPointRec(mousePos, rebootBtn)) {
      Hardware::reboot();
    } else if (CheckCollisionPointRec(mousePos, startOverBtn)) {
      currentPage = GETTING_STARTED;
    }
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
    currentPage = LOW_VOLTAGE;
  } else {
    currentPage = GETTING_STARTED;
  }
}

void Setup::run() {
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(COLOR_BLACK);

    switch (currentPage) {
      case LOW_VOLTAGE: low_voltage(); break;
      case GETTING_STARTED: getting_started(); break;
      case NETWORK_SETUP: network_setup(); break;
      case SOFTWARE_SELECTION: software_selection(); break;
      case DOWNLOADING: downloading(); break;
      case DOWNLOAD_FAILED: download_failed(lastUrl.c_str(), lastError.c_str()); break;
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
    currentPage = DOWNLOAD_FAILED;
  }
}

int main() {
  Setup setup;
  setup.run();
  return 0;
}