#pragma once

#include <string>

class Setup {
public:
  Setup();
  void run();

private:
  void download(const char* url);
  void finished(const char* url, const char* error = "");

  void prevPage();
  void nextPage();

  // Screen handling methods
  void low_voltage();
  void getting_started();
  void network_setup();
  void software_selection();
  void downloading();
  void download_failed(const char* url, const char* error);

  int currentPage;
  std::string lastUrl;
  std::string lastError;
};