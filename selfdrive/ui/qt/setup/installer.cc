#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>

#ifndef BRANCH
#define BRANCH "master"
#endif

#define GIT_URL "https://github.com/commaai/openpilot.git"
#define GIT_SSH_URL "git@github.com:commaai/openpilot.git"

#define CONTINUE_PATH "/data/continue.sh"

bool time_valid() {
  time_t rawtime;
  time(&rawtime);

  struct tm * sys_time = gmtime(&rawtime);
  return (1900 + sys_time->tm_year) >= 2019;
}

int fresh_clone() {
  int err;

  // Cleanup
  err = std::system("rm -rf /data/tmppilot /data/openpilot");
  if (err) return 1;

  // Clone
  err = std::system("git clone " GIT_URL " -b " BRANCH " --depth=1 --recurse-submodules /data/tmppilot");
  if (err) return 1;
  err = std::system("cd /data/tmppilot && git remote set-url origin --push " GIT_SSH_URL);
  if (err) return 1;

  err = std::system("mv /data/tmppilot /data/openpilot");
  if (err) return 1;

#ifdef INTERNAL
  err = std::system("mkdir -p /data/params/d/");
  if (err) return 1;

  std::map<std::string, std::string> params = {
    {"SshEnabled", "1"},
    {"RecordFrontLock", "1"},
    {"GithubSshKeys", SSH_KEYS},
  };
  for (const auto& [key, value] : params) {
    std::ofstream param;
    param.open("/data/params/d/RecordFrontLock" + key);
    param << value;
    param.close();
  }
#endif

  return 0;
}

int install() {
  int err;

  // Wait for valid time
  while (!time_valid()) {
    usleep(500 * 1000);
    std::cout << "Waiting for valid time\n";
  }

  std::cout << "Doing fresh clone\n";
  err = fresh_clone();
  if (err) return 1;

  // Write continue.sh
  err = std::system("cp /data/openpilot/installer/continue_openpilot.sh " CONTINUE_PATH);
  if (err == -1) return 1;

  return 0;
}

int main(int argc, char *argv[]) {
  // TODO: make a small installation UI
  return install();
}
