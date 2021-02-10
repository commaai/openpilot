#include <time.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iostream>

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
  err = std::system("rm -rf /tmp/openpilot /data/openpilot");
  if (err) return 1;

  // Clone
  err = std::system("git clone " GIT_URL " -b " BRANCH " --depth=1 /tmp/openpilot");
  if (err) return 1;
  err = std::system("cd /tmp/openpilot && git submodule update --init");
  if (err) return 1;
  err = std::system("cd /tmp/openpilot && git remote set-url origin --push " GIT_SSH_URL);
  if (err) return 1;

  err = std::system("mv /tmp/openpilot /data");
  if (err) return 1;

#ifdef SSH_KEYS
  err = std::system("mkdir -p /data/params/d/");
  if (err) return 1;

  std::ofstream param;
  param.open("/data/params/d/GithubSshKeys");
  param << SSH_KEYS;
  param.close();
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
