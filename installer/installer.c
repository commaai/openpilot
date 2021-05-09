#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#include "selfdrive/common/framebuffer.h"
#include "selfdrive/common/glutil.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/spinner.h"
#include "selfdrive/common/util.h"

#ifndef BRAND
#define BRAND openpilot
#endif

#define STR(X) #X
#define STR2(X) STR(X)
#define PASTE(A, B) A ## B
#define PASTE2(A, B) PASTE(A, B)
#define BRAND_S STR2(BRAND)
#define BRANCH_S STR2(BRANCH)

#define PRE_CHECKOUT_FOLDER "/system/comma/openpilot"
#define GIT_CLONE_COMMAND "git clone https://github.com/commaai/openpilot.git "


extern const uint8_t str_continue[] asm("_binary_continue_" BRAND_S "_sh_start");
extern const uint8_t str_continue_end[] asm("_binary_continue_" BRAND_S "_sh_end");

static bool time_valid() {
  time_t rawtime;
  time(&rawtime);

  struct tm * sys_time = gmtime(&rawtime);
  return (1900 + sys_time->tm_year) >= 2019;
}

static int use_pre_checkout() {
  int err;

  // Cleanup
  err = system("rm -rf /tmp/openpilot");
  if(err) return 1;
  err = system("rm -rf /data/openpilot");
  if(err) return 1;

  // Copy pre checkout into tmp so we can work on it
  err = system("cp -rp " PRE_CHECKOUT_FOLDER " /tmp");
  if(err) return 1;

  err = chdir("/tmp/openpilot");
  if(err) return 1;

  // Checkout correct branch
  err = system("git remote set-branches --add origin " BRANCH_S);
  if(err) return 1;
  err = system("git fetch origin " BRANCH_S);
  if(err) return 1;
  err = system("git checkout " BRANCH_S);
  if(err) return 1;
  err = system("git reset --hard origin/" BRANCH_S);
  if(err) return 1;

  // Move to final location
  err = system("mv /tmp/openpilot /data");
  if(err) return 1;

  return 0;
}

static int fresh_clone() {
  int err;

  // Cleanup
  err = chdir("/tmp");
  if(err) return 1;
  err = system("rm -rf /tmp/openpilot");
  if(err) return 1;

  err = system(GIT_CLONE_COMMAND " -b " BRANCH_S " --depth=1 openpilot");
  if(err) return 1;

  // Cleanup old folder in /data
  err = system("rm -rf /data/openpilot");
  if(err) return 1;

  // this won't move if /data/openpilot exists
  err = system("mv /tmp/openpilot /data");
  if(err) return 1;

  return 0;
}

static int do_install() {
  int err;


  // Wait for valid time
  while (!time_valid()){
    usleep(500 * 1000);
    printf("Waiting for valid time\n");
  }

  struct stat sb;
  if (stat(PRE_CHECKOUT_FOLDER, &sb) == 0 && S_ISDIR(sb.st_mode)) {
    printf("Pre-checkout found\n");
    err = use_pre_checkout();
  } else {
    printf("Doing fresh clone\n");
    err = fresh_clone();
  }
  if(err) return 1;


  // Write continue.sh
  FILE *of = fopen("/data/data/com.termux/files/continue.sh.new", "wb");
  if(of == NULL) return 1;

  size_t num = str_continue_end - str_continue;
  size_t num_written = fwrite(str_continue, 1, num, of);
  if (num != num_written) return 1;

  fclose(of);

  err = system("chmod +x /data/data/com.termux/files/continue.sh.new");
  if(err) return 1;

  err = rename("/data/data/com.termux/files/continue.sh.new", "/data/data/com.termux/files/continue.sh");
  if(err == -1) return 1;

  // Disable SSH
  err = system("setprop persist.neos.ssh 0");
  if(err) return 1;

  return 0;
}


void * run_spinner(void * args) {
  char *loading_msg = "Installing " BRAND_S;
  char *argv[2] = {NULL, loading_msg};
  spin(2, argv);
  return NULL;
}


int main() {
  pthread_t spinner_thread;
  int err = pthread_create(&spinner_thread, NULL, run_spinner, NULL);
  assert(err == 0);

  int status = do_install();

  return status;
}
