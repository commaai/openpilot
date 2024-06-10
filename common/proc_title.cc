#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

extern char **environ;

char* argv = nullptr;
int len_argv;

void _init(int argc, char* argv0) {
  char *ptr = environ[0] - 1;
  char *limit = ptr - 2048;

  for ( int i = argc - 1; i >= 1; i-- ) {
    ptr--;
    while ( *ptr && ptr > limit ) {
      ptr--;
    }

    *ptr = ' ';

    if ( ptr <= limit ) {
      return;
    }
  }

  ptr = ptr - strlen(argv0);

  if ( strncmp(ptr, argv0, strlen(argv0)) ) {
    return;
  }

  argv = ptr;
  len_argv = environ[0] - 1 - ptr;
}

void setProcTitle(char* new_title) {
  if (argv == nullptr) {
    return;
  }

  int new_title_len = strlen(new_title);

  if ( new_title_len > len_argv ) {
    new_title_len = len_argv - 1;
  }

  strncpy(argv, new_title, new_title_len);
  memset(argv + new_title_len, 0, len_argv - new_title_len);

  return;
}

std::string getProcTitle() {
  if (argv == nullptr) {
    return "";
  }

  return argv;
}
