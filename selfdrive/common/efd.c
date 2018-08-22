#include <stdlib.h>
#include <assert.h>

#ifdef __linux__
#include <sys/eventfd.h>
#else
#include <sys/time.h>
#include <sys/event.h>
#define EVENT_IDENT 42
#endif

#include "efd.h"


int efd_init() {
#ifdef __linux__
  return eventfd(0, EFD_CLOEXEC);
#else
  int fd = kqueue();
  assert(fd >= 0);

  struct kevent kev;
  EV_SET(&kev, EVENT_IDENT, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, NULL);

  struct timespec timeout = {0, 0};
  int err = kevent(fd, &kev, 1, NULL, 0, &timeout);
  assert(err != -1);

  return fd;
#endif
}

void efd_write(int fd) {
#ifdef __linux__
  eventfd_write(fd, 1);
#else
  struct kevent kev;
  EV_SET(&kev, EVENT_IDENT, EVFILT_USER, 0, NOTE_TRIGGER, 0, NULL);

  struct timespec timeout = {0, 0};
  int err = kevent(fd, &kev, 1, NULL, 0, &timeout);
  assert(err != -1);
#endif
}

void efd_clear(int fd) {
#ifdef __linux__
  eventfd_t efd_cnt;
  eventfd_read(fd, &efd_cnt);
#else
  struct kevent kev;
  struct timespec timeout = {0, 0};
  int nfds = kevent(fd, NULL, 0, &kev, 1, &timeout);
  assert(nfds != -1);
#endif
}
