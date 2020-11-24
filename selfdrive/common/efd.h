#ifndef EFD_H
#define EFD_H

#ifdef __cplusplus
extern "C" {
#endif

// event fd: a semaphore that can be poll()'d
int efd_init();
void efd_write(int fd);
void efd_clear(int fd);

#ifdef __cplusplus
}
#endif

#endif