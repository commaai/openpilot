/*
gcc -DTEST_RSA test_rsa.c ../crypto/rsa.c ../crypto/sha.c && ./a.out
*/

#include <stdio.h>
#include <stdlib.h>

#define MAX_LEN 0x40000
char buf[MAX_LEN];

#include "../crypto/sha.h"
#include "../crypto/rsa.h"
#include "../obj/cert.h"

int main() {
  FILE *f = fopen("../obj/panda.bin", "rb");
  int tlen = fread(buf, 1, MAX_LEN, f);
  fclose(f);
  printf("read %d\n", tlen);
  uint32_t *_app_start = (uint32_t *)buf;

  int len = _app_start[0];
  char digest[SHA_DIGEST_SIZE];
  SHA_hash(&_app_start[1], len-4, digest);
  printf("SHA hash done\n");

  if (!RSA_verify(&rsa_key, ((void*)&_app_start[0]) + len, RSANUMBYTES, digest, SHA_DIGEST_SIZE)) {
    printf("RSA fail\n");
  } else {
    printf("RSA match!!!\n");
  }

  return 0;
}

