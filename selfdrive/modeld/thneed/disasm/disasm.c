#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#define uint uint32_t
#define bool char
#define u64 uint64_t
#include "adreno_pm4types.h"

void parse_cmd_packet(uint32_t *src, int len) {
  int i = 0;
  while (i < len) {
		int pktsize;
    int pkttype = -1;

		if (pkt_is_type0(src[i])) {
      pkttype = 0;
			pktsize = type0_pkt_size(src[i]);
		} else if (pkt_is_type3(src[i])) {
      pkttype = 3;
			pktsize = type3_pkt_size(src[i]);
		} else if (pkt_is_type4(src[i])) {
      pkttype = 4;
      pktsize = type4_pkt_size(src[i]);
    } else if (pkt_is_type7(src[i])) {
      pkttype = 7;
      pktsize = type7_pkt_size(src[i]);
    }
    printf("%3d: type:%d size:%d\n", i, pkttype, pktsize);

    if (pkttype == -1) break;
    i += (1+pktsize);
  }
  assert(i == len);
}

int main() {
  FILE *f = fopen("../runs/run_3_0", "rb");

  uint64_t ll;
  int pkt = 0;
  while (fread(&ll, 1, 8, f) == 8) {
    printf("got packet with length %d\n", ll);
    uint32_t *dat = malloc(ll);
    fread(dat, 1, ll, f);
    if (pkt == 0 || pkt == 1) {
      parse_cmd_packet(dat, ll/4);
    }
    pkt++;
    free(dat);
  }

  fclose(f);
}

