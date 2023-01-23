#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>

void hexdump(uint32_t *d, int l) {
  for (int i = 0; i < l; i++) {
    if (i%0x10 == 0 && i != 0) printf("\n");
    printf("%8x ", d[i]);
  }
  printf("\n");
}

/* Power cluster primary PLL */
#define C0_PLL_MODE         0x0
#define C0_PLL_L_VAL        0x4
#define C0_PLL_ALPHA        0x8
#define C0_PLL_USER_CTL    0x10
#define C0_PLL_CONFIG_CTL  0x18
#define C0_PLL_CONFIG_CTL_HI 0x1C
#define C0_PLL_STATUS      0x28
#define C0_PLL_TEST_CTL_LO 0x20
#define C0_PLL_TEST_CTL_HI 0x24

/* Power cluster alt PLL */
#define C0_PLLA_MODE        0x100
#define C0_PLLA_L_VAL       0x104
#define C0_PLLA_ALPHA       0x108
#define C0_PLLA_USER_CTL    0x110
#define C0_PLLA_CONFIG_CTL  0x118
#define C0_PLLA_STATUS      0x128
#define C0_PLLA_TEST_CTL_LO 0x120

#define APC_DIAG_OFFSET 0x48
#define CLK_CTL_OFFSET 0x44
#define MUX_OFFSET 0x40
#define MDD_DROOP_CODE 0x7c
#define SSSCTL_OFFSET 0x160
#define PSCTL_OFFSET 0x164

int main() {
    int fd = open("/dev/mem", O_RDWR);
    if(fd < 0) {
        printf("Error: Unable to open /dev/mem. Check permissions.\n");
        return -1;
    }
    volatile uint32_t *mb = (uint32_t*)mmap(0, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x06400000);
    if(mb == MAP_FAILED) {
        printf("Error: mmap failed.\n");
        close(fd);
        return -1;
    }
    volatile uint32_t *mc = (uint32_t*)mmap(0, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x06480000);
    if(mc == MAP_FAILED) {
        printf("Error: mmap failed.\n");
        munmap((void *)mb, 0x1000);
        close(fd);
        return -1;
    }
    volatile uint32_t *md = (uint32_t*)mmap(0, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x09A20000);
if(md == MAP_FAILED) {
printf("Error: mmap failed.\n");
munmap((void *)mb, 0x1000);
munmap((void *)mc, 0x1000);
close(fd);
return -1;
}
while (1) {
printf("PLL MODE:%x L_VAL:%x ALPHA:%x USER_CTL:%x CONFIG_CTL:%x CONFIG_CTL_HI:%x STATUS:%x TEST_CTL_LO:%x TEST_CTL_HI:%x\n",
mb[C0_PLL_MODE/4], mb[C0_PLL_L_VAL/4], mb[C0_PLL_ALPHA/4],
mb[C0_PLL_USER_CTL/4], mb[C0_PLL_CONFIG_CTL/4], mb[C0_PLL_CONFIG_CTL_HI/4],
mb[C0_PLL_STATUS/4], mb[C0_PLL_TEST_CTL_LO/4], mb[C0_PLL_TEST_CTL_HI/4]);
printf(" MUX_OFFSET:%x CLK_CTL_OFFSET:%x APC_DIAG_OFFSET:%x MDD_DROOP_CODE:%x\n",
mb[MUX_OFFSET/4], mb[CLK_CTL_OFFSET/4], mb[APC_DIAG_OFFSET/4], mb[MDD_DROOP_CODE/4]);
printf(" PLLA MODE:%x L_VAL:%x ALPHA:%x USER_CTL:%x CONFIG_CTL:%x STATUS:%x TEST_CTL_LO:%x\n",
mc[C0_PLLA_MODE/4], mc[C0_PLLA_L_VAL/4], mc[C0_PLLA_ALPHA/4],
mc[C0_PLLA_USER_CTL/4], mc[C0_PLLA_CONFIG_CTL/4], mc[C0_PLLA_STATUS/4], mc[C0_PLLA_TEST_CTL_LO/4]);
printf(" SSSCTL_OFFSET:%x PSCTL_OFFSET:%x\n", md[SSSCTL_OFFSET/4], md[PSCTL_OFFSET/4]);
}
munmap((void *)mb, 0x1000);
munmap((void *)mc, 0x1000);
munmap((void *)md, 0x1000);
close(fd);
return 0;
}




