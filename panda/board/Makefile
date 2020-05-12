PROJ_NAME = panda
CFLAGS = -g -Wall -Wextra -Wstrict-prototypes -Werror

CFLAGS += -mlittle-endian -mthumb -mcpu=cortex-m4
CFLAGS += -mhard-float -DSTM32F4 -DSTM32F413xx -mfpu=fpv4-sp-d16 -fsingle-precision-constant
STARTUP_FILE = startup_stm32f413xx

include build.mk
