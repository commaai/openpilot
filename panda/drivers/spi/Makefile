obj-m += spidev_panda.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# GCC9 bug, apply kernel patch instead?
# https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=0b999ae3614d09d97a1575936bcee884f912b10e 
ccflags-y := -Wno-missing-attributes

default:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
