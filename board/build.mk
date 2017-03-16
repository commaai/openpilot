CFLAGS += -I inc -nostdlib
CFLAGS += -Tstm32_flash.ld

CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
OBJDUMP = arm-none-eabi-objdump

MACHINE = $(shell uname -m)

all: obj/$(PROJ_NAME).bin
	#$(OBJDUMP) -d obj/$(PROJ_NAME).elf
	./tools/enter_download_mode.py
	./tools/dfu-util-$(MACHINE) -a 0 -s 0x08000000 -D $<
	./tools/dfu-util-$(MACHINE) --reset-stm32 -a 0 -s 0x08000000

ifneq ($(wildcard ../.git/HEAD),) 
obj/gitversion.h: ../.git/HEAD ../.git/index
	echo "const uint8_t gitversion[] = \"$(shell git rev-parse HEAD)\";" > $@
else
obj/gitversion.h: 
	echo "const uint8_t gitversion[] = \"RELEASE\";" > $@
endif

obj/main.$(PROJ_NAME).o: main.c *.h obj/gitversion.h
	$(CC) $(CFLAGS) -o $@ -c $<

obj/$(STARTUP_FILE).o: $(STARTUP_FILE).s
	mkdir -p obj
	$(CC) $(CFLAGS) -o $@ -c $<

obj/$(PROJ_NAME).bin: obj/$(STARTUP_FILE).o obj/main.$(PROJ_NAME).o
	$(CC) $(CFLAGS) -o obj/$(PROJ_NAME).elf $^
	$(OBJCOPY) -v -O binary obj/$(PROJ_NAME).elf $@
	
clean:
	rm -f obj/*

