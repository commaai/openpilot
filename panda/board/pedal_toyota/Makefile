# :set noet
PROJ_NAME = comma

CFLAGS  = -O2 -Wall -std=gnu11 -DPEDAL
CFLAGS += -mlittle-endian -mthumb -mcpu=cortex-m3
CFLAGS += -msoft-float -DSTM32F2 -DSTM32F205xx
CFLAGS += -I ../inc -I ../ -I ../../ -nostdlib
CFLAGS += -T../stm32_flash.ld

STARTUP_FILE = startup_stm32f205xx

CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
OBJDUMP = arm-none-eabi-objdump
DFU_UTIL = "dfu-util"

# pedal only uses the debug cert
CERT = ../../certs/debug
CFLAGS += "-DALLOW_DEBUG"

canflash: obj/$(PROJ_NAME).bin
	../../tests/pedal/enter_canloader.py $<

usbflash: obj/$(PROJ_NAME).bin
	../../tests/pedal/enter_canloader.py; sleep 0.5
	PYTHONPATH=../../ python -c "from python import Panda; p = [x for x in [Panda(x) for x in Panda.list()] if x.bootstub]; assert(len(p)==1); p[0].flash('obj/$(PROJ_NAME).bin', reconnect=False)"

recover: obj/bootstub.bin obj/$(PROJ_NAME).bin
	../../tests/pedal/enter_canloader.py --recover; sleep 0.5
	$(DFU_UTIL) -d 0483:df11 -a 0 -s 0x08004000 -D obj/$(PROJ_NAME).bin
	$(DFU_UTIL) -d 0483:df11 -a 0 -s 0x08000000:leave -D obj/bootstub.bin

obj/main.o: main.c ../*.h
	mkdir -p obj
	$(CC) $(CFLAGS) -o $@ -c $<

obj/bootstub.o: ../bootstub.c ../*.h
	mkdir -p obj
	$(CC) $(CFLAGS) -o $@ -c $<

obj/$(STARTUP_FILE).o: ../$(STARTUP_FILE).s
	$(CC) $(CFLAGS) -o $@ -c $<

obj/%.o: ../../crypto/%.c
	$(CC) $(CFLAGS) -o $@ -c $<

obj/$(PROJ_NAME).bin: obj/$(STARTUP_FILE).o obj/main.o
  # hack
	$(CC) -Wl,--section-start,.isr_vector=0x8004000 $(CFLAGS) -o obj/$(PROJ_NAME).elf $^
	$(OBJCOPY) -v -O binary obj/$(PROJ_NAME).elf obj/code.bin
	SETLEN=1 ../../crypto/sign.py obj/code.bin $@ $(CERT)

obj/bootstub.bin: obj/$(STARTUP_FILE).o obj/bootstub.o obj/sha.o obj/rsa.o
	$(CC) $(CFLAGS) -o obj/bootstub.$(PROJ_NAME).elf $^
	$(OBJCOPY) -v -O binary obj/bootstub.$(PROJ_NAME).elf $@
	
clean:
	rm -f obj/*

