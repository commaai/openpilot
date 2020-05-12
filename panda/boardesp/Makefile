PATH := esp-open-sdk/xtensa-lx106-elf/bin:$(PATH)
CC = esp-open-sdk/xtensa-lx106-elf/bin/xtensa-lx106-elf-gcc
CFLAGS = -Iinclude/ -I. -I../ -mlongcalls -Iesp-open-sdk/ESP8266_NONOS_SDK_V1.5.4_16_05_20/driver_lib/include -std=c99 -DICACHE_FLASH
LDLIBS = -nostdlib -Wl,--start-group -lmain -lnet80211 -lwpa -llwip -lpp -lphy -Wl,--end-group -lgcc -ldriver -Wl,--gc-sections
LDFLAGS = -Teagle.app.v6.ld
OBJCP = esp-open-sdk/xtensa-lx106-elf/bin/xtensa-lx106-elf-objcopy
SDK_BASE = esp-open-sdk/ESP8266_NONOS_SDK_V1.5.4_16_05_20

ifeq ($(RELEASE),1)
  CERT = ../../pandaextra/certs/releaseesp
else
  CERT = ../certs/debugesp
  CFLAGS += "-DALLOW_DEBUG"
endif

flashall: user1.bin user2.bin
	../python/esptool.py write_flash 0 $(SDK_BASE)/bin/boot_v1.5.bin 0x01000 user1.bin 0x81000 user2.bin 0x3FE000 $(SDK_BASE)/bin/blank.bin

proxy-0x00000.bin: proxy
	../python/esptool.py elf2image $^

proxy: proxy.o elm327.o webserver.o sha.o

obj/proxy.o: proxy.c
	$(CC) $(CFLAGS) -c $^ -o $@

obj/elm327.o: elm327.c
	$(CC) $(CFLAGS) -c $^ -o $@

obj/webserver.o: webserver.c obj/cert.h obj/gitversion.h
	$(CC) $(CFLAGS) -c $< -o $@

obj/cert.h: ../crypto/getcertheader.py
	../crypto/getcertheader.py ../certs/debugesp.pub ../certs/releaseesp.pub > obj/cert.h

include ../common/version.mk

obj/sha.o: ../crypto/sha.c
	$(CC) $(CFLAGS) -c $^ -o $@

obj/rsa.o: ../crypto/rsa.c
	$(CC) $(CFLAGS) -c $^ -o $@

oldflash: proxy-0x00000.bin
	../python/esptool.py write_flash 0 proxy-0x00000.bin 0x40000 proxy-0x40000.bin

user1.bin: obj/proxy.o obj/elm327.o obj/webserver.o obj/sha.o obj/rsa.o
	$(CC) $(CFLAGS) $^ -o a.out -L$(SDK_BASE)/ld -T$(SDK_BASE)/ld/eagle.app.v6.new.1024.app1.ld $(LDLIBS)
	$(OBJCP) --only-section .text -O binary a.out eagle.app.v6.text.bin
	$(OBJCP) --only-section .data -O binary a.out eagle.app.v6.data.bin
	$(OBJCP) --only-section .rodata -O binary a.out eagle.app.v6.rodata.bin
	$(OBJCP) --only-section .irom0.text -O binary a.out eagle.app.v6.irom0text.bin
	COMPILE=gcc python2 ./esp-open-sdk/ESP8266_NONOS_SDK_V1.5.4_16_05_20/tools/gen_appbin.py a.out 2 0 32 4 0
	rm -f eagle.app.v6.*.bin
	mv eagle.app.flash.bin $@
	../crypto/sign.py $@ $@ $(CERT)

user2.bin: obj/proxy.o obj/elm327.o obj/webserver.o obj/sha.o obj/rsa.o
	$(CC) $(CFLAGS) $^ -o a.out -L$(SDK_BASE)/ld -T$(SDK_BASE)/ld/eagle.app.v6.new.1024.app2.ld $(LDLIBS)
	$(OBJCP) --only-section .text -O binary a.out eagle.app.v6.text.bin
	$(OBJCP) --only-section .data -O binary a.out eagle.app.v6.data.bin
	$(OBJCP) --only-section .rodata -O binary a.out eagle.app.v6.rodata.bin
	$(OBJCP) --only-section .irom0.text -O binary a.out eagle.app.v6.irom0text.bin
	COMPILE=gcc python2 ./esp-open-sdk/ESP8266_NONOS_SDK_V1.5.4_16_05_20/tools/gen_appbin.py a.out 2 0 32 4 0
	rm -f eagle.app.v6.*.bin
	mv eagle.app.flash.bin $@
	../crypto/sign.py $@ $@ $(CERT)

ota: user1.bin user2.bin
	curl http://192.168.0.10/espupdate1 --upload-file user1.bin
	curl http://192.168.0.10/espupdate2 --upload-file user2.bin

clean:
	rm -f proxy proxy.o proxy-0x00000.bin proxy-0x40000.bin eagle.app.* user1.bin user2.bin a.out obj/*
