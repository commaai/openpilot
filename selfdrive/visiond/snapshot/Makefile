CC = clang
CXX = clang++

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)

.PHONY: all
all: libvisionipc.so

visionipc.o: ../../common/visionipc.c ../../common/visionipc.h 
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -I../.. -I../../.. \
          -c -o '$@' ../../common/visionipc.c

ipc.o: ../../common/ipc.c ../../common/ipc.h 
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -I../.. -I../../.. \
          -c -o '$@' ../../common/ipc.c

libvisionipc.so: visionipc.o ipc.o
	$(CC) -shared -fPIC -o '$@' visionipc.o ipc.o

.PHONY: clean
clean:
	rm visionipc.o ipc.o libvisionipc.so
