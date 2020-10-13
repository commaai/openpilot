CC = clang
CXX = clang++

PHONELIBS = ../../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++1z -g -fPIC -O2 $(WARN_FLAGS)

ZMQ_FLAGS = -I$(PHONELIBS)/zmq/aarch64/include
ZMQ_LIBS = -L$(PHONELIBS)/zmq/aarch64/lib \
           -l:libzmq.a \
           -lgnustl_shared

NANOVG_FLAGS = -I$(PHONELIBS)/nanovg
JSON_FLAGS = -I$(PHONELIBS)/json/src

OPENGL_LIBS = -lGLESv3

FRAMEBUFFER_LIBS = -lutils -lgui -lEGL

OBJS = test.o \
       ../../common/framebuffer.o \
       ../../common/touch.o

DEPS := $(OBJS:.o=.d)

all: test

test: $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
	      $(FRAMEBUFFER_LIBS) \
	      $(CEREAL_LIBS) \
	      $(ZMQ_LIBS) \
        -L/system/vendor/lib64 \
        -lhardware \
        $(OPENGL_LIBS) \
        -lcutils -lm -llog

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -Iinclude -I.. -I../.. \
           -I$(PHONELIBS)/android_frameworks_native/include \
           -I$(PHONELIBS)/android_system_core/include \
           -I$(PHONELIBS)/android_hardware_libhardware/include \
           -c -o '$@' '$<'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -I.. -I../.. \
          $(NANOVG_FLAGS) \
          $(ZMQ_FLAGS) \
          $(CEREAL_CFLAGS) \
          $(JSON_FLAGS) \
          -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f ui $(OBJS) $(DEPS)

-include $(DEPS)
