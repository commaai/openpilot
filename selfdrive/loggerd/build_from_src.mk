CC = clang
CXX = clang++

ARCH := $(shell uname -m)

PHONELIBS = ../../phonelibs
BASEDIR = ../..

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args \
             -Wno-deprecated-declarations

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS) \
          -I$(PHONELIBS)/android_frameworks_native/include \
          -I$(PHONELIBS)/android_system_core/include \
          -I$(PHONELIBS)/android_hardware_libhardware/include
CXXFLAGS = -std=c++11 -g -fPIC -O2 $(WARN_FLAGS) \
            -I$(PHONELIBS)/android_frameworks_native/include \
            -I$(PHONELIBS)/android_system_core/include \
            -I$(PHONELIBS)/android_hardware_libhardware/include

ZMQ_LIBS = -l:libczmq.a -l:libzmq.a

MESSAGING_FLAGS = -I$(BASEDIR)/selfdrive/messaging
MESSAGING_LIBS = $(BASEDIR)/selfdrive/messaging/messaging.a

ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
ZMQ_LIBS += -lgnustl_shared
endif


BZIP_FLAGS = -I$(PHONELIBS)/bzip2/
BZIP_LIBS = -L$(PHONELIBS)/bzip2/ \
            -l:libbz2.a

# todo: dont use system ffmpeg libs
FFMPEG_LIBS = -lavformat \
              -lavcodec \
              -lswscale \
              -lavutil \
              -lz

LIBYUV_FLAGS = -I$(PHONELIBS)/libyuv/include
LIBYUV_LIBS = $(PHONELIBS)/libyuv/lib/libyuv.a

OPENMAX_FLAGS = -I$(PHONELIBS)/openmax/include
OPENMAX_LIBS = -lOmxVenc -lOmxCore

JSON_FLAGS = -I$(PHONELIBS)/json/src

YAML_FLAGS = -I$(PHONELIBS)/yaml-cpp/include
YAML_LIBS = $(PHONELIBS)/yaml-cpp/lib/libyaml-cpp.a

.PHONY: all
all: loggerd

include ../common/cereal.mk

OBJS += loggerd.o \
       logger.o \
       ../common/util.o \
       ../common/params.o \
       ../common/cqueue.o \
       ../common/swaglog.o \
       ../common/visionipc.o \
       ../common/ipc.o \
       $(PHONELIBS)/json/src/json.o

ifeq ($(ARCH),x86_64)
CXXFLAGS += "-DDISABLE_ENCODER"
ZMQ_LIBS = -L$(BASEDIR)/external/zmq/lib/ \
           -l:libczmq.a -l:libzmq.a
EXTRA_LIBS = -lpthread
OPENMAX_LIBS = ""
YAML_LIBS = $(PHONELIBS)/yaml-cpp/x64/lib/libyaml-cpp.a
else
OBJS += encoder.o \
        raw_logger.o
EXTRA_LIBS = -lcutils -llog -lgnustl_shared
endif

DEPS := $(OBJS:.o=.d)

loggerd: $(OBJS) $(MESSAGING_LIBS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
	      $(LIBYUV_LIBS) \
        $(CEREAL_LIBS) \
        $(ZMQ_LIBS) \
        -L/usr/lib \
        $(FFMPEG_LIBS) \
        -L/system/vendor/lib64 \
        $(OPENMAX_LIBS) \
        $(YAML_LIBS) \
        $(EXTRA_LIBS) \
        $(BZIP_LIBS) \
        -lm

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           $(CEREAL_CXXFLAGS) \
           $(LIBYUV_FLAGS) \
           $(ZMQ_FLAGS) \
           $(MESSAGING_FLAGS) \
           $(OPENMAX_FLAGS) \
           $(YAML_FLAGS) \
           $(BZIP_FLAGS) \
           -Iinclude \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
           $(LIBYUV_FLAGS) \
           $(ZMQ_FLAGS) \
           $(OPENMAX_FLAGS) \
           $(JSON_FLAGS) \
           $(BZIP_FLAGS) \
           -Iinclude \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f loggerd $(OBJS) $(DEPS)

-include $(DEPS)
