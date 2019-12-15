CC = clang
CXX = clang++

PHONELIBS = ../../phonelibs
BASEDIR = ../..

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS) \
          -I$(PHONELIBS)/android_frameworks_native/include \
          -I$(PHONELIBS)/android_system_core/include \
          -I$(PHONELIBS)/android_hardware_libhardware/include
CXXFLAGS = -std=c++11 -g -fPIC -O2 $(WARN_FLAGS) \
            -I$(PHONELIBS)/android_frameworks_native/include \
            -I$(PHONELIBS)/android_system_core/include \
            -I$(PHONELIBS)/android_hardware_libhardware/include

MESSAGING_FLAGS = -I$(BASEDIR)/selfdrive/messaging
MESSAGING_LIBS = $(BASEDIR)/selfdrive/messaging/messaging.a

# Sensord can only be compiled for the phone
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
EXTRA_LIBS += -lgnustl_shared


JSON_FLAGS = -I$(PHONELIBS)/json/src

DIAG_LIBS = -L/system/vendor/lib64 -ldiag -ltime_genoff

.PHONY: all
all: sensord gpsd

include ../common/cereal.mk

SENSORD_OBJS = sensors.o \
       ../common/swaglog.o \
       $(PHONELIBS)/json/src/json.o

GPSD_OBJS = gpsd.o \
       rawgps.o \
       ../common/swaglog.o \
       $(PHONELIBS)/json/src/json.o

DEPS := $(SENSORD_OBJS:.o=.d) $(GPSD_OBJS:.o=.d)

sensord: $(SENSORD_OBJS) $(MESSAGING_LIBS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(EXTRA_LIBS) \
            -lhardware

gpsd: $(GPSD_OBJS) $(MESSAGING_LIBS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(DIAG_LIBS) \
            $(EXTRA_LIBS) \
            -lhardware

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) \
           $(CEREAL_CXXFLAGS) \
           $(MESSAGING_FLAGS) \
           $(JSON_FLAGS) \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'


%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) \
           $(JSON_FLAGS) \
           $(ZMQ_FLAGS) \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f sensord gpsd $(OBJS) $(DEPS)

-include $(DEPS)
