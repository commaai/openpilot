CC = clang
CXX = clang++

ARCH := $(shell uname -m)

PHONELIBS = ../../phonelibs
BASEDIR = ../..

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++11 -g -fPIC -O2 $(WARN_FLAGS)

MESSAGING_FLAGS = -I$(BASEDIR)/selfdrive/messaging
MESSAGING_LIBS = $(BASEDIR)/selfdrive/messaging/messaging.a

ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
EXTRA_LIBS += -lgnustl_shared
endif


.PHONY: all
all: logcatd

include ../common/cereal.mk

OBJS = logcatd.o \
       $(CEREAL_OBJS)

DEPS := $(OBJS:.o=.d)

logcatd: $(OBJS) $(MESSAGING_LIBS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(EXTRA_LIBS) \
            -llog

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) \
           -I$(PHONELIBS)/android_system_core/include \
           $(CEREAL_CXXFLAGS) \
           $(MESSAGING_FLAGS) \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f logcatd $(OBJS) $(DEPS)

-include $(DEPS)
