ARCH := $(shell uname -m)

CC = clang
CXX = clang++

PHONELIBS = ../../phonelibs
BASEDIR = ../..

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++11 -g -fPIC -O2 $(WARN_FLAGS)

ZMQ_LIBS = -l:libczmq.a -l:libzmq.a

ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
ZMQ_LIBS += -lgnustl_shared
endif

#ZMQ_FLAGS = -I$(PHONELIBS)/zmq/aarch64/include

.PHONY: all
all: proclogd

include ../common/cereal.mk

OBJS = proclogd.o \
       $(CEREAL_OBJS)

DEPS := $(OBJS:.o=.d)

proclogd: $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(ZMQ_LIBS) \
            -llog

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) \
           $(CEREAL_CXXFLAGS) \
           $(ZMQ_FLAGS) \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f proclogd $(OBJS) $(DEPS)

-include $(DEPS)
